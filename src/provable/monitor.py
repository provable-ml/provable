"""provable.monitor -- the watch that runs after a performance certificate.

A performance certificate from a holdout bounds a model's mean loss, or its
mean paired gain over the sealed baseline, on the law the holdout sampled.
After the model is deployed the law can move. watch() opens a monitor on a
performance certificate registered in the book and charges its level to the
book; update() appends forward rows, the outcomes and the predictions issued
before them, and runs an e-detector against the certified level. The first
time the evidence reaches 1/delta the watch raises an alarm and stops.

  watch(cert, ledger, delta=0.05, horizon=None)
                                   -> Watch       bind to a certificate of the
                                                  book and charge delta
  w.update(y, p, times=, baseline=) -> WatchState append forward rows and run
                                                  the detector over them
  w.close(cause)                   -> WatchState  end the watch; the charge
                                                  stays
  w.audit()                        -> WatchAudit  replay the detector from the
                                                  stored rows

What is theorem and what is design:
  theorem  If, given the rows before it, every monitored row has a mean loss
           at most the certified level (a mean paired gain at least the
           certified bound), the watch raises an alarm, at any row of an
           unbounded horizon or of a declared horizon of T rows, with
           probability at most delta. The evidence is a mixture, over the
           row s at which a change may start (weight 1/(s(s+1)), or 1/T for
           s <= T under a horizon) and over a grid of bets lambda, of the
           betting
           processes prod_r (1 + lambda (u_r - m)) on the loss scaled to
           [0, 1], m the certified level on that scale. Under the hypothesis
           the mixture is a nonnegative supermartingale that starts at 1, and
           Ville's inequality bounds the chance that it ever reaches 1/delta.
           Cited: e-detectors (Shin, Ramdas and Rinaldo), betting processes
           for bounded variables (Waudby-Smith and Ramdas), Ville's
           inequality.
  design   The bet grid lambda = 2^-k / m for k = 1..8 (so that every factor
           is at least 1/2), uniform weight over the grid, the start weights
           (a horizon T trades the anytime guarantee for a smaller price on
           late changes: log T in place of 2 log s), the declared loss
           range, and the start estimate: the
           start row of the largest single-start process (the CUSUM argmax
           over the grid). How fast an alarm follows a change is not a
           theorem here; scenario C22 measures it.
  meaning  An alarm says that, from some row on, the live mean loss of the
           monitored predictions exceeded the certified level. It does not
           void the certificate, which remains a statement about the law the
           holdout sampled. That the predictions are the certified model's
           and were issued before their outcomes is the analyst's
           undertaking: the watch checks the clock (rows after the certified
           rows, in order) and the loss, not the model.
  range    The detector needs bounded losses. A certificate sealed with a
           loss range is watched on that range; brier and zero_one are
           bounded; logloss is bounded by its probability clip; squared and
           absolute losses need watch(loss_range=(lo, hi)) with lo at most
           the loss's natural minimum, so that clipping only lowers losses
           and an alarm on the clipped loss is an alarm on the loss.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from .certificates.records import CertificateKind, content_id, to_dict
from .certificates.store import Ledger
from . import holdout as _ho
from .holdout import BINARY, EPOCH, LOSSES, NATURAL, _canon, _losses, _npy_bytes, _npy_load, _op_hash, _quiet, _sha, _write_atomic

GRID = tuple(2.0 ** -k for k in range(1, 9))
STORE = "watches"
STATUSES = ("watching", "alarm", "ended", "closed")


def _lg(x: float) -> Optional[float]:
    return None if x == -math.inf else float(x)


def _ug(x: Optional[float]) -> float:
    return -math.inf if x is None else float(x)


def _log_weights(s: np.ndarray, horizon: Optional[int]) -> tuple:
    """The log prior weight of a change starting at each row s, and the log
    of the weight of the starts after s (the processes not yet begun, still
    at 1): 1/(s(s+1)) and 1/(s+1) without a horizon; 1/T and (T - s)/T with
    a horizon T."""
    if horizon is None:
        return -(np.log(s) + np.log1p(s)), -np.log1p(s)
    T = float(horizon)
    return np.full(len(s), -math.log(T)), np.log(np.maximum(T - s, 0.0)) - math.log(T)


@_quiet
def _advance(det: dict, u: np.ndarray, m: float, log_thr: float, horizon: Optional[int] = None) -> tuple:
    """Run the detector over the scaled rows u from its state det (rows seen,
    and per bet the log of the start-weighted sum A, and the log of the
    largest single-start process B, unweighted, with its start row: the
    CUSUM statistic and its argmax). Returns the new state, the
    rows used (all of them, or up to and including the alarm row), the alarm
    offset (or None) and the log evidence of every row used. The caller
    passes no rows beyond the horizon."""
    t0, n = int(det["rows"]), len(u)
    s = t0 + 1.0 + np.arange(n, dtype=float)
    logw, lrest = _log_weights(s, horizon)
    la, lb, st = np.empty((len(GRID), n)), np.empty((len(GRID), n)), np.empty((len(GRID), n), dtype=np.int64)
    for k, c in enumerate(GRID):
        cum = np.cumsum(np.log1p((c / m) * (u - m)))
        prev = np.concatenate([[0.0], cum[:-1]])
        acc = np.logaddexp.accumulate(logw - prev)
        a0 = _ug(det["logA"][k])
        la[k] = cum + (acc if a0 == -math.inf else np.logaddexp(a0, acc))
        run = np.maximum.accumulate(-prev)
        new = np.maximum.accumulate(np.where(-prev >= run, s, 0.0)).astype(np.int64)
        b0 = _ug(det["logB"][k])
        keep = b0 >= run
        lb[k] = cum + np.where(keep, b0, run)
        st[k] = np.where(keep, int(det["start"][k]), new)
    ev = np.logaddexp(np.logaddexp.reduce(la, axis=0) - math.log(len(GRID)), lrest)
    hit = np.nonzero(ev >= log_thr)[0]
    used = n if hit.size == 0 else int(hit[0]) + 1
    j = used - 1
    out = {"rows": t0 + used, "logA": [_lg(x) for x in la[:, j]], "logB": [_lg(x) for x in lb[:, j]], "start": [int(x) for x in st[:, j]], "evidence": float(ev[j]), "peak": max(float(det["peak"]), float(ev[:used].max()))}
    return out, used, (None if hit.size == 0 else int(hit[0])), ev[:used]


def _start_of(det: dict) -> int:
    k = int(np.argmax([_ug(x) for x in det["logB"]]))
    return int(det["start"][k])


def _times_ns(times, n: int) -> np.ndarray:
    """The rows' times as int64 nanoseconds since the epoch, UTC (a naive
    time is read as UTC), checked for count, gaps and order."""
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(times)) if not isinstance(times, (pd.Series, pd.Index)) else pd.Series(times), utc=True))
    if len(t) != n:
        raise ValueError(f"update refused: {len(t)} times for {n} rows")
    if t.isna().any():
        raise ValueError("update refused: a time is missing")
    out = np.asarray(t.as_unit("ns").asi8, dtype=np.int64)
    if np.any(np.diff(out) < 0):
        raise ValueError("update refused: the times go backwards; rows arrive in time order")
    return out


def _iso(ns: Optional[int]) -> Optional[str]:
    return None if ns is None else pd.Timestamp(int(ns), tz="UTC").isoformat()


@dataclass
class WatchState:
    """Where a watch stands after an operation: rows monitored, the mean of
    the monitored statistic against the certified level, the evidence
    against the threshold 1/delta, and the alarm, with the estimated row at
    which the breach started. path holds the evidence of every row of the
    last update (up to the alarm row), or None."""
    watch: str
    certificate: str
    claim: str
    loss: str
    level: float
    delta: float
    horizon: Optional[int]
    rows: int
    received: int
    mean: Optional[float]
    evidence: float
    peak: float
    status: str
    alarm: Optional[dict]
    start: Optional[int]
    last_time: Optional[str]
    path: Optional[np.ndarray] = None

    @property
    def threshold(self) -> float:
        return 1.0 / self.delta

    def render(self) -> str:
        what = "mean loss" if self.claim == "risk" else "mean paired gain"
        rel = "at most" if self.claim == "risk" else "at least"
        head = [["watch", self.watch], ["certificate", self.certificate], ["claim", f"{what} {rel} {self.level:.6g} ({self.loss} loss)"], ["rows monitored", f"{self.rows}" + (f" of {self.received} received" if self.received != self.rows else "") + ("" if self.horizon is None else f" (horizon {self.horizon})")], [what, "" if self.mean is None else f"{self.mean:.6g}"], ["evidence (alarm at 1/delta)", f"{math.exp(min(self.evidence, 700.0)):.4g} of {self.threshold:.4g}"], ["status", self.status]]
        if self.alarm is not None:
            head.append(["alarm", f"row {self.alarm['row']}" + (f" ({self.alarm['time']})" if self.alarm.get("time") else "") + f"; the evidence points to a breach starting near row {self.alarm['start']}"])
        if self.last_time is not None:
            head.append(["last row's time", self.last_time])
        return "\n".join([f"-- watch {self.watch} --", tabulate(head, tablefmt="psql")])

    def __str__(self):
        return self.render()


@dataclass
class WatchAudit:
    """The replay of a watch from its log and its row files: one row per
    check, PASS or FAIL, and the overall verdict."""
    watch: str
    rows: list
    ok: bool

    def render(self) -> str:
        out = [f"-- audit: {self.watch} --", tabulate([[r[0], r[1], str(r[2])[:90]] for r in self.rows], headers=["item", "verdict", "detail"], tablefmt="psql")]
        out.append(f"audit {'PASS' if self.ok else 'FAIL'}: {sum(1 for r in self.rows if r[1] == 'PASS')} of {len(self.rows)} items recomputed and matched")
        return "\n".join(out)

    def __str__(self):
        return self.render()


def watch(certificate, ledger, *, delta: float = 0.05, horizon: Optional[int] = None, loss_range=None, label: str = "") -> "Watch":
    """Open a watch on a performance certificate of the book and charge its
    level delta to the book's account before any forward row is seen. The
    certificate must be registered in this book and active. The monitored
    statistic is the certificate's loss (risk claim) or the model's loss
    minus the baseline's (gain claim), on the certificate's declared loss
    range; an unbounded loss sealed without a range needs loss_range=(lo, hi)
    here, with lo at most the loss's natural minimum. Without a horizon the
    guarantee covers every row the watch will ever take; with horizon=T it
    covers T rows, the watch ends after them, and a change late in the
    stretch is caught sooner."""
    if ledger is None:
        raise ValueError("watch refused: a watch lives in the book of its certificate; pass ledger= (a Ledger or a book directory)")
    led = ledger if isinstance(ledger, Ledger) else Ledger(str(ledger))
    kind = certificate.kind.value if isinstance(certificate.kind, CertificateKind) else str(certificate.kind)
    if kind != CertificateKind.PERFORMANCE.value:
        raise ValueError(f"watch refused: a watch monitors a performance certificate (a holdout's bound on a model's loss); this certificate is {kind}")
    delta = float(delta)
    if not (0.0 < delta < 0.5):
        raise ValueError("watch refused: delta must lie in (0, 0.5)")
    if horizon is not None:
        if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 1:
            raise ValueError("watch refused: horizon is a number of rows, at least 1")
        horizon = int(horizon)
    fam, art = certificate.guarantee.family, certificate.artifact
    claim, loss = fam.get("claim"), fam.get("loss")
    if claim not in ("risk", "gain") or loss not in LOSSES or "bound" not in art:
        raise ValueError("watch refused: the certificate carries no claim form, loss or bound to watch")
    nat_lo, nat_hi = NATURAL[loss]
    sealed = fam.get("loss_range")
    if sealed is not None:
        lo, hi = float(sealed[0]), float(sealed[1])
        if loss_range is not None and (float(loss_range[0]), float(loss_range[1])) != (lo, hi):
            raise ValueError(f"watch refused: the certificate was sealed with the loss range {[lo, hi]}; its claim is about losses clipped to that range, so the watch uses the same one")
    elif loss_range is not None:
        lo, hi = float(loss_range[0]), float(loss_range[1])
        if not hi > lo:
            raise ValueError("watch refused: loss_range must be (lo, hi) with hi > lo")
        if lo > nat_lo:
            raise ValueError(f"watch refused: loss_range starts at {lo:g}, above the {loss} loss's natural minimum {nat_lo:g}; clipping must only lower losses, so that an alarm on the clipped loss is an alarm on the loss")
    elif math.isfinite(nat_hi):
        lo, hi = float(nat_lo), float(nat_hi)
    else:
        raise ValueError(f"watch refused: the {loss} loss is unbounded and the certificate declared no range; the detector needs bounded losses, so pass loss_range=(0, hi), and losses above hi are clipped to it")
    level = float(art["bound"])
    if claim == "risk":
        a, b, m0 = lo, hi, level
    else:
        a, b, m0 = -(hi - lo), hi - lo, -level
    m = (m0 - a) / (b - a)
    if not m > 0.0:
        raise ValueError(f"watch refused: the certified level {level:.6g} sits at or beyond the {'bottom' if claim == 'risk' else 'top'} of the range; no monitored row can be shown to breach it")
    cid = content_id(to_dict(certificate))
    t1s = [s.t1 for s in certificate.ell_test.slices]
    after = None
    if t1s:
        t1 = max(pd.Timestamp(t).tz_localize("UTC") if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t).tz_convert("UTC") for t in t1s)
        if t1 > pd.Timestamp(EPOCH):
            after = int(t1.value)
    with led.locked():
        reg = {o["id"] for o in led._ops if o["op"] == "register"}
        if cid not in reg:
            raise ValueError(f"watch refused: certificate {cid} is not registered in {led.path}; a watch charges the book its certificate lives in")
        if cid in led._voided_ids():
            raise ValueError(f"watch refused: certificate {cid} is voided")
        rem = led.statement().remaining
        if rem + 1e-12 < delta:
            raise ValueError(f"watch refused: a watch at delta {delta:g} charges the book, and {led.path} has {rem:.4f} remaining; raise the book's budget or watch at a level it can carry")
        state = {"status": "watching", "rows": 0, "received": 0, "sum": 0.0, "logA": [None] * len(GRID), "logB": [None] * len(GRID), "start": [0] * len(GRID), "evidence": 0.0, "peak": 0.0, "alarm": None, "clocked": None, "last_time": None}
        op = {"op": "watch", "action": "open", "seq": 0, "prev": None, "at": _ho._now(), "certificate": cid, "holdout": art.get("holdout"), "model_sha": art.get("model_sha"), "claim": claim, "loss": loss, "range": [lo, hi], "stat_range": [a, b], "level": level, "m": m, "delta": delta, "horizon": horizon, "grid": list(GRID), "after": after, "label": str(label), "state": state}
        wid = "WA-" + _sha(_canon(op))[:16].upper()
        op["watch"] = wid
        root = os.path.join(led.path, STORE, wid)
        os.makedirs(root, exist_ok=True)
        led._append(op)
        _write_atomic(os.path.join(root, "head.json"), _canon({"seq": 0, "hash": _op_hash(led.watch_ops(wid)[-1])}))
    return Watch(led, wid)


class Watch:
    """A watch in a book. Construct with Watch(ledger, id) to reopen one;
    watch() returns it. State lives in the book: every method re-reads the
    log, so successive handles on the same book agree."""

    def __init__(self, ledger, watch_id: str):
        self.ledger = ledger if isinstance(ledger, Ledger) else Ledger(str(ledger))
        self.id = str(watch_id)
        ops = self.ledger.watch_ops(self.id)
        if not ops or ops[0].get("action") != "open":
            raise KeyError(f"no watch {self.id} in {self.ledger.path}")

    @classmethod
    def load(cls, ledger, watch_id: str) -> "Watch":
        return cls(ledger, watch_id)

    def _ops(self) -> List[dict]:
        return self.ledger.watch_ops(self.id)

    @property
    def open_op(self) -> dict:
        return dict(self._ops()[0])

    @property
    def state(self) -> dict:
        return dict(self._ops()[-1]["state"])

    @property
    def root(self) -> str:
        return os.path.join(self.ledger.path, STORE, self.id)

    def _check_head(self) -> None:
        import json
        last = self._ops()[-1]
        try:
            with open(os.path.join(self.root, "head.json"), "rb") as fh:
                head = json.loads(fh.read())
        except FileNotFoundError:
            raise ValueError(f"refused: the head of {self.id} is missing; its log cannot be anchored")
        if head.get("seq") != last["seq"] or head.get("hash") != _op_hash(last):
            raise ValueError(f"refused: the log of {self.id} does not match its head (#{head.get('seq')}); it was edited or truncated after the last update, and the watch takes nothing more until that is resolved")

    def _append(self, action: str, payload: dict, state: dict) -> dict:
        last = self._ops()[-1]
        op = {"op": "watch", "watch": self.id, "action": action, "seq": int(last["seq"]) + 1, "prev": _op_hash(last), "at": _ho._now(), "state": state}
        op.update(payload)
        self.ledger._append(op)
        stored = self._ops()[-1]
        _write_atomic(os.path.join(self.root, "head.json"), _canon({"seq": stored["seq"], "hash": _op_hash(stored)}))
        return stored

    @_quiet
    def _statistic(self, yv: np.ndarray, pv: np.ndarray, bv: Optional[np.ndarray]) -> np.ndarray:
        """The monitored statistic in its own units: the model's loss (risk
        claim) or the model's loss minus the baseline's (gain claim), each
        clipped to the range; a non-finite prediction scores the top."""
        o = self.open_op
        lo, hi = o["range"]
        ok = np.isfinite(pv)
        lm = np.clip(_losses(o["loss"], yv, np.where(ok, pv, 0.0)), lo, hi)
        lm[~ok] = hi
        if o["claim"] == "risk":
            return lm
        return lm - np.clip(_losses(o["loss"], yv, bv), lo, hi)

    def _snapshot(self, st: dict, path=None) -> WatchState:
        o = self.open_op
        n = int(st["rows"])
        mean = None if n == 0 else float(st["sum"]) / n
        if mean is not None and o["claim"] == "gain":
            mean = -mean
        return WatchState(watch=self.id, certificate=o["certificate"], claim=o["claim"], loss=o["loss"], level=float(o["level"]), delta=float(o["delta"]), horizon=o.get("horizon"), rows=n, received=int(st["received"]), mean=mean, evidence=float(st["evidence"]), peak=float(st["peak"]), status=st["status"], alarm=st["alarm"], start=None if n == 0 else _start_of(st), last_time=_iso(st["last_time"]), path=path)

    def update(self, y, p, *, times=None, baseline=None, label: str = "") -> WatchState:
        """Append forward rows and run the detector over them, in order. y are
        the outcomes and p the predictions issued before them; baseline, for
        a gain claim, the baseline's predictions on the same rows. times, the
        rows' times (when each row starts), are given with every update or
        with none; with them the first row may not start before the
        certified rows end, and the rows run forward in time across
        updates. At the first row whose evidence
        reaches 1/delta the watch raises its alarm and stops: the rows after
        it in the same update are stored but not monitored, and later
        updates are refused. Rows beyond a horizon are stored but not
        monitored, and the watch ends there."""
        o = self.open_op
        yv = np.asarray(y, dtype=float).ravel()
        pv = np.asarray(p, dtype=float).ravel()
        n = len(yv)
        if n == 0:
            raise ValueError("update refused: no rows")
        if len(pv) != n:
            raise ValueError(f"update refused: {len(pv)} predictions for {n} outcomes")
        if not np.all(np.isfinite(yv)):
            raise ValueError("update refused: an outcome is missing or not finite")
        if o["loss"] in BINARY and not np.all(np.isin(yv, (0.0, 1.0))):
            raise ValueError(f"update refused: loss {o['loss']!r} takes a binary target in {{0, 1}}")
        bv = None
        if o["claim"] == "gain":
            if baseline is None:
                raise ValueError("update refused: the certificate bounds a paired gain over the sealed baseline; pass baseline= the baseline's predictions on the same rows")
            bv = np.asarray(baseline, dtype=float).ravel()
            if len(bv) != n or not np.all(np.isfinite(bv)):
                raise ValueError("update refused: the baseline's predictions must be finite, one per row")
        elif baseline is not None:
            raise ValueError("update refused: a risk claim is watched on the model's loss alone; it takes no baseline")
        tv = None if times is None else _times_ns(times, n)
        x = self._statistic(yv, pv, bv)
        a, b = o["stat_range"]
        u = (x - a) / (b - a)
        with self.ledger.locked():
            self._check_head()
            st = self.state
            if st["status"] != "watching":
                why = {"alarm": f"raised its alarm at row {st['alarm']['row'] if st['alarm'] else ''}", "ended": f"reached its horizon of {o.get('horizon')} rows", "closed": "was closed"}[st["status"]]
                raise ValueError(f"update refused: {self.id} {why}; a watch monitors one stretch of deployment, so open a new watch on a new certificate")
            if st["clocked"] is not None and st["clocked"] != (tv is not None):
                raise ValueError(f"update refused: {self.id} was {'given' if st['clocked'] else 'not given'} times at its first update; times come with every update or with none")
            if tv is not None:
                if st["last_time"] is not None and tv[0] < int(st["last_time"]):
                    raise ValueError(f"update refused: the first row's time {_iso(tv[0])} precedes the last watched row's {_iso(st['last_time'])}")
                if o["after"] is not None and tv[0] < int(o["after"]):
                    raise ValueError(f"update refused: the first row's time {_iso(tv[0])} precedes the end of the certified rows, {_iso(o['after'])}; the watch takes forward rows")
            room = len(u) if o.get("horizon") is None else int(o["horizon"]) - int(st["rows"])
            det, used, hit, path = _advance(st, u[:room], float(o["m"]), math.log(1.0 / float(o["delta"])), o.get("horizon"))
            seq = int(self._ops()[-1]["seq"]) + 1
            files = {"y": yv, "p": pv}
            if bv is not None:
                files["b"] = bv
            if tv is not None:
                files["t"] = tv
            d = os.path.join(self.root, str(seq))
            os.makedirs(d, exist_ok=True)
            shas = {}
            for name, arr in files.items():
                raw = _npy_bytes(arr)
                with open(os.path.join(d, f"{name}.npy"), "wb") as fh:
                    fh.write(raw)
                shas[name] = _sha(raw)
            new = dict(st)
            new.update(det)
            new["received"] = int(st["received"]) + n
            new["sum"] = float(st["sum"]) + float(np.sum(x[:used]))
            new["clocked"] = tv is not None
            new["last_time"] = None if tv is None else int(tv[used - 1])
            if hit is not None:
                new["status"] = "alarm"
                new["alarm"] = {"row": int(det["rows"]), "time": None if tv is None else _iso(tv[hit]), "start": _start_of(det), "evidence": float(det["evidence"])}
            elif o.get("horizon") is not None and int(det["rows"]) >= int(o["horizon"]):
                new["status"] = "ended"
            self._append("update", {"rows_in": n, "used": used, "files": shas, "label": str(label)}, new)
            return self._snapshot(new, path=np.exp(np.minimum(path, 700.0)))

    def close(self, cause: str = "declared") -> WatchState:
        """End the watch. The charge stays: it was taken before any row was
        seen. Idempotent."""
        with self.ledger.locked():
            self._check_head()
            st = self.state
            if st["status"] == "watching":
                new = dict(st)
                new["status"] = "closed"
                self._append("close", {"cause": str(cause)}, new)
                st = new
            return self._snapshot(st)

    def snapshot(self) -> WatchState:
        return self._snapshot(self.state)

    def audit(self) -> WatchAudit:
        """Replay the watch from its log and its row files under the book
        lock: the hash chain, the head and the id; every update's files
        against the hashes logged with it; and the detector re-run over the
        stored rows, batch by batch, with every logged state compared in
        full."""
        rows = []
        with self.ledger.locked():
            ops = self._ops()
            chain = all(ops[i]["prev"] == _op_hash(ops[i - 1]) and ops[i]["seq"] == i for i in range(1, len(ops))) and ops[0]["seq"] == 0 and ops[0]["prev"] is None
            rows.append(["hash chain", "PASS" if chain else "FAIL", f"{len(ops)} operations"])
            o0 = dict(ops[0])
            wid = o0.pop("watch", None)
            rows.append(["watch id", "PASS" if wid == self.id and "WA-" + _sha(_canon(o0))[:16].upper() == self.id else "FAIL", "id = hash of the open record"])
            try:
                self._check_head()
                rows.append(["head", "PASS", f"#{ops[-1]['seq']}"])
            except ValueError as e:
                rows.append(["head", "FAIL", str(e)])
            st = dict(ops[0]["state"])
            o = ops[0]
            ok_replay = True
            for op in ops[1:]:
                if op["action"] == "close":
                    st = dict(st)
                    st["status"] = "closed"
                    same = st == op["state"]
                    rows.append([f"#{op['seq']} close", "PASS" if same else "FAIL", "state matches" if same else "logged state differs"])
                    ok_replay &= same
                    continue
                d = os.path.join(self.root, str(op["seq"]))
                arrs, good = {}, True
                for name, want in op["files"].items():
                    try:
                        with open(os.path.join(d, f"{name}.npy"), "rb") as fh:
                            raw = fh.read()
                    except FileNotFoundError:
                        good = False
                        break
                    if _sha(raw) != want:
                        good = False
                        break
                    arrs[name] = _npy_load(raw)
                if not good:
                    rows.append([f"#{op['seq']} update", "FAIL", "a row file is missing or changed"])
                    ok_replay = False
                    break
                x = self._statistic(arrs["y"].astype(float), arrs["p"].astype(float), arrs.get("b"))
                a, b = o["stat_range"]
                room = len(x) if o.get("horizon") is None else int(o["horizon"]) - int(st["rows"])
                det, used, hit, _ = _advance(st, ((x - a) / (b - a))[:room], float(o["m"]), math.log(1.0 / float(o["delta"])), o.get("horizon"))
                new = dict(st)
                new.update(det)
                new["received"] = int(st["received"]) + len(x)
                new["sum"] = float(st["sum"]) + float(np.sum(x[:used]))
                tv = arrs.get("t")
                new["clocked"] = tv is not None
                new["last_time"] = None if tv is None else int(tv[used - 1])
                if hit is not None:
                    new["status"] = "alarm"
                    new["alarm"] = {"row": int(det["rows"]), "time": None if tv is None else _iso(tv[hit]), "start": _start_of(det), "evidence": float(det["evidence"])}
                elif o.get("horizon") is not None and int(det["rows"]) >= int(o["horizon"]):
                    new["status"] = "ended"
                same = new == op["state"]
                rows.append([f"#{op['seq']} update", "PASS" if same else "FAIL", f"{used} of {len(x)} rows monitored; state matches" if same else "the replayed state differs from the logged one"])
                ok_replay &= same
                st = op["state"]
        return WatchAudit(watch=self.id, rows=rows, ok=all(r[1] == "PASS" for r in rows))

    def render(self) -> str:
        return self.snapshot().render()

    def __str__(self):
        return self.render()

    def __repr__(self):
        return f"Watch({self.id!r}, book={self.ledger.path!r})"
