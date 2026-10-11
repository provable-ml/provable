"""provable.holdout -- priced peeking through a mediated holdout.

A Holdout is a set of rows put in custody when it is sealed. Its content
hashes, row identities, split, baseline and protocol are committed to the
book before anyone sees it, and from then on it is reached only through
its methods. Every answer is written to a hash-chained log in the book
together with its codelength under the code fixed at sealing; the
certificate the holdout finally issues is priced by the realized
codelength of that log, or taken on a sealed exam slice opened once at
the clean bar.

  seal(X, y, ledger, ...)  -> Holdout   commit, split, custody, protocol
  ho.ask(model, lineage)   -> Answer    comparative mode: does the model
                                        beat the incumbent by more than eta
                                        paired standard errors (the
                                        Ladder's rule); priced in bits
  ho.check(model, guess, width, lineage) -> Answer
                                        numeric mode: Guess and Check; a
                                        confirmed guess releases the guess,
                                        a failure a value on a grid
  ho.open(model, lineage)  -> Certificate
                                        the exam slice once at the clean
                                        bar, or (no exam slice sealed) the
                                        transcript certificate at z_L
  ho.retire()              -> Release   ends the tuning slice; its rows
                                        leave custody for training
  ho.transcript()          -> HoldoutTranscript
                                        the public log: log numbers, answers,
                                        their bits and the running codelength
  ho.audit(models)         -> HoldoutAudit
                                        replay everything from the log and
                                        the custody files

What is theorem and what is design:
  theorem  For a code fixed at sealing (Kraft's inequality at every step),
           a closed channel (every choice of the analyst a function of
           their own rows, independent randomness and past answers), and
           a single-model tail P(D(f) > c_gamma(f)) <= gamma, the
           certificate at c at level delta 2^-L, L the realized codelength,
           fails with probability at most delta for every strategy and
           every stopping rule. Cited in substance: the transcript union
           bound of Guess and Check (Rogers, Roth, Smith, Srebro, Thakkar,
           Woodworth, AISTATS 2020, Theorem 3.2) and the Ladder (Blum and
           Hardt, ICML 2015); stated here for any code fixed at sealing.
  theorem  An exam slice never queried certifies at the clean bar; with a
           share s of the rows it is the narrower certificate exactly when
           s > s(L) = (z_0 / z_L)^2.
  theorem  Guess and Check (Algorithm 1 of Rogers et al. 2020, with the
           paper's Hoeffding tolerance): every released answer covers its
           population value, simultaneously, with probability at least
           1 - delta, for statistics with a declared range.
  theorem  The single-model tail is exact when the loss has a range (a
           declared loss_range, or the natural range of a bounded loss):
           tail="betting" (the default there) and tail="hoeffding" bound
           the mean of n independent values in the range at every level.
  design   The comparative rule and eta, the KT default code, the exam
           share, the retirement threshold on s(L), the route (exam slice
           or transcript certificate) fixed at sealing, and the Gaussian
           form of the single-model tail where the loss has no range (the
           CLT bar, the default there; the certificate says "CLT
           approximation").
  channel  Until retire() hands released rows back, nothing the library
           returns depends on the sealed values except the priced answers and
           the issued certificates, and its refusals depend only on
           declarations, the public lineage (the seal's, and the rows and
           envelopes of the tuning and exam slices; under exam="last" the
           split follows the order of the sealed end times) and the public
           log: a loss outside a declared range is clipped to it, a
           non-finite prediction falls back to the sealed baseline's (or,
           with no baseline, scores the top of the range), arithmetic on
           sealed values ignores numpy's floating-point error settings, and
           an outcome-derived lag column is checked against the sealed clock
           at the seal only, after which the holdout accepts exactly the lag
           declarations it sealed. Models run on the sealed rows inside the
           custodian, outside the book lock; what a model does with what it
           is given (its errors and side effects included) is the analyst's
           undertaking, as is the closed channel. Declared outcome-derived
           columns that reach other sealed rows switch predictions to one row
           per call.
  custody  API-level mediation plus salted hash commitments. The rows sit in
           the book directory under holdouts/<id>/ as numpy and JSON files
           (no pickle). Every row and prediction file is checked against its
           salted hash before it is parsed; the custody metadata (salt and
           schema) and the custody head are plain JSON, and a changed salt or
           schema fails the same hashes. The custody head anchors the public
           log. The seal refuses rows that an active certificate of the book
           names. The book refuses a certificate or a refinement on rows in
           custody other than the holdout's own (a performance certificate
           whose content hashes to an id its log names, registered once), and
           the audit finds one written around the API. A process that reads
           the custody files directly is not stopped, and physical custody (a
           key held apart, a second person) is outside the library.
"""
from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import os
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from tabulate import tabulate

from .certificates import transcript as tr
from .certificates.lineage import Lineage, Slice, ColumnDeclaration, check_columns, clock_overlaps, identity_overlap, overlap_report, same_source, source_identity
from .certificates.records import PEEK_FIELD, Certificate, CertificateKind, Guarantee, GuaranteeKind, content_id, from_dict, is_unknown, issue, to_dict, _lineage_from_list, _lineage_to_list
from .certificates.runrecord import PeekDecl, RunRecord
from .certificates.store import Ledger

LOSSES = ("squared", "absolute", "logloss", "brier", "zero_one")
BINARY = ("logloss", "brier", "zero_one")
BOUNDED = {"brier": (0.0, 1.0), "zero_one": (0.0, 1.0)}
NATURAL = {"squared": (0.0, math.inf), "absolute": (0.0, math.inf), "logloss": (0.0, -math.log(1e-12)), "brier": (0.0, 1.0), "zero_one": (0.0, 1.0)}
LOSS_NOTE = {"brier": "probabilities clipped to [0, 1]", "logloss": "probabilities clipped to [1e-12, 1 - 1e-12]"}
MODES = ("comparative", "numeric")
CLAIMS = ("risk", "gain")
TAILS = ("auto", "gaussian", "hoeffding", "betting")
TAIL_NOTE = {"gaussian": "CLT approximation", "hoeffding": "exact, Hoeffding bound", "betting": "exact, betting bound"}
EXAMS = ("random", "last")
SEALINGS = ("historical", "forward")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
CUSTODY = "holdouts"
ACTIONS = ("ask", "check", "family", "open", "retire")


# ------------------------------------------------------------------ plumbing

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canon(d: Any) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _op_hash(op: Mapping[str, Any]) -> str:
    return _sha(_canon(dict(op)))


def _salted(salt: str, b: bytes) -> str:
    return _sha(bytes.fromhex(salt) + b)


def _npy_bytes(a) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


def _npy_load(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


def _pred_bytes(p) -> bytes:
    """The bytes of a prediction file: what is written, and what its
    salted hash commits."""
    return _npy_bytes(np.asarray(p, dtype=float))


def _quiet(fn):
    """Arithmetic on sealed values runs with numpy's floating-point errors
    ignored whatever the caller's np.seterr: an error raised or not raised
    by a sealed value would be an answer outside the log."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with np.errstate(all="ignore"):
            return fn(*args, **kwargs)
    return run


def _utc(t) -> datetime:
    """A UTC-aware datetime, floored to the microsecond a datetime holds
    (explicitly, so that no conversion warning depends on a sealed clock's
    nanoseconds)."""
    t = pd.Timestamp(t).floor("us").to_pydatetime()
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def _known(lin: Lineage) -> Lineage:
    """The instrumented slices of a lineage (UNKNOWN slices dropped)."""
    return Lineage(frozenset(s for s in lin.slices if not is_unknown(Lineage.of(s))))


def _used_rows(led: Ledger) -> Lineage:
    """Every row an active certificate of the book names, test or
    proposal, uninstrumented certificates included as far as their
    lineage is known: rows the seal refuses."""
    lin = Lineage()
    for c in led.active():
        lin = lin.union(_known(c.ell_test), _known(c.ell_prop))
    return lin


def _utc_lineage(lin: Lineage) -> Lineage:
    """The same slices with UTC-aware clock windows, so naive and aware
    declarations compare."""
    return Lineage(frozenset(Slice(source_id=s.source_id, dataset_tag=s.dataset_tag, rows=s.rows, t0=_utc(s.t0), t1=_utc(s.t1), aliases=s.aliases) for s in lin.slices))


def _covers(a: Lineage, b: Lineage) -> bool:
    """Every row of b is a row of a slice of a naming the same source
    (equal ids, or one id among the other's aliases)."""
    for sb in b.slices:
        same = [sa for sa in a.slices if same_source(sa, sb)]
        if not same:
            return False
        if any(sa.rows is None for sa in same):
            continue
        if sb.rows is None or not sb.rows <= frozenset().union(*(sa.rows for sa in same)):
            return False
    return True


def _complete_frame(X: pd.DataFrame, verb: str) -> None:
    """The A25 completeness rule at the seal: a gapped column refuses by
    name with its count; how a gap is filled is a registration decision
    made before the verb."""
    for c in X.columns:
        col = X[c]
        bad = int(col.isna().sum())
        if not bad and pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            bad = int((~np.isfinite(np.asarray(col.astype("float64")))).sum())
        if bad:
            raise ValueError(f"{verb} refused: column '{c}' carries {bad} missing or non-finite value(s) in {len(X)} rows; the holdout takes complete columns, and how a gap is filled (imputation, a missingness indicator, dropping the rows) is a registration decision made before sealing")


def _need(loss: str) -> str:
    return "proba" if loss in ("logloss", "brier") else ("label" if loss == "zero_one" else "value")


def _predict(model, X: pd.DataFrame, need: str) -> np.ndarray:
    if isinstance(model, (int, float, np.integer, np.floating)) and not isinstance(model, bool):
        return np.full(len(X), float(model))
    if need == "proba" and hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=float)
        if p.ndim == 2 and p.shape[1] == 1:
            raise ValueError("predict_proba returned a single column; the holdout needs P(y = 1), so pass a model fit on both classes or a callable")
        out = p[:, 1] if p.ndim == 2 else p
    elif hasattr(model, "predict"):
        out = model.predict(X)
    elif callable(model):
        out = model(X)
    else:
        raise TypeError("a model is a number (a constant prediction), an object with predict (or predict_proba), or a callable on the frame")
    out = np.asarray(out, dtype=float).ravel()
    if out.shape[0] != len(X):
        raise ValueError(f"the model returned {out.shape[0]} predictions for {len(X)} rows")
    return out


@_quiet
def _losses(loss: str, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    if loss == "squared":
        return (y - p) ** 2
    if loss == "absolute":
        return np.abs(y - p)
    if loss == "logloss":
        q = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))
    if loss == "brier":
        return (y - np.clip(p, 0.0, 1.0)) ** 2
    if loss == "zero_one":
        return (p != y).astype(float)
    raise ValueError(f"unknown loss {loss!r}; losses are {LOSSES}")


# ------------------------------------------------------------------ custody

def _json_scalar(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    raise TypeError(type(v).__name__)


def _encode_frame(X: pd.DataFrame) -> tuple:
    """X as named byte blobs (numpy files and JSON, no pickle) plus a
    schema; the seal refuses a frame that does not round-trip exactly."""
    files, schema = {}, []
    if X.columns.has_duplicates:
        raise ValueError("seal refused: X carries duplicate column names; custody stores one column per name")
    for k, c in enumerate(X.columns):
        try:
            name = _json_scalar(c)
        except TypeError:
            raise ValueError(f"seal refused: column name {c!r} is not a string or a number")
        col = X[c]
        dt = col.dtype
        ent = {"name": name, "dtype": str(dt)}
        if isinstance(dt, pd.CategoricalDtype):
            try:
                cats = [_json_scalar(v) for v in dt.categories]
            except TypeError:
                raise ValueError(f"seal refused: categorical column '{c}' has categories that are not strings or numbers")
            ent.update(kind="cat", categories=cats, ordered=bool(dt.ordered), file=f"x{k}.npy")
            files[ent["file"]] = _npy_bytes(np.asarray(col.cat.codes, dtype=np.int64))
        elif pd.api.types.is_bool_dtype(dt):
            ent.update(kind="bool", file=f"x{k}.npy")
            files[ent["file"]] = _npy_bytes(np.asarray(col.astype(bool), dtype=bool))
        elif pd.api.types.is_datetime64_any_dtype(dt):
            tz = getattr(dt, "tz", None)
            ent.update(kind="dt", tz=None if tz is None else str(tz), file=f"x{k}.npy")
            vals = col.dt.tz_convert("UTC").dt.tz_localize(None) if tz is not None else col
            files[ent["file"]] = _npy_bytes(np.asarray(vals.to_numpy().astype("datetime64[ns]").astype(np.int64)))
        elif pd.api.types.is_timedelta64_dtype(dt):
            ent.update(kind="td", file=f"x{k}.npy")
            files[ent["file"]] = _npy_bytes(np.asarray(col.to_numpy().astype("timedelta64[ns]").astype(np.int64)))
        elif pd.api.types.is_numeric_dtype(dt):
            ent.update(kind="num", file=f"x{k}.npy")
            base = np.dtype(dt.numpy_dtype) if hasattr(dt, "numpy_dtype") else np.dtype(dt)
            files[ent["file"]] = _npy_bytes(np.asarray(col.to_numpy(dtype=base)))
        else:
            try:
                vals = [_json_scalar(v) for v in col.tolist()]
            except TypeError as e:
                raise ValueError(f"seal refused: column '{c}' holds {e} values; custody stores strings, numbers and booleans")
            ent.update(kind="obj", file=f"x{k}.json")
            files[ent["file"]] = _canon(vals)
        schema.append(ent)
    if pd.api.types.is_integer_dtype(X.index.dtype):
        index = {"kind": "int", "file": "index.npy"}
        files["index.npy"] = _npy_bytes(np.asarray(X.index, dtype=np.int64))
    elif isinstance(X.index, pd.DatetimeIndex):
        tz = X.index.tz
        index = {"kind": "dt", "tz": None if tz is None else str(tz), "dtype": str(X.index.dtype), "name": _json_scalar(X.index.name) if X.index.name is None or isinstance(X.index.name, (str, int, float)) else str(X.index.name), "file": "index.npy"}
        vals = X.index.tz_convert("UTC").tz_localize(None) if tz is not None else X.index
        files["index.npy"] = _npy_bytes(np.asarray(vals.to_numpy().astype("datetime64[ns]").astype(np.int64)))
    else:
        try:
            vals = [_json_scalar(v) for v in X.index.tolist()]
        except TypeError:
            raise ValueError("seal refused: X's index labels are not strings or numbers")
        index = {"kind": "obj", "file": "index.json"}
        files["index.json"] = _canon(vals)
    return files, {"columns": schema, "index": index}


def _decode_frame(files: Mapping[str, bytes], schema: Mapping[str, Any]) -> pd.DataFrame:
    data = {}
    for ent in schema["columns"]:
        b = files[ent["file"]]
        kind = ent["kind"]
        if kind == "obj":
            s = pd.Series(json.loads(b), dtype=object)
        elif kind == "cat":
            s = pd.Series(pd.Categorical.from_codes(_npy_load(b), categories=ent["categories"], ordered=ent["ordered"]))
        elif kind == "dt":
            s = pd.Series(_npy_load(b).astype("datetime64[ns]"))
            if ent["tz"] is not None:
                s = s.dt.tz_localize("UTC").dt.tz_convert(ent["tz"])
        elif kind == "td":
            s = pd.Series(_npy_load(b).astype("timedelta64[ns]"))
        else:
            s = pd.Series(_npy_load(b))
        if str(s.dtype) != ent["dtype"]:
            s = s.astype(ent["dtype"])
        data[len(data)] = s
    X = pd.DataFrame(data)
    X.columns = [e["name"] for e in schema["columns"]]
    ix = schema["index"]
    ib = files[ix["file"]]
    if ix["kind"] == "int":
        X.index = pd.Index(_npy_load(ib))
    elif ix["kind"] == "dt":
        idx = pd.DatetimeIndex(_npy_load(ib).astype("datetime64[ns]"), name=ix.get("name"))
        if ix["tz"] is not None:
            idx = idx.tz_localize("UTC").tz_convert(ix["tz"])
        X.index = idx.astype(ix["dtype"]) if str(idx.dtype) != ix["dtype"] else idx
    else:
        X.index = pd.Index(json.loads(ib), dtype=object)
    return X


def _x_sha(salt: str, files: Mapping[str, bytes], schema: Mapping[str, Any]) -> str:
    parts = [_canon(schema)] + [files[e["file"]] for e in schema["columns"]] + [files[schema["index"]["file"]]]
    return _salted(salt, b"".join(hashlib.sha256(p).digest() for p in parts))


# ------------------------------------------------------------------ records

@dataclass
class Answer:
    """One answer of the custodian, exactly what the analyst sees: the
    answer, its codelength, and the running price. Comparative answers
    carry accepted; numeric answers carry the released value, its width,
    whether the guess was confirmed, and the per-query level. number counts
    the asks (or checks); seq is the answer's position in the log."""
    holdout: str
    number: int
    seq: int
    action: str
    bits: float
    total_bits: float
    bar: Optional[float]
    clean_share: Optional[float]
    tuning: str
    accepted: Optional[bool] = None
    confirmed: Optional[bool] = None
    value: Optional[float] = None
    width: Optional[float] = None
    level: Optional[float] = None
    failures: Optional[int] = None
    halted: bool = False

    @property
    def interval(self):
        if self.value is None or self.width is None:
            return None
        return (self.value - self.width, self.value + self.width)

    def render(self) -> str:
        if self.action == "ask":
            head = f"ask {self.number} (log #{self.seq}): {'ACCEPT' if self.accepted else 'REJECT'}"
            tail = f"{self.bits:.3f} bits; spent {self.total_bits:.2f}; z_L {self.bar:.3f}; s(L) {self.clean_share:.3f}; tuning {self.tuning}"
            return head + " | " + tail
        if self.halted:
            return f"check {self.number} (log #{self.seq}): HALTED, the width leaves no grid and the mechanism stops | level {self.level:.3g}; {self.failures} failure(s); tuning {self.tuning}"
        lo, hi = self.interval
        verdict = "CONFIRMED" if self.confirmed else "FAILED"
        return f"check {self.number} (log #{self.seq}): {verdict} {self.value:.6g} +/- {self.width:.6g} = [{lo:.6g}, {hi:.6g}] | level {self.level:.3g} ({self.bits:.2f} bits); {self.failures} failure(s); tuning {self.tuning}"

    def __str__(self):
        return self.render()


@dataclass
class Release:
    """Rows leaving custody at retirement: the tuning slice, and the exam
    slice when it has been opened. They are spent for this holdout and
    free for training."""
    holdout: str
    X: pd.DataFrame
    y: np.ndarray
    lineage: Lineage
    slices: tuple
    bits: float
    cause: str

    def render(self) -> str:
        return f"retired {self.holdout}: released {len(self.X)} rows ({', '.join(self.slices)}) after {self.bits:.2f} bits; cause: {self.cause}"

    def __str__(self):
        return self.render()


@dataclass
class HoldoutAudit:
    """The replay of a holdout from its log and its custody files: one row
    per check, PASS or FAIL, and the overall verdict."""
    holdout: str
    rows: list
    ok: bool

    def render(self) -> str:
        out = [f"-- audit: {self.holdout} --", tabulate([[r[0], r[1], str(r[2])[:90]] for r in self.rows], headers=["item", "verdict", "detail"], tablefmt="psql")]
        out.append(f"audit {'PASS' if self.ok else 'FAIL'}: {sum(1 for r in self.rows if r[1] == 'PASS')} of {len(self.rows)} items recomputed and matched")
        return "\n".join(out)

    def __str__(self):
        return self.render()


@dataclass
class HoldoutTranscript:
    """The public log of a holdout, one row per operation in log order: the
    log number, what the analyst was answered, and what it cost in bits."""
    holdout: str
    rows: list

    def render(self) -> str:
        body = [[r["seq"], r["action"], r["label"], r["answer"], "" if r["bits"] is None else f"{r['bits']:.3f}", f"{r['transcript_bits']:.3f}", r["certificate"]] for r in self.rows]
        return "\n".join([f"-- transcript: {self.holdout} --", tabulate(body, headers=["#", "action", "label", "answer", "bits", "transcript bits", "certificate"], tablefmt="psql")])

    def __str__(self):
        return self.render()


# ------------------------------------------------------------------ outcome columns

def _decl_to_dict(c: ColumnDeclaration) -> dict:
    return {"column": c.column, "dataset_tag": c.dataset_tag, "lag": None if c.lag is None else c.lag.total_seconds(), "window": None if c.window is None else c.window.total_seconds(), "rows": None if c.rows is None else _lineage_to_list(c.rows), "outcome_derived": bool(c.outcome_derived)}


def _row_slices(sl: Slice, ids: np.ndarray, clk) -> list:
    return [Slice(source_id=sl.source_id, dataset_tag=sl.dataset_tag, rows=frozenset([int(ids[i])]), t0=_utc(clk[0][i]), t1=_utc(clk[1][i]), aliases=sl.aliases) for i in range(len(ids))]


def _column_findings(cols: Sequence[ColumnDeclaration], H: Lineage, rows: Optional[list], exam_pos: Optional[np.ndarray]) -> tuple:
    """Check outcome-derived column declarations against the sealed rows,
    at the seal (the analyst holds the rows and the clock then).
    Violations: UNKNOWN lineage; an explicit-rows declaration naming a
    sealed row, or an outcome delivered after the first sealed decision
    time; a lag declaration without the holdout's clock; a lag reaching a
    sealed row's own outcome window or an outcome delivered after its
    decision time (check_columns per row, at the row's start). Reach: a
    lag whose source window covers another sealed row's outcome window
    (frame-level evaluation would show that outcome to the other row's
    prediction), and whether it reaches an exam row from a tuning row.
    Reach is computed for every lag declaration whatever its dataset tag:
    a tag is a name, and false staleness is the safe error. A refusal
    names the column and the rule, never a row's window."""
    (sl,) = tuple(H.slices)
    violations, reach, exam_reach = [], [], []
    starts = [r.t0 for r in rows or []]
    first = min(starts) if starts else _utc(sl.t0)
    exam_set = set() if exam_pos is None else set(int(i) for i in exam_pos)
    for col in cols:
        if not col.outcome_derived:
            continue
        if col.rows is not None:
            if is_unknown(col.rows):
                violations.append({"column": col.column, "reason": "column lineage is UNKNOWN", "detail": "an undeclared source cannot be cleared"})
                continue
            ov = identity_overlap(col.rows, H)
            if ov:
                violations.append({"column": col.column, "reason": "sealed outcome rows in column lineage", "detail": overlap_report(ov)})
                continue
            late = [s for s in _utc_lineage(col.rows).slices if s.t1 > first]
            if late:
                violations.append({"column": col.column, "reason": "outcome row delivers after the first sealed decision time", "detail": f"source {late[0].source_id} window ends {late[0].t1.isoformat()} after {first.isoformat()}"})
                continue
        if col.lag is not None:
            if rows is None:
                violations.append({"column": col.column, "reason": "a lag declaration needs the holdout's clock", "detail": "seal with clock=(start, end) to check lagged outcome columns row by row"})
                continue
            bad = None
            for r in rows:
                v = check_columns([col], [r], gate_time=r.t0)
                if v:
                    bad = v[0]
                    break
            if bad is not None:
                violations.append({"column": bad["column"], "reason": bad["reason"], "detail": f"lag {col.lag}" + ("" if col.window is None else f", window {col.window}")})
                continue
            starts_s = np.array([r.t0.timestamp() for r in rows])
            ends_s = np.array([r.t1.timestamp() for r in rows])
            order = np.argsort(starts_s, kind="stable")
            ss, ee = starts_s[order], ends_s[order]
            maxlen = float((ends_s - starts_s).max()) if len(rows) else 0.0
            hits = xhits = 0
            for i, r in enumerate(rows):
                a, b = col.source_window(r)
                a, b = a.timestamp(), b.timestamp()
                lo = int(np.searchsorted(ss, a - maxlen, side="left"))
                hi = int(np.searchsorted(ss, b, side="left"))
                cand = order[lo:hi][ee[lo:hi] > a]
                cand = cand[cand != i]
                if len(cand):
                    hits += 1
                    if exam_set and i not in exam_set and any(int(c) in exam_set for c in cand):
                        xhits += 1
            if hits:
                reach.append({"column": col.column, "rows": hits})
            if xhits:
                exam_reach.append({"column": col.column, "rows": xhits})
    return violations, reach, exam_reach


# ------------------------------------------------------------------ seal

def _resolve_ledger(ledger) -> Ledger:
    if isinstance(ledger, Ledger):
        return ledger
    if ledger is None:
        raise ValueError("seal refused: a holdout lives in a book; pass ledger= (a Ledger or a book directory)")
    return Ledger(str(ledger))


def _holdout_lineage(X: pd.DataFrame, lineage, dataset_tag: str, clock) -> tuple:
    """Row identity of the sealed rows: (lineage, ids, how). An explicit
    lineage is one slice whose rows are X's index labels in their source
    table; "self" makes the holdout its own source table."""
    if lineage is None:
        raise ValueError("seal refused: the holdout's row identity is undeclared. Pass lineage= the Lineage of X's rows in their source table (slice_from_frame(source, tag, rows=X.index)), or lineage='self' when the holdout is its own source table; the gate compares model lineage to these identities")
    if isinstance(lineage, str):
        if lineage != "self":
            raise ValueError("seal refused: lineage is a Lineage or the string 'self'")
        sid, aliases = source_identity(X.reset_index(drop=True))
        if clock is not None:
            t0, t1 = _utc(pd.to_datetime(pd.Series(clock[0]), utc=True).min()), _utc(pd.to_datetime(pd.Series(clock[1]), utc=True).max())
        else:
            t0 = t1 = EPOCH
        lin = Lineage.of(Slice(source_id=sid, dataset_tag=str(dataset_tag), rows=frozenset(range(len(X))), t0=t0, t1=t1, aliases=aliases))
        return lin, np.arange(len(X)), "position"
    if not isinstance(lineage, Lineage):
        raise TypeError("lineage is a Lineage or the string 'self'")
    if is_unknown(lineage):
        raise ValueError("seal refused: the holdout's lineage is UNKNOWN (predates instrumentation); a holdout seals instrumented rows only")
    if len(lineage.slices) != 1:
        raise ValueError(f"seal refused: the holdout's lineage carries {len(lineage.slices)} slices; seal one source at a time (one slice whose rows are X's index labels)")
    lineage = _utc_lineage(lineage)
    (sl,) = tuple(lineage.slices)
    if sl.rows is None:
        raise ValueError("seal refused: the holdout's slice names no rows (whole-source taint); seal explicit rows")
    try:
        ids = np.asarray(X.index, dtype=np.int64)
    except (TypeError, ValueError):
        raise ValueError("seal refused: with an explicit lineage, X's index labels are the row ids in the source table and must be integers")
    if len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != set(sl.rows):
        raise ValueError(f"seal refused: the lineage names {len(sl.rows)} rows and X's index labels do not match them exactly ({len(ids)} rows); build the lineage from the same rows, slice_from_frame(source, tag, rows=X.index)")
    return lineage, ids, "index"


def seal(X, y, ledger, *, lineage=None, loss: str = "squared", claim: str = "risk", baseline=None, baseline_lineage: Optional[Lineage] = None, mode: str = "comparative", code: str = tr.KT, eta: float = 1.0, delta: float = 0.05, exam_share: float = 0.2, exam: str = "random", retire_at: float = 0.05, tail: str = "auto", loss_range: Optional[Sequence[float]] = None, seed: int = 0, dataset_tag: str = "holdout", clock=None, columns: Optional[Iterable[ColumnDeclaration]] = None, sealing: str = "historical", label: str = "") -> "Holdout":
    """Seal rows into custody and commit the protocol before anyone looks.

    The book records, before any answer: salted hashes of X, y, the clock
    and the baseline's predictions, the row identities, the split seed and
    the exam slice, the declared outcome-derived columns of X, and the
    protocol (mode, route, code, eta, delta, exam share, retirement
    threshold, tail, loss and range, claim form, sealing type).
    mode="comparative" answers ask (a baseline is the first incumbent);
    mode="numeric" answers check (Guess and Check; a loss range is
    required unless the loss is bounded). The route is fixed here: with an
    exam slice the certificate is taken on it at the clean bar, without one
    (exam_share=0) on the tuning slice at the transcript bar z_L.
    tail="auto" resolves here and is recorded as resolved: "betting" (an
    exact bound) when the loss has a range, declared or natural, and
    "gaussian" (the CLT bar) when it has none; "gaussian", "hoeffding" and
    "betting" may be named, the last two only with a range.

    columns declares X's outcome-derived columns (ColumnDeclaration, lag or
    rows form). A declaration that reaches a sealed outcome per row
    refuses; one whose window covers another sealed row's outcome switches
    every prediction to one row per call, and needs exam='last' when it
    would carry an exam outcome into a tuning row. Rows already spent in
    the book, or already under seal in it, refuse."""
    led = _resolve_ledger(ledger)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if X.columns.has_duplicates:
        raise ValueError("seal refused: X carries duplicate column names; custody stores one column per name")
    if isinstance(y, pd.Series) and not y.index.equals(X.index):
        raise ValueError("seal refused: y's index does not match X's; align them before sealing")
    yv = np.asarray(y, dtype=float).ravel()
    n = len(X)
    if yv.shape[0] != n:
        raise ValueError(f"seal refused: X has {n} rows and y has {yv.shape[0]}")
    if n < 4:
        raise ValueError(f"seal refused: {n} rows; a holdout needs at least 4")
    if loss not in LOSSES:
        raise ValueError(f"seal refused: loss {loss!r} is not one of {LOSSES}")
    for name, val, allowed in (("claim", claim, CLAIMS), ("mode", mode, MODES), ("code", code, tr.CODES), ("tail", tail, TAILS), ("exam", exam, EXAMS), ("sealing", sealing, SEALINGS)):
        if val not in allowed:
            raise ValueError(f"seal refused: {name} {val!r} is not one of {allowed}")
    if not (0.0 < float(delta) < 0.5):
        raise ValueError("seal refused: delta must lie in (0, 0.5)")
    if float(eta) < 0:
        raise ValueError("seal refused: eta must be nonnegative")
    if not (0.0 <= float(exam_share) <= 0.9):
        raise ValueError("seal refused: exam_share must lie in [0, 0.9]")
    if not (0.0 < float(retire_at) < 1.0):
        raise ValueError("seal refused: retire_at must lie in (0, 1)")
    _complete_frame(X, "seal")
    bad = int((~np.isfinite(yv)).sum())
    if bad:
        raise ValueError(f"seal refused: the target carries {bad} missing or non-finite value(s) in {n} rows; how a gap is filled is a registration decision made before sealing")
    if loss in BINARY and not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError(f"seal refused: loss {loss!r} takes a binary target in {{0, 1}}")
    rng_ = None
    if loss_range is not None:
        lo, hi = float(loss_range[0]), float(loss_range[1])
        if not hi > lo:
            raise ValueError("seal refused: loss_range must be (lo, hi) with hi > lo")
        rng_ = [lo, hi]
    elif loss in BOUNDED:
        rng_ = list(BOUNDED[loss])
    if (mode == "numeric" or tail in ("hoeffding", "betting")) and rng_ is None:
        raise ValueError(f"seal refused: {'numeric mode' if mode == 'numeric' else 'the ' + ('Hoeffding' if tail == 'hoeffding' else 'betting') + ' tail'} needs a declared loss range; loss {loss!r} is unbounded, pass loss_range=(lo, hi)")
    if tail == "auto":
        tail = "betting" if rng_ is not None else "gaussian"
    if mode == "comparative" and baseline is None:
        raise ValueError("seal refused: comparative mode compares each model with an incumbent; seal a baseline (a model, or a number for a constant prediction)")
    if claim == "gain" and baseline is None:
        raise ValueError("seal refused: a gain claim is a paired gain over the sealed baseline; seal one")
    if baseline is not None and baseline_lineage is None:
        raise ValueError("seal refused: declare the rows the baseline was fit on (baseline_lineage=...); Lineage() declares that it was fit on no rows")
    if clock is not None and (len(clock[0]) != n or len(clock[1]) != n):
        raise ValueError("seal refused: clock is a (start, end) pair aligned with X's rows")
    H, ids, how = _holdout_lineage(X, lineage, dataset_tag, clock)
    _book_clear(led, H)
    if baseline is not None:
        if is_unknown(baseline_lineage):
            raise ValueError("seal refused: the baseline's lineage is UNKNOWN; the gate cannot clear it")
        ov = identity_overlap(baseline_lineage, H)
        if ov:
            raise ValueError(f"seal refused: the baseline's lineage meets the sealed rows: {overlap_report(ov)}")
    n_exam = int(round(float(exam_share) * n))
    n_tune = n - n_exam
    if n_tune < 2 or (n_exam and n_exam < 2):
        raise ValueError(f"seal refused: the split leaves {n_tune} tuning and {n_exam} exam rows; each slice needs at least 2")
    cl = None
    if clock is not None:
        cl = (pd.to_datetime(pd.Series(clock[0]), utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[ns]"), pd.to_datetime(pd.Series(clock[1]), utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[ns]"))
    if exam == "random":
        perm = np.random.default_rng(int(seed)).permutation(n)
    else:
        order = np.argsort(cl[1], kind="stable") if cl is not None else np.arange(n)
        perm = np.concatenate([order[:n_tune], order[n_tune:]])
    tune = np.sort(perm[:n_tune]).astype(int)
    exam_pos = np.sort(perm[n_tune:]).astype(int)
    split = {"tune": tune.tolist(), "exam": exam_pos.tolist()}
    cols = list(columns or [])
    rowslices = _row_slices(next(iter(H.slices)), ids, cl) if cl is not None else None
    violations, reach, exam_reach = _column_findings(cols, H, rowslices, exam_pos if n_exam else None)
    if violations:
        raise ValueError("seal refused: declared outcome columns reach sealed outcomes: " + "; ".join(f"[{x['column']}] {x['reason']}: {x['detail']}" for x in violations[:6]))
    if exam_reach:
        raise ValueError("seal refused: declared outcome columns carry exam outcomes into tuning rows (" + ", ".join(f"{r['column']} on {r['rows']} rows" for r in exam_reach) + "); seal with exam='last' so that the exam slice follows the tuning slice in time")
    rowwise = bool(reach)
    base_pred = None
    if baseline is not None:
        base_pred = np.concatenate([_predict(baseline, X.iloc[[i]], _need(loss)) for i in range(n)]) if rowwise else _predict(baseline, X, _need(loss))
        bad = int((~np.isfinite(base_pred)).sum())
        if bad:
            raise ValueError(f"seal refused: the baseline returned {bad} non-finite predictions on the sealed rows")
    files, schema = _encode_frame(X)
    if not _decode_frame(files, schema).equals(X):
        raise ValueError("seal refused: X does not round-trip exactly through custody (its column types are not preserved); convert the columns before sealing")
    salt = secrets.token_hex(16)
    y_b = _npy_bytes(yv)
    clock_b = None if cl is None else _npy_bytes(np.stack([cl[0].astype(np.int64), cl[1].astype(np.int64)]))
    base_b = None if base_pred is None else _pred_bytes(base_pred)
    route = ("exam" if n_exam else "kraft") if mode == "comparative" else "check"
    protocol = {"mode": mode, "route": route, "code": code, "eta": float(eta), "delta": float(delta), "exam_share": float(exam_share), "exam": exam, "retire_at": float(retire_at), "retire_bits": tr.bits_for_share(float(delta), float(retire_at)), "tail": tail, "loss": loss, "loss_range": rng_, "claim": claim, "sealing": sealing, "seed": int(seed), "rowwise": rowwise}
    lin_list = _lineage_to_list(H)
    commit = {"x_sha": _x_sha(salt, files, schema), "y_sha": _salted(salt, y_b), "rows_sha": _sha(_canon(lin_list)), "split_sha": _sha(_canon(split)), "baseline_sha": None if base_b is None else _salted(salt, base_b), "clock_sha": None if clock_b is None else _salted(salt, clock_b), "n": n, "n_tune": n_tune, "n_exam": n_exam, "ids": how}
    state = {"asks": 0, "accepts": 0, "bits": 0.0, "bar": tr.z_bar(float(delta), 0.0), "clean_share": 1.0, "checks": 0, "failures": 0, "gammas": [], "tuning": "open", "exam": "sealed" if n_exam else "none", "incumbent": 0, "family": None, "certificates": []}
    op = {"op": "holdout", "action": "seal", "seq": 0, "prev": None, "at": _now(), "commit": commit, "protocol": protocol, "lineage": lin_list, "baseline_lineage": None if baseline is None else _lineage_to_list(baseline_lineage), "columns": [_decl_to_dict(c) for c in cols], "label": str(label), "state": state}
    hid = "HO-" + _sha(_canon(op))[:16].upper()
    op["holdout"] = hid
    root = os.path.join(led.path, CUSTODY, hid)
    with led.locked():
        _book_clear(led, H)
        os.makedirs(os.path.join(root, "x"), exist_ok=True)
        os.makedirs(os.path.join(root, "pred"), exist_ok=True)
        for name, b in files.items():
            with open(os.path.join(root, "x", name), "wb") as fh:
                fh.write(b)
        with open(os.path.join(root, "y.npy"), "wb") as fh:
            fh.write(y_b)
        if clock_b is not None:
            with open(os.path.join(root, "clock.npy"), "wb") as fh:
                fh.write(clock_b)
        if base_b is not None:
            with open(os.path.join(root, "pred", "baseline.npy"), "wb") as fh:
                fh.write(base_b)
        with open(os.path.join(root, "split.json"), "wb") as fh:
            fh.write(_canon(split))
        with open(os.path.join(root, "custody.json"), "wb") as fh:
            fh.write(_canon({"salt": salt, "schema": schema}))
        led._append(op)
        _write_atomic(os.path.join(root, "head.json"), _canon({"seq": 0, "hash": _op_hash(led.holdout_ops(hid)[-1])}))
    return Holdout(led, hid)


def _book_clear(led: Ledger, H: Lineage) -> None:
    """The seal's book checks, run before the baseline and again under the
    book lock: no sealed row named by an active certificate of the book
    (test or proposal rows, as far as known) or under another seal."""
    ov = identity_overlap(H, _used_rows(led))
    if ov:
        raise ValueError(f"seal refused: rows already spent in the book {led.path}: {overlap_report(ov)}; accounting starts at the seal and earlier use of these rows cannot be priced afterwards")
    for hid in led.holdout_ids():
        ov = identity_overlap(H, _lineage_from_list(led.holdout_ops(hid)[0]["lineage"]))
        if ov:
            raise ValueError(f"seal refused: rows already under seal in {hid}: {overlap_report(ov)}; one holdout per row")


def custody_meets(ledger, lineage: Lineage, certificate_id: Optional[str] = None) -> list:
    """The holdouts of a book holding rows in custody that the lineage
    meets, as (holdout id, overlap report). A holdout whose log names the
    certificate id is passed over (register_guard admits that only for the
    holdout's own registration). Custody is read only for a holdout whose
    tuning slice has retired while its exam slice is still sealed;
    otherwise the seal's public lineage and the log's state decide."""
    led = _resolve_ledger(ledger)
    out = []
    for hid in led.holdout_ids():
        ops = led.holdout_ops(hid)
        if certificate_id is not None and any(o.get("certificate") == certificate_id for o in ops):
            continue
        ov = identity_overlap(lineage, _lineage_from_list(ops[0]["lineage"]))
        if not ov:
            continue
        st = ops[-1].get("state", {})
        if st.get("tuning") == "retired":
            if st.get("exam") in ("opened", "none"):
                continue
            ov = identity_overlap(lineage, Holdout(led, hid).exam_lineage())
        if ov:
            out.append((hid, overlap_report(ov)))
    return out


def register_guard(ledger, cert: Certificate, cid: str) -> None:
    """The check Ledger.register applies in a book holding a holdout. An id
    a holdout's log names is taken once, by a performance certificate whose
    content hashes to it: the holdout's own registration. Any other
    certificate whose test rows meet rows still in a holdout's custody
    refuses by name."""
    led = _resolve_ledger(ledger)
    owners = [hid for hid in led.holdout_ids() if any(o.get("certificate") == cid for o in led.holdout_ops(hid))]
    if owners:
        if cert.kind != CertificateKind.PERFORMANCE:
            raise ValueError(f"register refused: {cid} is an id in the log of {owners[0]}, which issues performance certificates only; a {getattr(cert.kind, 'value', cert.kind)} certificate cannot take it")
        if any(o["id"] == cid for o in led._recorded()):
            raise ValueError(f"register refused: {cid}, a certificate of {owners[0]}, is already registered in the book")
    hits = custody_meets(led, cert.ell_test, cid if owners else None)
    if hits:
        hid, rep = hits[0]
        raise ValueError(f"register refused: the certificate's test rows meet rows in the custody of {hid}: {rep}; sealed rows are reached through the holdout until retire() releases them")


# ------------------------------------------------------------------ the holdout

class Holdout:
    """A sealed holdout in a book. Construct with Holdout(ledger, id) to
    reopen one; seal() returns it. State lives in the book: every method
    re-reads the log, so successive handles on the same book agree."""

    def __init__(self, ledger, holdout_id: str):
        self.ledger = _resolve_ledger(ledger)
        self.id = str(holdout_id)
        ops = self.ledger.holdout_ops(self.id)
        if not ops or ops[0].get("action") != "seal":
            raise KeyError(f"no sealed holdout {self.id} in {self.ledger.path}")
        self._cache = {}

    @classmethod
    def load(cls, ledger, holdout_id: str) -> "Holdout":
        return cls(ledger, holdout_id)

    # ---------------------------------------------------------- reads
    def _ops(self) -> List[dict]:
        return self.ledger.holdout_ops(self.id)

    @property
    def seal_op(self) -> dict:
        return self._ops()[0]

    @property
    def protocol(self) -> dict:
        return dict(self.seal_op["protocol"])

    @property
    def commit(self) -> dict:
        return dict(self.seal_op["commit"])

    @property
    def state(self) -> dict:
        return dict(self._ops()[-1]["state"])

    @property
    def root(self) -> str:
        return os.path.join(self.ledger.path, CUSTODY, self.id)

    @property
    def lineage(self) -> Lineage:
        return _lineage_from_list(self.seal_op["lineage"])

    def _baseline_lineage(self) -> Lineage:
        b = self.seal_op.get("baseline_lineage")
        return Lineage() if b is None else _lineage_from_list(b)

    def _read(self, *parts) -> bytes:
        with open(os.path.join(self.root, *parts), "rb") as fh:
            return fh.read()

    def _load(self, force: bool = False) -> dict:
        """Custody rows: every row file is checked against the seal's salted
        commitment before it is parsed (custody.json, the salt and the
        schema, is plain JSON; a changed salt or schema fails the same
        hashes). The parsed rows are cached per handle and the cache is
        replaced whole, never emptied, so readers on one handle always see
        a complete snapshot; force re-reads and re-checks."""
        cm = self.commit
        stamp = tuple(os.path.getmtime(os.path.join(self.root, f)) for f in ("custody.json", "y.npy", "split.json"))
        c = self._cache
        if force or c.get("stamp") != stamp:
            cu = json.loads(self._read("custody.json"))
            salt, schema = cu["salt"], cu["schema"]
            files = {e["file"]: self._read("x", e["file"]) for e in schema["columns"]}
            files[schema["index"]["file"]] = self._read("x", schema["index"]["file"])
            if _x_sha(salt, files, schema) != cm["x_sha"]:
                raise ValueError(f"custody of {self.id} does not match its seal: the rows changed after sealing")
            yb = self._read("y.npy")
            if _salted(salt, yb) != cm["y_sha"]:
                raise ValueError(f"custody of {self.id} does not match its seal: the target changed after sealing")
            clk = None
            if cm["clock_sha"] is not None:
                cb = self._read("clock.npy")
                if _salted(salt, cb) != cm["clock_sha"]:
                    raise ValueError(f"custody of {self.id} does not match its seal: the clock changed after sealing")
                c2 = _npy_load(cb)
                clk = (c2[0].astype("datetime64[ns]"), c2[1].astype("datetime64[ns]"))
            sb = self._read("split.json")
            if _sha(sb) != cm["split_sha"]:
                raise ValueError(f"custody of {self.id} does not match its seal: the split changed after sealing")
            split = json.loads(sb)
            if cm["baseline_sha"] is not None and _salted(salt, self._read("pred", "baseline.npy")) != cm["baseline_sha"]:
                raise ValueError(f"custody of {self.id} does not match its seal: the baseline's predictions changed after sealing")
            c = {"stamp": stamp, "salt": salt, "X": _decode_frame(files, schema), "y": _npy_load(yb).astype(float), "clock": clk, "tune": np.asarray(split["tune"], int), "exam": np.asarray(split["exam"], int)}
            self._cache = c
        return c

    def _data(self):
        c = self._load()
        return c["X"], c["y"], c["tune"], c["exam"]

    def _ids(self) -> np.ndarray:
        X, _, _, _ = self._data()
        return np.asarray(X.index, dtype=np.int64) if self.commit["ids"] == "index" else np.arange(len(X))

    def _sub_lineage(self, pos: np.ndarray) -> Lineage:
        (sl,) = tuple(self.lineage.slices)
        c = self._load()
        ids = (np.asarray(c["X"].index, dtype=np.int64) if self.commit["ids"] == "index" else np.arange(len(c["X"])))[np.asarray(pos, int)]
        clk = c["clock"]
        if clk is not None and len(pos):
            t0, t1 = _utc(pd.Series(clk[0][pos]).min()), _utc(pd.Series(clk[1][pos]).max())
        else:
            t0, t1 = sl.t0, sl.t1
        return Lineage.of(Slice(source_id=sl.source_id, dataset_tag=sl.dataset_tag, rows=frozenset(int(i) for i in ids), t0=t0, t1=t1, aliases=sl.aliases))

    def tuning_lineage(self) -> Lineage:
        _, _, tune, _ = self._data()
        return self._sub_lineage(tune)

    def exam_lineage(self) -> Lineage:
        _, _, _, ex = self._data()
        return self._sub_lineage(ex)

    def _sealed_lineage(self, st: dict) -> Lineage:
        """Rows still in custody: the tuning slice until it retires, the
        exam slice until it is released together with an opened exam."""
        _, _, tune, ex = self._data()
        keep = [] if st["tuning"] == "retired" else list(tune)
        if not (st["tuning"] == "retired" and st["exam"] == "opened"):
            keep += list(ex)
        if not keep:
            return Lineage()
        return self._sub_lineage(np.sort(np.asarray(keep, int)))

    def _pred(self, key, want: Optional[str] = None) -> np.ndarray:
        """A stored prediction file, its bytes checked against their
        commitment (the seal's baseline hash, or the hash logged with the
        answer, or the one passed) before they are parsed."""
        b = self._read("pred", f"{key}.npy")
        salt = self._load()["salt"]
        if want is None:
            if key == "baseline":
                want = self.commit["baseline_sha"]
            else:
                ops = [o for o in self._ops() if o.get("seq") == int(key)]
                want = ops[0].get("model_sha") if ops else None
        if want is None or _salted(salt, b) != want:
            raise ValueError(f"custody of {self.id} does not match its log: prediction file {key} changed after it was written")
        return _npy_load(b).astype(float)

    def _model_sha(self, p) -> str:
        return _salted(self._load()["salt"], _pred_bytes(p))

    # ---------------------------------------------------------- losses
    def _substituted(self, p: np.ndarray) -> int:
        return int((~np.isfinite(np.asarray(p, dtype=float))).sum())

    @_quiet
    def _losses_on(self, p: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Per-row losses of predictions p on the given sealed rows, total
        by construction: a non-finite prediction falls back to the sealed
        baseline's for that row (with no baseline it scores the top of the
        declared range), and a declared range clips the loss. Nothing here
        refuses, so nothing here reveals the sealed values."""
        pr = self.protocol
        _, y, _, _ = self._data()
        p = np.asarray(p, dtype=float).copy()
        bad = ~np.isfinite(p)
        if bad.any() and self.commit["baseline_sha"] is not None:
            p[bad] = self._pred("baseline")[rows][bad]
            bad = np.zeros(len(p), dtype=bool)
        lf = _losses(pr["loss"], y[rows], np.where(bad, 0.0, p))
        rng_ = pr.get("loss_range")
        if rng_ is not None:
            lf = np.clip(lf, rng_[0], rng_[1])
            lf[bad] = rng_[1]
        return lf

    @_quiet
    def _statistic(self, p: np.ndarray, rows: np.ndarray) -> tuple:
        """Per-row statistic of the claim form (risk: the model's loss;
        gain: the baseline's loss minus the model's) and its range when a
        loss range is declared."""
        pr = self.protocol
        lf = self._losses_on(p, rows)
        rng_ = pr.get("loss_range")
        if pr["claim"] == "risk":
            return lf, (None if rng_ is None else (rng_[0], rng_[1]))
        lb = self._losses_on(self._pred("baseline")[rows], rows)
        span = None if rng_ is None else (rng_[1] - rng_[0])
        return lb - lf, (None if span is None else (-span, span))

    def _clipped(self) -> bool:
        """True when the declared range can cut the loss: it lies inside the
        loss's natural range at either end."""
        pr = self.protocol
        rng_ = pr.get("loss_range")
        if rng_ is None:
            return False
        lo, hi = NATURAL[pr["loss"]]
        return rng_[0] > lo or rng_[1] < hi

    # ---------------------------------------------------------- writes
    def _check_head(self) -> None:
        ops = self._ops()
        try:
            head = json.loads(self._read("head.json"))
        except FileNotFoundError:
            raise ValueError(f"refused: the custody head of {self.id} is missing; its log cannot be anchored")
        last = ops[-1]
        if head.get("seq") != last["seq"] or head.get("hash") != _op_hash(last):
            raise ValueError(f"refused: the log of {self.id} does not match its custody head (#{head.get('seq')}); it was edited or truncated after the last answer, and the holdout answers nothing more until that is resolved")

    def _append(self, action: str, payload: dict, state: dict) -> dict:
        ops = self._ops()
        last = ops[-1]
        op = {"op": "holdout", "holdout": self.id, "action": action, "seq": int(last["seq"]) + 1, "prev": _op_hash(last), "at": _now(), "state": state}
        op.update(payload)
        self.ledger._append(op)
        stored = self._ops()[-1]
        _write_atomic(os.path.join(self.root, "head.json"), _canon({"seq": stored["seq"], "hash": _op_hash(stored)}))
        return stored

    def _gate(self, verb: str, lineage, columns, st: dict) -> list:
        """The lineage gate: declared, instrumented, identity-disjoint from
        every row still in custody. Outcome-derived columns: a lag
        declaration is checked against the sealed clock row by row, so it
        belongs to the protocol; after the seal the gate accepts exactly the
        lag declarations the seal checked and refuses any other by name
        without reading the clock. A rows declaration is checked against the
        rows still in custody: UNKNOWN lineage, a row in custody, or an
        outcome delivered after the earliest decision time in custody (the
        start of the custody envelope) refuses. Clock overlaps across
        dataset tags are recorded, not refused. Refusals depend on
        declarations, the public lineage (the seal's, and the tuning and
        exam envelopes) and the public log only."""
        if lineage is None:
            raise ValueError(f"{verb} refused: declare the rows the model was fit on (lineage=...); the holdout admits declared lineage only, and Lineage() declares that the model was fit on no rows")
        if not isinstance(lineage, Lineage):
            raise TypeError("lineage is a Lineage")
        if is_unknown(lineage):
            raise ValueError(f"{verb} refused: the model's lineage is UNKNOWN (predates instrumentation); the gate cannot clear it")
        lineage = _utc_lineage(lineage)
        custody = self._sealed_lineage(st)
        ov = identity_overlap(lineage, custody)
        if ov:
            raise ValueError(f"{verb} refused: the model's lineage meets the sealed rows of {self.id}: {overlap_report(ov)}")
        flags = clock_overlaps(lineage, _utc_lineage(self.lineage))
        sealed = [dict(c) for c in (self.seal_op.get("columns") or [])]
        for col in list(columns or []):
            if not col.outcome_derived:
                continue
            if col.lag is not None:
                if _decl_to_dict(col) not in sealed:
                    raise ValueError(f"{verb} refused: [{col.column}] an outcome-derived lag column is checked against the sealed clock row by row, so it is part of the protocol: declare it when sealing (seal(..., columns=...)); after the seal the holdout accepts exactly the lag declarations it sealed")
                continue
            if col.rows is None:
                continue
            if is_unknown(col.rows):
                raise ValueError(f"{verb} refused: [{col.column}] column lineage is UNKNOWN: an undeclared source cannot be cleared")
            if custody.is_empty():
                continue
            ov = identity_overlap(col.rows, custody)
            if ov:
                raise ValueError(f"{verb} refused: [{col.column}] sealed outcome rows in column lineage: {overlap_report(ov)}")
            first = min(s.t0 for s in custody.slices)
            late = [s for s in _utc_lineage(col.rows).slices if s.t1 > first]
            if late:
                raise ValueError(f"{verb} refused: [{col.column}] outcome row delivers after the earliest decision time in custody: source {late[0].source_id} window ends {late[0].t1.isoformat()}, after {first.isoformat()}")
        return flags

    def _predict_on(self, model, rows: np.ndarray) -> np.ndarray:
        X, _, _, _ = self._data()
        need = _need(self.protocol["loss"])
        Z = X.iloc[rows]
        if self.protocol.get("rowwise"):
            return np.concatenate([_predict(model, Z.iloc[[i]], need) for i in range(len(Z))]) if len(Z) else np.zeros(0)
        return _predict(model, Z, need)

    def _incumbent_pred(self, inc: int, rows: np.ndarray) -> np.ndarray:
        if int(inc) == 0:
            return self._pred("baseline")[rows]
        return self._pred(int(inc))

    # ---------------------------------------------------------- ask
    def ask(self, model, lineage: Optional[Lineage] = None, columns: Optional[Iterable[ColumnDeclaration]] = None, label: str = "") -> Answer:
        """Comparative answer: ACCEPT iff the model's paired gain over the
        incumbent on the tuning slice exceeds eta paired standard errors
        (the Ladder's rule); an accepted model becomes the incumbent. The
        answer is all the analyst sees; its codelength under the sealed
        code is added to the bits spent. When the clean share falls to the
        retirement threshold the tuning slice retires itself. The checks run
        under the book lock before the model and again after it, against the
        state the answer is written to; the model itself runs outside the
        lock."""
        columns, label = list(columns or []), str(label)
        def ready():
            self._check_head()
            pr, st = self.protocol, self.state
            if pr["mode"] != "comparative":
                raise ValueError(f"ask refused: {self.id} is sealed in numeric mode; its answers come from check")
            self._tuning_open("ask", st)
            return pr, st, self._gate("ask", lineage, columns, st)
        with self.ledger.locked():
            ready()
        _, _, tune, _ = self._data()
        p = self._predict_on(model, tune)
        with self.ledger.locked(), np.errstate(all="ignore"):
            pr, st, flags = ready()
            d = self._losses_on(self._incumbent_pred(st["incumbent"], tune), tune) - self._losses_on(p, tune)
            accepted = _ladder(d, pr["eta"])
            b = tr.answer_bits(pr["code"], st["asks"], st["accepts"], accepted)
            L = float(st["bits"]) + b
            s = tr.clean_share(pr["delta"], L)
            seq = int(self._ops()[-1]["seq"]) + 1
            with open(os.path.join(self.root, "pred", f"{seq}.npy"), "wb") as fh:
                fh.write(_pred_bytes(p))
            new = dict(st)
            new.update({"asks": st["asks"] + 1, "accepts": st["accepts"] + int(accepted), "bits": L, "bar": tr.z_bar(pr["delta"], L), "clean_share": s, "incumbent": seq if accepted else st["incumbent"]})
            payload = {"model_sha": self._model_sha(p), "lineage": _lineage_to_list(lineage), "flags": len(flags), "label": str(label), "answer": "accept" if accepted else "reject", "answer_bits": b}
            if s <= pr["retire_at"]:
                new["tuning"] = "retired"
                payload["retire"] = f"threshold: s(L) {s:.4f} <= retire_at {pr['retire_at']:g}"
            self._append("ask", payload, new)
            return Answer(holdout=self.id, number=new["asks"], seq=seq, action="ask", bits=b, total_bits=L, bar=new["bar"], clean_share=s, tuning=new["tuning"], accepted=accepted)

    @staticmethod
    def _retirement(ops: list) -> tuple:
        for o in ops:
            if o["action"] == "retire":
                return o["seq"], o["cause"]
            if o["action"] == "ask" and o.get("retire"):
                return o["seq"], o["retire"]
        return None, None

    def _tuning_open(self, verb: str, st: dict) -> None:
        if st["tuning"] == "open":
            return
        ops = self._ops()
        if st["tuning"] == "retired":
            seq, cause = self._retirement(ops)
            raise ValueError(f"{verb} refused: the tuning slice of {self.id} retired at #{seq} ({cause}); seal the next block for further questions")
        if st["tuning"] == "certified":
            o = [o for o in ops if o["action"] == "open" and o["slice"] == "tuning"][0]
            raise ValueError(f"{verb} refused: the tuning slice of {self.id} closed at #{o['seq']}, when its transcript certificate was issued")
        if st["tuning"] == "halted":
            o = [o for o in ops if o["action"] == "check" and o["outcome"] == "halted"][0]
            raise ValueError(f"{verb} refused: Guess and Check on {self.id} halted at #{o['seq']}; the mechanism stops after a failure leaves no grid")
        raise ValueError(f"{verb} refused: tuning slice of {self.id} is {st['tuning']}")

    # ---------------------------------------------------------- check
    def check(self, model, guess: float, width: float, lineage: Optional[Lineage] = None, columns: Optional[Iterable[ColumnDeclaration]] = None, label: str = "") -> Answer:
        """Numeric answer by Guess and Check (Rogers et al. 2020, Algorithm
        1, Hoeffding tolerance). The analyst submits a model, a guess of
        its claim statistic (the mean loss, or the mean paired gain over
        the baseline) and a width. With beta_i the per-query level and
        tol its Hoeffding tolerance: if the guess lies within width - tol
        of the tuning-slice value, the guess is CONFIRMED and released
        with the width; otherwise the query FAILS and releases the
        tuning-slice value on a grid of step width - tol, with the width;
        a failure that leaves no grid HALTS the mechanism. Every released
        interval covers its population value, simultaneously, at
        1 - delta. The family is one performance certificate, registered
        before the first answer and charged once; when a check widens its
        proposal lineage a successor is registered and the predecessor
        voided, so the charge stays at delta. The model runs outside the
        book lock, between two locked passes of the same checks."""
        columns, label, guess, width = list(columns or []), str(label), float(guess), float(width)
        def ready():
            self._check_head()
            pr, st = self.protocol, self.state
            if pr["mode"] != "numeric":
                raise ValueError(f"check refused: {self.id} is sealed in comparative mode; its answers come from ask")
            self._tuning_open("check", st)
            if not (float(width) > 0):
                raise ValueError("check refused: the width must be positive")
            fam = st.get("family")
            if fam is not None and fam in self.ledger._voided_ids():
                raise ValueError(f"check refused: the family certificate {fam} of {self.id} was voided; its answers would no longer be charged to the book")
            return pr, st, self._gate("check", lineage, columns, st)
        with self.ledger.locked():
            ready()
        _, _, tune, _ = self._data()
        p = self._predict_on(model, tune)
        with self.ledger.locked(), np.errstate(all="ignore"):
            pr, st, flags = ready()
            new = self._family_for(dict(st), lineage)
            sv, (a, b) = self._statistic(p, tune)
            span = b - a
            a_h = float(((sv - a) / span).mean())
            a_g = (float(guess) - a) / span
            tau = float(width) / span
            i = st["checks"] + 1
            gammas = list(st["gammas"])
            bits_i = tr.gnc_bits(i, gammas)
            beta_i = tr.gnc_level(pr["delta"], i, gammas)
            tol = tr.hoeffding_tol_bits(pr["delta"], bits_i, len(tune))
            if abs(a_g - a_h) <= tau - tol:
                outcome, released, gamma = "confirmed", float(guess), None
            else:
                gamma = tau - tol
                if gamma > 0:
                    outcome = "failed"
                    released = a + tr.gnc_round(a_h, gamma) * span
                    gammas.append(gamma)
                else:
                    outcome, released, gamma = "halted", None, None
            failures = st["failures"] + int(outcome != "confirmed")
            seq = int(self._ops()[-1]["seq"]) + 1
            with open(os.path.join(self.root, "pred", f"{seq}.npy"), "wb") as fh:
                fh.write(_pred_bytes(p))
            new.update({"checks": i, "failures": failures, "gammas": gammas, "bits": tr.gnc_bits(i + 1, gammas), "tuning": "halted" if outcome == "halted" else new["tuning"]})
            self._append("check", {"model_sha": self._model_sha(p), "lineage": _lineage_to_list(lineage), "flags": len(flags), "label": str(label), "guess": float(guess), "width": float(width), "outcome": outcome, "released": released, "gamma": gamma, "level": beta_i, "query_bits": bits_i, "tol": tol}, new)
            return Answer(holdout=self.id, number=i, seq=seq, action="check", bits=bits_i, total_bits=new["bits"], bar=None, clean_share=None, tuning=new["tuning"], confirmed=outcome == "confirmed", value=released, width=float(width), level=beta_i, failures=failures, halted=outcome == "halted")

    def min_width(self) -> float:
        """The Hoeffding tolerance of the next check, in the units of the
        claim statistic: a width at or below it cannot confirm, and a
        failure at it halts the mechanism. Computed from the public
        transcript only (the number of checks, the grids of past failures,
        the tuning-slice size), so it reveals nothing."""
        pr, st = self.protocol, self.state
        if pr["mode"] != "numeric":
            raise ValueError(f"min_width: {self.id} is sealed in comparative mode")
        rng_ = pr["loss_range"]
        span = (rng_[1] - rng_[0]) * (1.0 if pr["claim"] == "risk" else 2.0)
        bits = tr.gnc_bits(st["checks"] + 1, list(st["gammas"]))
        return float(span * tr.hoeffding_tol_bits(pr["delta"], bits, int(self.commit["n_tune"])))

    def _family_for(self, st: dict, lineage: Lineage) -> dict:
        """The Guess and Check family certificate covering this check:
        registered at the first check, and superseded when the check's
        model lineage (with the baseline's) is not inside its proposal
        rows. The op is logged before the certificate is registered."""
        need = _utc_lineage(lineage).union(self._baseline_lineage())
        fam = st.get("family")
        prev_prop = Lineage()
        if fam is not None:
            recs = {}
            for o in self.ledger._registered():
                if o["record"].get("kind") == CertificateKind.PERFORMANCE.value and content_id(o["record"]) == o["id"]:
                    recs.setdefault(o["id"], o["record"])
            if fam not in recs:
                raise ValueError(f"check refused: the family certificate {fam} of {self.id} is missing from the book (its registration did not complete); audit names it")
            prev_prop = from_dict(recs[fam]).ell_prop
            if _covers(prev_prop, need):
                return st
        pr = self.protocol
        rng_ = pr["loss_range"]
        span = rng_[1] - rng_[0]
        rng_stat = rng_ if pr["claim"] == "risk" else [-span, span]
        famd = {"rule": "Guess and Check (Rogers et al. 2020, Algorithm 1; Hoeffding tolerance)", "holdout": self.id, "route": "check", "claim": pr["claim"], "loss": pr["loss"], "clipped": self._clipped(), "range": rng_stat, "n": int(self.commit["n_tune"]), "seal": _op_hash(self.seal_op), "supersedes": fam}
        stat = f"mean {pr['loss']} loss" if pr["claim"] == "risk" else f"mean {pr['loss']} paired gain over the sealed baseline"
        notes = ([f"loss clipped to [{rng_[0]:g}, {rng_[1]:g}]"] if self._clipped() else []) + ([LOSS_NOTE[pr["loss"]]] if pr["loss"] in LOSS_NOTE else []) + ["a non-finite prediction scored as the sealed baseline's" if self.commit["baseline_sha"] is not None else "a non-finite prediction scored at the top of the range"]
        stat += " (" + "; ".join(notes) + ")"
        cert = issue(claim=f"performance: every answer check releases on {self.id} covers the population {stat}, simultaneously (FWER {pr['delta']:g}; Guess and Check)", guarantee=Guarantee(kind=GuaranteeKind.FWER, level=float(pr["delta"]), family=famd), ell_test=self.tuning_lineage(), proposer_rows=prev_prop.union(need), artifact={"holdout": self.id}, support=(), gram_context={}, kind=CertificateKind.PERFORMANCE)
        cid = content_id(to_dict(cert))
        new = dict(st)
        new["family"] = cid
        new["certificates"] = [c for c in st["certificates"] if c != fam] + [cid]
        self._append("family", {"certificate": cid, "supersedes": fam}, new)
        self.ledger.register(cert)
        if fam is not None:
            self.ledger.void(fam, f"superseded by {cid}: the family's proposal lineage widened")
        return new

    # ---------------------------------------------------------- open
    def open(self, model, lineage: Optional[Lineage] = None, columns: Optional[Iterable[ColumnDeclaration]] = None, label: str = "") -> Certificate:
        """Issue the holdout's performance certificate for one model and
        register it in the book (one charge at delta). With an exam slice
        sealed, the exam slice is opened once and the bar is the clean one;
        without one (comparative, exam_share=0), the certificate is taken on
        the tuning slice at level delta 2^-L, L the bits spent, and the
        tuning slice closes. A risk claim bounds the model's mean loss from
        above; a gain claim bounds its mean paired gain over the sealed
        baseline from below. The open is logged before the certificate is
        registered, so a slice is never left open behind an issued
        certificate. The model runs outside the book lock, between two
        locked passes of the same checks."""
        columns, label = list(columns or []), str(label)
        def ready():
            self._check_head()
            pr, st = self.protocol, self.state
            ops = self._ops()
            if st["exam"] in ("sealed", "opened"):
                if st["exam"] == "opened":
                    o = [o for o in ops if o["action"] == "open" and o["slice"] == "exam"][0]
                    raise ValueError(f"open refused: the exam slice of {self.id} was opened at #{o['seq']} ({o['certificate']}); a slice is opened once")
                where = "exam"
            elif pr["route"] == "kraft":
                if st["tuning"] == "certified":
                    o = [o for o in ops if o["action"] == "open" and o["slice"] == "tuning"][0]
                    raise ValueError(f"open refused: the transcript certificate of {self.id} was issued at #{o['seq']} ({o['certificate']}); a slice is opened once")
                self._tuning_open("open", st)
                where = "tuning"
            else:
                raise ValueError(f"open refused: {self.id} certifies through check (numeric mode) and sealed no exam slice")
            self._gate("open", lineage, columns, st)
            return st, ops, where
        with self.ledger.locked():
            _, _, where0 = ready()
        _, _, tune, ex = self._data()
        rows = ex if where0 == "exam" else tune
        p = self._predict_on(model, rows)
        with self.ledger.locked(), np.errstate(all="ignore"):
            st, ops, where = ready()
            if where != where0:
                raise ValueError(f"open refused: the state of {self.id} changed while the model ran; call open again")
            cert, art, bar = self._certificate(p, rows, where, st, lineage, ops)
            cid = content_id(to_dict(cert))
            seq = int(ops[-1]["seq"]) + 1
            with open(os.path.join(self.root, "pred", f"{seq}.npy"), "wb") as fh:
                fh.write(_pred_bytes(p))
            new = dict(st)
            new["certificates"] = list(st["certificates"]) + [cid]
            if where == "exam":
                new["exam"] = "opened"
            else:
                new["tuning"] = "certified"
            self._append("open", {"slice": where, "model_sha": art["model_sha"], "lineage": _lineage_to_list(lineage), "label": str(label), "certificate": cid, "timestamp": cert.timestamp.isoformat(), "estimate": art["estimate"], "se": art["se"], "bar": float(bar), "width": art["width"], "bound": art["bound"]}, new)
            self.ledger.register(cert)
            return cert

    @_quiet
    def _certificate(self, p: np.ndarray, rows: np.ndarray, where: str, st: dict, lineage: Lineage, ops: list, timestamp=None) -> tuple:
        pr = self.protocol
        sv, rng_ = self._statistic(p, rows)
        n = len(sv)
        L = float(st["bits"])
        bar_bits = L if where == "tuning" else 0.0
        mean = float(sv.mean())
        se = float(sv.std(ddof=1) / math.sqrt(n))
        if pr["tail"] == "gaussian":
            bar = tr.z_bar(pr["delta"], bar_bits)
            width = bar * se
        elif pr["tail"] == "hoeffding":
            bar = tr.hoeffding_width(pr["delta"], bar_bits, n, rng_[1] - rng_[0])
            width = bar
        else:
            edge = tr.betting_bound(sv, rng_[0], rng_[1], pr["delta"], bar_bits, side="upper" if pr["claim"] == "risk" else "lower")
            width = max(edge - mean, 0.0) if pr["claim"] == "risk" else max(mean - edge, 0.0)
            bar = width
        bound = mean + width if pr["claim"] == "risk" else mean - width
        msha = self._model_sha(p)
        k_sub = self._substituted(p)
        fam = {"rule": "Kraft union bound over transcripts" if where == "tuning" else "exam slice opened once, clean bar", "holdout": self.id, "route": pr["route"], "slice": where, "mode": pr["mode"], "code": pr["code"], "eta": pr["eta"], "tail": pr["tail"], "claim": pr["claim"], "loss": pr["loss"], "clipped": self._clipped(), "loss_range": pr.get("loss_range"), "bits": L, "bar_bits": bar_bits, "level_at_bar": float(pr["delta"]) * 2.0 ** (-bar_bits), "bar": float(bar), "n": n, "asks": st["asks"], "accepts": st["accepts"], "checks": st["checks"], "substituted": k_sub, "log_head": _op_hash(ops[-1]), "seal": _op_hash(ops[0])}
        art = {"holdout": self.id, "model_sha": msha, "estimate": mean, "se": se, "width": float(width), "bound": float(bound), "substituted": k_sub}
        clip = f", loss clipped to [{pr['loss_range'][0]:g}, {pr['loss_range'][1]:g}]" if self._clipped() else ""
        if pr["loss"] in LOSS_NOTE:
            clip += f", {LOSS_NOTE[pr['loss']]}"
        if k_sub:
            clip += f", {k_sub} of {n} predictions not finite and scored as the sealed baseline's" if self.commit["baseline_sha"] is not None else f", {k_sub} of {n} predictions not finite and scored at the top of the range"
        if pr["claim"] == "risk":
            text = f"performance: mean {pr['loss']} loss of model {msha[:12]} at most {bound:.6g} (FWER {pr['delta']:g}, {TAIL_NOTE[pr['tail']]}; {self.id}, {where} slice, {L:.2f} bits spent{clip})"
        else:
            text = f"performance: mean {pr['loss']} paired gain of model {msha[:12]} over the sealed baseline at least {bound:.6g} (FWER {pr['delta']:g}, {TAIL_NOTE[pr['tail']]}; {self.id}, {where} slice, {L:.2f} bits spent{clip})"
        peeks = [{"holdout": self.id, "bits": L, "rows": self.tuning_lineage()}] if L > 0 else []
        cert = issue(claim=text, guarantee=Guarantee(kind=GuaranteeKind.FWER, level=float(pr["delta"]), family=fam), ell_test=self._sub_lineage(rows), proposer_rows=_utc_lineage(lineage).union(self._baseline_lineage()), artifact=art, support=(f"model:{msha[:16]}",), gram_context={}, kind=CertificateKind.PERFORMANCE, peeks=peeks, timestamp=timestamp)
        return cert, art, bar

    # ---------------------------------------------------------- retire
    def retire(self, cause: str = "declared") -> Release:
        """End the tuning slice and release its rows from custody (with the
        exam slice when it has been opened). Idempotent: a slice already
        retired, by declaration or at the threshold, is released again."""
        cause = str(cause)
        with self.ledger.locked():
            self._check_head()
            st = self.state
            if st["tuning"] != "retired":
                new = dict(st)
                new["tuning"] = "retired"
                self._append("retire", {"cause": str(cause)}, new)
                st = new
            _, cause = self._retirement(self._ops())
            X, y, tune, ex = self._data()
            rows, slices = tune, ("tuning",)
            if st["exam"] == "opened":
                rows, slices = np.sort(np.concatenate([tune, ex])), ("tuning", "exam")
            return Release(holdout=self.id, X=X.iloc[rows].copy(), y=y[rows].copy(), lineage=self._sub_lineage(rows), slices=slices, bits=float(st["bits"]), cause=cause)

    # ---------------------------------------------------------- audit
    def audit(self, models: Optional[Mapping[int, Any]] = None) -> HoldoutAudit:
        """Replay the holdout from its log and custody, under the book lock
        so that no write lands mid-replay: the hash chain, the custody head,
        the dates and the id first (a log that fails them is not replayed,
        so an edited log is never answered); then the seal's commitments
        against the custody files, and every operation through the state
        machine (every answer and its codelength recomputed from the stored
        predictions, every certificate rebuilt and matched to its record in
        the book, every logged state compared in full); then the book
        around the holdout: the log's certificates registered once, as
        performance certificates, and active; no other active record naming
        the holdout (in any field but its peeks); none registered on rows
        while they were in its custody; none before the seal naming it or
        its rows. models, keyed by log number
        (Answer.seq), are re-run on the released rows and matched to the
        logged hashes, after the lock and only when the replay passed with
        every row released; that answers questions about the rows, so it is
        refused while any row is in custody."""
        models = dict(models or {})
        with self.ledger.locked(), np.errstate(all="ignore"):
            self.ledger.refresh(force=True)
            ops = self._ops()
            if models:
                try:
                    sealed = self._sealed_lineage(dict(ops[-1].get("state", {})))
                except (ValueError, FileNotFoundError, KeyError):
                    sealed = self.lineage
                if not sealed.is_empty():
                    raise ValueError(f"audit refused: re-running models on rows still in custody would answer questions outside the log of {self.id}; audit the models after retire() has released every slice")
            rows, sim = self._audit_log(ops)
        if models:
            if not all(r[1] == "PASS" for r in rows) or sim is None or not (sim["tuning"] == "retired" and sim["exam"] in ("opened", "none")):
                rows.append(["models", "FAIL", "not re-run: the log did not replay to a state with every row released"])
            else:
                _, _, tune, ex = self._data()
                answered = {o["seq"]: o for o in ops[1:] if o.get("action") in ("ask", "check", "open")}
                for k in sorted(models):
                    o = answered.get(int(k))
                    if o is None:
                        rows.append([f"#{k} model", "FAIL", f"no answer at #{k} in the log"])
                        continue
                    rr = tune if (o["action"] in ("ask", "check") or o.get("slice") == "tuning") else ex
                    q = self._predict_on(models[k], rr)
                    rows.append([f"#{k} model", "PASS" if self._model_sha(q) == o["model_sha"] else "FAIL", "re-run on the released rows, matched to the logged hash"])
        return HoldoutAudit(holdout=self.id, rows=rows, ok=all(r[1] == "PASS" for r in rows))

    def _audit_log(self, ops: list) -> tuple:
        """The audit rows for the log and the book, and the replayed final
        state (None when the replay stopped)."""
        rows = []
        ok_chain = all(ops[i].get("seq") == i for i in range(len(ops))) and ops[0].get("prev") is None and all(ops[i].get("prev") == _op_hash(ops[i - 1]) for i in range(1, len(ops)))
        rows.append(["hash chain", "PASS" if ok_chain else "FAIL", f"{len(ops)} operations, head {_op_hash(ops[-1])[:16]}"])
        try:
            head = json.loads(self._read("head.json"))
            ok_head = head.get("seq") == ops[-1].get("seq") and head.get("hash") == _op_hash(ops[-1])
            rows.append(["custody head", "PASS" if ok_head else "FAIL", f"custody anchors #{head.get('seq')}; the log ends at #{ops[-1].get('seq')}"])
        except (FileNotFoundError, ValueError):
            rows.append(["custody head", "FAIL", "head.json missing or unreadable"])
        try:
            ats = [datetime.fromisoformat(o["at"]) for o in ops]
            rows.append(["dates", "PASS" if all(ats[i] <= ats[i + 1] for i in range(len(ats) - 1)) else "FAIL", f"sealed {ops[0]['at']}; every later operation dated after the seal"])
        except (KeyError, ValueError, TypeError):
            rows.append(["dates", "FAIL", "an operation carries no valid date"])
        seal0 = dict(ops[0])
        hid = seal0.pop("holdout", None)
        rows.append(["holdout id", "PASS" if "HO-" + _sha(_canon(seal0))[:16].upper() == hid else "FAIL", "id = hash of the seal record"])
        if any(r[1] == "FAIL" for r in rows):
            rows.append(["replay", "FAIL", "not replayed: the log does not match its chain, its custody head, its dates or its id"])
            return rows, None
        cm, pr = self.commit, self.protocol
        try:
            c = self._load(force=True)
            X, y, tune, ex = c["X"], c["y"], c["tune"], c["exam"]
            rows.append(["custody rows", "PASS", "x, y, split" + ("" if cm["clock_sha"] is None else ", clock") + ("" if cm["baseline_sha"] is None else ", baseline") + " match the seal's salted commitments"])
        except Exception as e:
            rows.append(["custody rows", "FAIL", str(e) or type(e).__name__])
            return rows, None
        rows.append(["row identities", "PASS" if _sha(_canon(ops[0]["lineage"])) == cm["rows_sha"] else "FAIL", f"rows_sha {cm['rows_sha'][:16]}"])
        init = {"asks": 0, "accepts": 0, "bits": 0.0, "bar": tr.z_bar(pr["delta"], 0.0), "clean_share": 1.0, "checks": 0, "failures": 0, "gammas": [], "tuning": "open", "exam": "sealed" if cm["n_exam"] else "none", "incumbent": 0, "family": None, "certificates": []}
        rows.append(["seal state", "PASS" if _same_state(ops[0]["state"], init) else "FAIL", "the initial state the protocol implies"])
        sim = init
        books = {}
        for o in self.ledger._registered():
            books.setdefault(o["id"], o["record"])
        voided = self.ledger._voided_ids()
        for o in ops[1:]:
            a = o.get("action")
            tag = f"#{o.get('seq')} {a}"
            try:
                if a not in ACTIONS:
                    raise _Fail(f"unknown action {a!r}")
                p = None
                if a in ("ask", "check", "open"):
                    try:
                        p = self._pred(o["seq"], want=o.get("model_sha") or "-")
                    except FileNotFoundError:
                        raise _Fail("stored predictions missing")
                    except ValueError:
                        raise _Fail("stored predictions do not match the logged hash")
                with np.errstate(all="ignore"):
                    sim, detail = self._replay(o, sim, p, tune, ex, ops, books, voided)
                if not _same_state(o.get("state", {}), sim):
                    raise _Fail("the logged state differs from the replay")
                rows.append([tag, "PASS", detail])
            except (_Fail, ValueError, FileNotFoundError, KeyError, TypeError, ArithmeticError) as e:
                rows.append([tag, "FAIL", str(e) or type(e).__name__])
                return rows, None
        rows.append(["book", *self._book_row(ops, sim, books, voided, tune, ex)])
        return rows, sim

    def _book_row(self, ops: list, sim: dict, books: dict, voided: dict, tune: np.ndarray, ex: np.ndarray) -> list:
        """The book around the holdout, walked in order. The log's
        certificates: each registered once, as a performance certificate,
        and active. Every other record FAILs when it is active and names the
        holdout in any field but its peeks; when it was registered while
        rows it tests were in the holdout's custody, voided or not; or when
        it precedes the seal and names the holdout, or names a sealed row
        (test or proposal) without being voided before the seal (the seal
        refuses such rows)."""
        bad = []
        for cid in sim["certificates"]:
            if cid not in books:
                bad.append(f"{cid} missing from the book")
            elif cid in voided:
                bad.append(f"{cid} voided in the book ({voided[cid]})")
        mine = {o.get("certificate") for o in ops if o.get("certificate")}
        H = self._sub_lineage(np.sort(np.concatenate([tune, ex]).astype(int)))
        custody = {"all": H, "exam": self._sub_lineage(ex) if len(ex) else Lineage(), "none": Lineage()}
        phase, seen, early, gone = None, set(), [], set()
        for o in self.ledger._ops:
            op = o.get("op")
            if op == "holdout" and o.get("holdout") == self.id:
                if phase is None:
                    bad += [f"{cid} precedes the seal of {self.id}, spends its rows and was not voided before it" for cid in early if cid not in gone]
                s_ = o.get("state") or {}
                phase = "all" if s_.get("tuning") != "retired" else ("exam" if s_.get("exam") == "sealed" else "none")
                continue
            if op == "void":
                gone.add(o.get("id"))
                continue
            if op not in ("register", "refine"):
                continue
            cid, rec = o.get("id"), o.get("record") or {}
            if cid in mine:
                if phase is not None and op == "register" and rec.get("kind") == CertificateKind.PERFORMANCE.value and cid not in seen:
                    seen.add(cid)
                else:
                    bad.append(f"{cid}, an id of the log of {self.id}, is registered again or as another kind")
                continue
            try:
                c = from_dict(rec)
            except Exception:
                bad.append(f"{cid}: record unreadable")
                continue
            named = self.id in json.dumps({k: v for k, v in rec.items() if k != PEEK_FIELD}, sort_keys=True, default=str)
            if phase is None:
                if named:
                    bad.append(f"{cid} names {self.id} before its seal")
                elif identity_overlap(_known(c.full_lineage()), H):
                    early.append(cid)
                continue
            if named and cid not in voided:
                bad.append(f"{cid} names {self.id} but is not in its log")
                continue
            if phase == "none":
                continue
            ov = identity_overlap(c.ell_test, custody[phase])
            if ov:
                what = "the whole source" if any(x.get("rows") is None for x in ov) else f"{sum(len(x['rows']) for x in ov)} row(s)"
                bad.append(f"{cid} registered on {what} in the custody of {self.id} and not in its log" + (f" (voided: {voided[cid]})" if cid in voided else ""))
        return ["FAIL" if bad else "PASS", "; ".join(bad) if bad else f"{len(sim['certificates'])} log certificate(s) active; no other active record names it or sits on custody rows"]

    @_quiet
    def _replay(self, o: dict, sim: dict, p, tune, ex, ops: list, books: dict, voided: dict) -> tuple:
        """One state transition of the audit, with its validity conditions."""
        pr = self.protocol
        a = o["action"]
        new = dict(sim)
        if a == "ask":
            if pr["mode"] != "comparative" or sim["tuning"] != "open":
                raise _Fail(f"an ask while the tuning slice is {sim['tuning']} (mode {pr['mode']})")
            d = self._losses_on(self._incumbent_pred(sim["incumbent"], tune), tune) - self._losses_on(p, tune)
            acc = _ladder(d, pr["eta"])
            b = tr.answer_bits(pr["code"], sim["asks"], sim["accepts"], acc)
            L = sim["bits"] + b
            if o.get("answer") != ("accept" if acc else "reject") or abs(o.get("answer_bits", -1.0) - b) > 1e-9:
                raise _Fail(f"logged {o.get('answer')}; the replay gives {'accept' if acc else 'reject'} at {b:.4f} bits")
            s_ = tr.clean_share(pr["delta"], L)
            new.update({"asks": sim["asks"] + 1, "accepts": sim["accepts"] + int(acc), "bits": L, "bar": tr.z_bar(pr["delta"], L), "clean_share": s_, "incumbent": o["seq"] if acc else sim["incumbent"]})
            crossed = s_ <= pr["retire_at"]
            if crossed != bool(o.get("retire")):
                raise _Fail("the retirement threshold and the logged retirement disagree")
            if crossed:
                new["tuning"] = "retired"
            return new, f"{o['answer']} at {b:.4f} bits; spent {L:.4f}" + ("; retired at the threshold" if crossed else "")
        if a == "retire":
            if sim["tuning"] == "retired":
                raise _Fail("a second retirement")
            new["tuning"] = "retired"
            cause = str(o.get("cause") or "")
            return new, "declared" + (f": {cause}" if cause and cause != "declared" else "")
        if a == "family":
            if pr["mode"] != "numeric":
                raise _Fail("a family certificate on a comparative holdout")
            rec = books.get(o["certificate"])
            if rec is None or rec.get("kind") != CertificateKind.PERFORMANCE.value or content_id(rec) != o["certificate"]:
                raise _Fail(f"family certificate {o['certificate']} missing from the book, altered, or not a performance certificate")
            if o.get("supersedes") != sim.get("family"):
                raise _Fail("the family op does not supersede the current family")
            if sim.get("family") is not None and sim["family"] not in voided:
                raise _Fail(f"superseded family {sim['family']} was not voided")
            new["family"] = o["certificate"]
            new["certificates"] = [c for c in sim["certificates"] if c != sim.get("family")] + [o["certificate"]]
            return new, f"Guess and Check family certificate {o['certificate']}"
        if a == "check":
            if pr["mode"] != "numeric" or sim["tuning"] != "open" or sim.get("family") is None:
                raise _Fail(f"a check while the tuning slice is {sim['tuning']}, or without its family certificate")
            sv, (lo, hi) = self._statistic(p, tune)
            span = hi - lo
            a_h = float(((sv - lo) / span).mean())
            i = sim["checks"] + 1
            bits_i = tr.gnc_bits(i, sim["gammas"])
            tol = tr.hoeffding_tol_bits(pr["delta"], bits_i, len(tune))
            tau = o["width"] / span
            gammas = list(sim["gammas"])
            if abs((o["guess"] - lo) / span - a_h) <= tau - tol:
                out, rel = "confirmed", o["guess"]
            else:
                g = tau - tol
                if g > 0:
                    out, rel = "failed", lo + tr.gnc_round(a_h, g) * span
                    gammas.append(g)
                else:
                    out, rel = "halted", None
            good = o.get("outcome") == out and ((rel is None and o.get("released") is None) or (rel is not None and o.get("released") is not None and abs(rel - o["released"]) < 1e-9)) and abs(o.get("query_bits", -1.0) - bits_i) < 1e-9
            if not good:
                raise _Fail(f"logged {o.get('outcome')}; the replay gives {out}")
            new.update({"checks": i, "failures": sim["failures"] + int(out != "confirmed"), "gammas": gammas, "bits": tr.gnc_bits(i + 1, gammas), "tuning": "halted" if out == "halted" else sim["tuning"]})
            return new, f"{out}; level {tr.gnc_level(pr['delta'], i, sim['gammas']):.4g}"
        if a == "open":
            if o.get("slice") == "exam":
                if sim["exam"] != "sealed":
                    raise _Fail(f"an exam open while the exam slice is {sim['exam']}")
                rr = ex
            else:
                if pr["route"] != "kraft" or sim["tuning"] != "open":
                    raise _Fail(f"a transcript certificate while the tuning slice is {sim['tuning']} (route {pr['route']})")
                rr = tune
            rec = books.get(o["certificate"])
            if rec is None or rec.get("kind") != CertificateKind.PERFORMANCE.value or content_id(rec) != o["certificate"]:
                raise _Fail(f"certificate {o['certificate']} missing from the book, altered, or not a performance certificate")
            cert, art, bar = self._certificate(p, rr, o["slice"], sim, _lineage_from_list(o["lineage"]), ops[: o["seq"]], timestamp=datetime.fromisoformat(o["timestamp"]))
            if content_id(to_dict(cert)) != o["certificate"]:
                raise _Fail("the certificate rebuilt from the log and custody differs from the one in the book")
            new["certificates"] = list(sim["certificates"]) + [o["certificate"]]
            if o["slice"] == "exam":
                new["exam"] = "opened"
            else:
                new["tuning"] = "certified"
            return new, f"{o['slice']} slice bound {art['bound']:.6g} at bar {bar:.4f}; certificate {o['certificate']}"
        raise _Fail(f"unknown action {a!r}")

    # ---------------------------------------------------------- lint and render
    def transcript(self) -> HoldoutTranscript:
        """The holdout's public log, one row per operation in log order: the
        log number (the key audit(models=...) takes), the action, the label
        given with the call, the answer returned, the bits that answer was
        priced at (an ask's codelength under the sealed code; a check's L_i,
        its level being delta 2^-L_i; the bits a transcript certificate's
        bar carried), and the codelength of the transcript after it (in
        numeric mode, the bits the next check is priced at). Nothing in it
        goes beyond what was returned to the analyst or registered in the
        book."""
        rows = []
        for o in self._ops():
            a, st = o.get("action"), o.get("state", {})
            bits = None
            if a == "seal":
                pr, cm = o["protocol"], o["commit"]
                ans = f"{pr['mode']} / {pr['route']}: {cm['n_tune']} tuning, {cm['n_exam']} exam rows"
            elif a == "ask":
                ans, bits = o["answer"].upper() + ("; the tuning slice retired at the threshold" if o.get("retire") else ""), float(o["answer_bits"])
            elif a == "check":
                bits = float(o["query_bits"])
                ans = "HALTED" if o["outcome"] == "halted" else f"{o['outcome'].upper()} {o['released']:.6g} +/- {o['width']:.6g}"
            elif a == "family":
                ans = "family certificate registered" + (f", superseding {o['supersedes']}" if o.get("supersedes") else "")
            elif a == "open":
                bits = float(st.get("bits", 0.0)) if o.get("slice") == "tuning" else 0.0
                ans = f"{o.get('slice')} slice opened: bound {o['bound']:.6g}"
            else:
                ans = f"retired: {o.get('cause', '')}"
            rows.append({"seq": o.get("seq"), "action": a, "label": o.get("label", ""), "answer": ans, "bits": bits, "transcript_bits": float(st.get("bits", 0.0)), "certificate": o.get("certificate") or ""})
        return HoldoutTranscript(holdout=self.id, rows=rows)

    def runrecord(self) -> RunRecord:
        """The holdout's run as a declared record for lint: the issued
        certificate's test and proposal rows (the tuning slice and the
        bits spent until something is opened), the tuning transcript as a
        peek, and the bits its bar priced (the bits at the transcript
        certificate; none on the exam slice)."""
        pr, st = self.protocol, self.state
        ops = self._ops()
        opens = [o for o in ops if o["action"] == "open"]
        L = float(st["bits"])
        tune_lin = self.tuning_lineage()
        if opens:
            o = opens[-1]
            test = self.exam_lineage() if o["slice"] == "exam" else tune_lin
            prop = _lineage_from_list(o["lineage"]).union(self._baseline_lineage())
            peek_bits = float(o["state"]["bits"])
            priced = peek_bits if o["slice"] == "tuning" else 0.0
        else:
            test, prop, priced, peek_bits = tune_lin, self._baseline_lineage(), L, L
        rr = RunRecord(name=f"holdout {self.id}", instrumented=True, test_rows=test, proposal_rows=prop, q=float(pr["delta"]))
        if peek_bits > 0:
            rr.peeks.append(PeekDecl(rows=tune_lin, bits=peek_bits, holdout=self.id, label="tuning transcript"))
        rr.priced_bits = priced
        return rr

    def render(self) -> str:
        pr, st, cm = self.protocol, self.state, self.commit
        head = [["holdout", self.id], ["mode / route", f"{pr['mode']} / {pr['route']}"], ["claim", f"{pr['claim']} ({pr['loss']} loss{'' if pr['loss_range'] is None else ', range ' + str(pr['loss_range'])})"], ["rows (tuning / exam)", f"{cm['n_tune']} / {cm['n_exam']}"], ["delta", f"{pr['delta']:g}"]]
        if pr["mode"] == "comparative":
            head += [["code / eta", f"{pr['code']} / {pr['eta']:g}"], ["asks (accepts)", f"{st['asks']} ({st['accepts']})"], ["bits spent", f"{st['bits']:.3f}"], ["transcript bar z_L", f"{st['bar']:.4f}"], ["clean share s(L)", f"{st['clean_share']:.4f}"], ["retirement", f"s(L) <= {pr['retire_at']:g} (at {pr['retire_bits']:.2f} bits)"]]
        else:
            head += [["checks (failures)", f"{st['checks']} ({st['failures']})"], ["transcript bits", f"{st['bits']:.3f}"]]
            if st["tuning"] == "open":
                nb = tr.gnc_bits(st["checks"] + 1, list(st["gammas"]))
                head += [["next check priced at", f"{nb:.3f} bits (level {pr['delta'] * 2.0 ** -nb:.3g})"], ["min width, next check", f"{self.min_width():.6g}"]]
        if pr.get("rowwise"):
            head += [["predictions", "one row per call (declared outcome columns reach other sealed rows)"]]
        head += [["tuning slice", st["tuning"]], ["exam slice", st["exam"]], ["certificates", ", ".join(st["certificates"]) or "none"]]
        return "\n".join([f"-- holdout {self.id} --", tabulate(head, tablefmt="psql")])

    def __str__(self):
        return self.render()

    def __repr__(self):
        return f"Holdout({self.id!r}, book={self.ledger.path!r})"


class _Fail(Exception):
    pass


def _write_atomic(path: str, data: bytes) -> None:
    """Write a small file by rename, so a reader sees the old bytes or the
    new ones."""
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


@_quiet
def _ladder(d: np.ndarray, eta: float) -> bool:
    """The comparative rule: accept iff the mean paired gain exceeds eta
    paired standard errors (a strictly positive mean when the differences
    are constant)."""
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    return bool(m > eta * se) if se > 0 else bool(m > 0)


def _same_state(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if k == "gammas":
            if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y) or any(abs(float(u) - float(v)) > 1e-12 for u, v in zip(x, y)):
                return False
        elif isinstance(x, float) or isinstance(y, float):
            if x is None or y is None or isinstance(x, bool) or isinstance(y, bool) or abs(float(x) - float(y)) > 1e-9 * max(1.0, abs(float(y))):
                return False
        elif x != y:
            return False
    return True
