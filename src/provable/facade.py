"""The workflow facade: certify, reuse, lint, examine.

Verbs are named after the analyst's situation; outputs keep the theory's
terminology, and every priced number carries its source statement and
exactness grade. The engine underneath is untouched: certify composes
CertifiedDiscovery, examine composes the transported spectrum, and the
byte-neutrality of the composition is a standing test.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from tabulate import tabulate

from .sieve import CertifiedDiscovery
from .certificates import (Certificate, Findings, Ledger, Lineage, RunRecord, admit, certificate_from_run, run_checks, runrecord_from_dict, runrecord_from_ledger, source_fingerprint, source_identity)
from .certificates.issuance import _slice_for
from .certificates.records import GuaranteeKind, eval_atom_spec
from .certificates.lineage import Slice
from datetime import datetime, timezone

PROFILES = {"default": {}, "fast": {"rounds": 2, "stability_reps": 0, "n_harvest": 20}}


class ManifestMismatch(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _instrument_sha() -> str:
    try:
        with open(__file__, "rb") as fh:
            return _sha(fh.read())
    except OSError:
        return "unavailable"


def _manifest(cd, X: pd.DataFrame, y, params: Mapping[str, Any]) -> dict:
    from . import __version__
    return {
        "x_sha": source_fingerprint(pd.DataFrame(X).reset_index(drop=True)),
        "y_sha": _sha(np.asarray(y, float).ravel().tobytes()),
        "dictionary_sha": _sha("|".join(cd.atom_names_).encode()),
        "config_sha": _sha(json.dumps({k: repr(v) for k, v in sorted(params.items())}, sort_keys=True).encode()),
        "instrument_sha": _instrument_sha(),
        "seed": int(cd.seed),
        "provable": __version__,
    }


def _penumbra_block(cd) -> dict:
    """The corrected penumbra: raw band mass, the estimated null floor
    (Storey null count from the residual-scan p-values times the exact
    per-null band mass 2(phi(1) - z phi(z)) at z = zbar_pool), and the
    corrected value. Source: instrument paper Thm 6a; the floor is the
    null expectation of the band statistic."""
    d = cd.diag_
    raw = float(d["penumbra_score"])
    z = float(d["zbar_pool"])
    scan_t = np.asarray(d.get("scan_t", []), float)
    if scan_t.size == 0:
        return {"raw": raw, "floor": None, "m0": None, "corrected": None, "grade": "unavailable: scan not exported"}
    p_scan = 2 * norm.sf(np.abs(scan_t))
    m0 = float(min(len(p_scan), np.sum(p_scan > 0.5) / 0.5))
    per_null = 2 * (norm.pdf(1.0) - z * norm.pdf(z))
    floor = m0 * per_null
    return {"raw": raw, "floor": floor, "m0": m0, "per_null": per_null, "corrected": max(0.0, raw - floor), "grade": "est: Storey null count at lambda=0.5", "note": "floor is the fixed-dictionary null expectation; with a data-adaptive dictionary (cart or harvested cuts) the corrected value retains genuine selection mass"}


def _selective_ci(cd) -> Optional[pd.DataFrame]:
    """Selective intervals on the certified set at alpha = khat*q/m over
    the core family (FCR); the marginal intervals remain in the machine
    record for continuity."""
    if cd.core_res_ is None or not cd.cert_sel_.any():
        return None
    m = len(cd.core_)
    alpha_sel = float(cd.khat_ * cd.fdr_q / m)
    ci = np.asarray(cd.core_res_.conf_int(alpha=alpha_sel))[1:]
    rows = []
    for pos in np.where(cd.cert_sel_)[0]:
        j = cd.core_[pos]
        sd = float(cd._sd[j])
        rows.append({"atom": cd.atom_names_[j], "atom_id": int(j), "ci_lo_sel": ci[pos, 0] / sd, "ci_hi_sel": ci[pos, 1] / sd})
    out = pd.DataFrame(rows)
    out.attrs["alpha_sel"] = alpha_sel
    return out


def _atom_card(cd, unit_cost=None) -> Optional[pd.DataFrame]:
    """Per certified atom, the user-facing marginal effects, labeled
    est and never budget-bearing: firing rate, when-fired effect
    (E[y|fired] minus E[y|not fired]) with its se, and the agreement
    rate of sign(y) with the certified direction, each on S2 and on
    the held-out test rows. Not computed on the proposal rows, which
    selected the atoms and would inflate every number. With a
    declared unit cost, a net line in the certified direction."""
    if not cd.certified_:
        return None
    eff = np.asarray(cd.core_res_.params)[1:]
    rows = []
    for pos in np.where(cd.cert_sel_)[0]:
        j = cd.core_[pos]
        d = 1 if eff[pos] >= 0 else -1
        rec = {"atom": cd.atom_names_[j], "direction": d}
        raw = cd.Z[:, j] * cd._sd[j] + cd._mu[j]
        for tag, idx in (("s2", cd.idx_s2), ("test", cd.idx_test)):
            idx = np.asarray(idx, int)
            f = raw[idx] > 0.5
            yy = np.asarray(cd.y, float)[idx]
            nf = int(f.sum())
            rec[f"fire_{tag}"] = nf / len(idx)
            if nf and nf < len(idx):
                a, b = yy[f], yy[~f]
                rec[f"eff_{tag}"] = float(a.mean() - b.mean())
                rec[f"se_{tag}"] = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
                rec[f"agree_{tag}"] = float((np.sign(a) == d).mean())
            else:
                rec[f"eff_{tag}"] = rec[f"se_{tag}"] = rec[f"agree_{tag}"] = float("nan")
        if unit_cost is not None:
            rec["net_test"] = d * rec["eff_test"] - 2.0 * float(unit_cost)
        rows.append(rec)
    return pd.DataFrame(rows)


@dataclass
class Report:
    """The standard report of an instrumented run. Wraps the engine's
    report_ dict verbatim as .engine (one report, two views)."""
    cd: Any
    manifest: dict
    claims: pd.DataFrame
    alpha_sel: Optional[float]
    penumbra: dict
    atom_card: Optional[pd.DataFrame]
    unit_cost: Optional[float]
    certificate: Optional[Certificate]
    ledger_path: Optional[str]
    statement: Optional[Any]
    dataset_tag: str

    @property
    def engine(self) -> dict:
        return self.cd.report_

    @property
    def state(self) -> dict:
        c = self.engine["certification"]
        return {"n": self.engine["setup"]["n"], "P": c["pool"], "core": c["core"], "k": c["k"], "khat": c["khat"], "q": c["q"]}

    def engine_vs_core(self) -> dict:
        """The finding row: frozen-dictionary certified core vs the free
        engine on test R2. A near-zero lead says the dictionary spans
        what the engine sees; a positive lead is mass dark to the
        dictionary but visible to the engine."""
        perf = self.engine["performance"]
        rows = perf if isinstance(perf, dict) else {r.get("model"): r for r in perf}
        core_k = next((k for k in rows if str(k).startswith("certified core [train-fit]")), None)
        eng_k = next((k for k in rows if str(k).startswith("xgboost")), None)
        if core_k is None or eng_k is None:
            return {"core_r2": None, "engine_r2": None, "lead": None}
        def r2(v):
            return float(v.get("test R2", v.get("test_r2"))) if isinstance(v, dict) else float(v[1])
        c, e = r2(rows[core_k]), r2(rows[eng_k])
        return {"core_r2": c, "engine_r2": e, "lead": e - c}

    @property
    def refusal(self) -> Optional[str]:
        ph = self.cd.diag_["phase"]
        return ph if str(ph).startswith("I ") else None

    def runrecord(self) -> RunRecord:
        cd = self.cd
        test = Lineage.of(_slice_for(cd, cd.idx_s2, self.dataset_tag))
        prop = Lineage.of(_slice_for(cd, cd.idx_s1, self.dataset_tag))
        pv = [float(x) for x in np.asarray(cd.core_res_.pvalues)[1:]] if cd.core_res_ is not None else None
        cert_idx = list(np.where(cd.cert_sel_)[0]) if cd.core_res_ is not None else None
        return RunRecord(name="instrumented run", instrumented=True, test_rows=test, proposal_rows=prop, q=float(cd.fdr_q), m=len(cd.core_), khat=int(cd.khat_), pvalues=pv, certified_idx=cert_idx)

    def core_fit(self, X=None) -> np.ndarray:
        """The netting object: the certified tier's fitted value as a
        column. The certified tier is one OLS of the target on the
        certified atoms alone (canonical refit scope), so its fitted
        value is the joint fitted value of every certified cell for a
        row, the number the atom card's per-atom marginals do NOT sum
        to. With X=None, the run's own rows; with X, fresh rows
        through the stored dictionary."""
        if X is None:
            base = self.cd.cert_pred_all_
            if isinstance(base, dict):
                base = base[self.cd.refit_scope]
            return np.asarray(base, float).copy()
        return np.asarray(self.cd.predict(X, model="certified"), float)

    def to_json(self) -> str:
        def num(x):
            if isinstance(x, (np.floating,)):
                return float(x)
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            return str(x)
        payload = {"schema": "provable/report/v1", "manifest": self.manifest, "state": self.state, "engine_vs_core": self.engine_vs_core(), "atom_card": None if self.atom_card is None else self.atom_card.to_dict("records"), "claims": self.claims.to_dict("records"), "alpha_sel": self.alpha_sel, "penumbra": self.penumbra, "certificate_id": None if self.certificate is None else self.certificate.id, "ledger": self.ledger_path, "engine": self.engine}
        return json.dumps(payload, default=num, sort_keys=True)

    def render(self) -> str:
        out = ["-- certify: standard report --"]
        out.append(tabulate([[k, v] for k, v in self.manifest.items()], headers=["manifest", ""], tablefmt="psql"))
        s = self.state
        out.append(tabulate([[s["n"], s["P"], s["core"], s["k"], s["khat"], s["q"]]], headers=["n", "P", "core", "k cert", "khat", "q"], tablefmt="psql"))
        ec = self.engine_vs_core()
        if ec["lead"] is not None:
            out.append(f"engine vs certified core (test R2): engine {ec['engine_r2']:.4g} vs core {ec['core_r2']:.4g}; lead {ec['lead']:+.4g}")
        if len(self.claims):
            cols = [c for c in ("atom", "effect", "ci_lo", "ci_hi", "ci_lo_sel", "ci_hi_sel", "t", "p") if c in self.claims.columns]
            out.append("-- certified claims (selective intervals at alpha_sel = khat*q/m = %.3g; FCR) --" % (self.alpha_sel or float("nan")))
            out.append(tabulate(self.claims[cols].head(15).values.tolist(), headers=cols, tablefmt="psql", floatfmt=".4g"))
        if self.atom_card is not None and len(self.atom_card):
            out.append("-- atom card (est; marginal per-atom effects on S2 and held-out test, never the proposal rows; net across cells via rep.core_fit()) --")
            pc = self.atom_card
            cols = [c for c in ("atom", "direction", "fire_s2", "fire_test", "eff_s2", "se_s2", "eff_test", "se_test", "agree_s2", "agree_test", "net_test") if c in pc.columns]
            out.append(tabulate(pc[cols].values.tolist(), headers=cols, tablefmt="psql", floatfmt=".4g"))
            if self.unit_cost is not None:
                out.append(f"(net_test = direction x eff_test - 2 x {self.unit_cost:g}, declared unit cost)")
        p = self.penumbra
        if p.get("floor") is not None:
            out.append("-- penumbra (Thm 6a band; null floor subtracted, est) --")
            out.append(tabulate([[f"{p['raw']:.2f}", f"{p['m0']:.1f}", f"{p['floor']:.2f}", f"{p['corrected']:.2f}", p["grade"]]], headers=["raw band mass", "m0 (est nulls)", "null floor", "corrected", "grade"], tablefmt="psql"))
        if self.refusal:
            out.append(f"REFUSAL: {self.refusal}")
        if self.certificate is not None:
            out.append(f"ledger: 1 carrier certificate appended to {self.ledger_path} ({self.certificate.id})")
        elif self.ledger_path:
            out.append(f"ledger: nothing certified; no certificate issued ({self.ledger_path})")
        if self.statement is not None:
            out.append(f"book after this run: charged {self.statement.charged:.4f} of budget {self.statement.budget:.4f}, remaining {self.statement.remaining:.4f}")
        return "\n".join(out)

    def __str__(self):
        return self.render()


def certify(X, y, q: float = 0.10, ledger: Optional[str] = None, profile: str = "default", seed: int = 0, dataset_tag: str = "panel", clock=None, expect_manifest: Optional[Mapping[str, Any]] = None, consumed: Iterable[Certificate] = (), unit_cost: Optional[float] = None, verbose: bool = False, **expert) -> Report:
    """Run the instrument, issue the carrier certificate, return the
    standard report. The engine is composed, not modified; expert kwargs
    pass through to CertifiedDiscovery."""
    params = dict(PROFILES.get(profile, {}))
    params.update(expert)
    consumed = list(consumed)
    if consumed:
        pre = admit(consumed, Lineage(), level=q, on_stale="refuse")
    cd = CertifiedDiscovery(fdr_q=q, seed=seed, verbose=verbose, **params)
    cd.run(pd.DataFrame(X), np.asarray(y, float).ravel())
    man = _manifest(cd, X, y, dict(fdr_q=q, seed=seed, profile=profile, **expert))
    if expect_manifest is not None:
        exp = dict(expect_manifest)
        bad = {k: (exp[k], man.get(k)) for k in exp if str(man.get(k)) != str(exp[k])}
        if bad:
            lines = "; ".join(f"{k}: expected {a}, saw {b}" for k, (a, b) in bad.items())
            raise ManifestMismatch(f"refused before any ledger write: manifest mismatch ({lines})")
    claims = cd.claims_.copy()
    sel = _selective_ci(cd)
    alpha_sel = None
    if sel is not None and len(claims):
        claims = claims.merge(sel[["atom_id", "ci_lo_sel", "ci_hi_sel"]], on="atom_id", how="left")
        alpha_sel = sel.attrs["alpha_sel"]
    pen = _penumbra_block(cd)
    cert, led_path, stmt = None, None, None
    if ledger is not None:
        led = ledger if isinstance(ledger, Ledger) else Ledger(str(ledger), budget=None)
        led_path = led.path
        test_rows = Lineage.of(_slice_for(cd, cd.idx_s2, dataset_tag, clock))
        adm = led.admit(rows=test_rows, level=q, on_stale="refuse")
        peeked = list(adm.peeked())
        if consumed:
            peeked += [v for v in admit(consumed, test_rows, level=q, on_stale="demote").items if v.verdict.name == "PEEKED"]
        if peeked:
            named = "; ".join(f"{v.cert_id} ({v.bits:.2f} bits)" for v in peeked)
            raise ValueError(f"certify refused: the certifying rows were seen through counted transcripts ({named}); a run on them holds its level only at q 2^-L, and certify does not price peeks; certify on rows no transcript saw")
        cert = certificate_from_run(cd, dataset_tag=dataset_tag, clock=clock, consumed=consumed)
        if cert is not None:
            led.register(cert)
        stmt = led.statement()
    return Report(cd=cd, manifest=man, claims=claims, alpha_sel=alpha_sel, penumbra=pen, atom_card=_atom_card(cd, unit_cost), unit_cost=unit_cost, certificate=cert, ledger_path=led_path, statement=stmt, dataset_tag=dataset_tag)


def reuse(certs, rows: Lineage, ledger: Optional[str] = None, level: Optional[float] = None, mode: str = "guidance", on_stale: str = "refuse"):
    """Package-level spelling of the ledger's admit: the price of reuse,
    quoted in its own currency (an error-rate verdict, inflation, and the
    pass level for the next test)."""
    if ledger is not None:
        led = ledger if isinstance(ledger, Ledger) else Ledger(str(ledger))
        return led.admit(certs=certs, rows=rows, level=level, mode=mode, on_stale=on_stale)
    if level is None:
        raise ValueError("reuse without a ledger needs an explicit level (the pass level for the next test)")
    return admit(list(certs), rows, level=float(level), mode=mode, on_stale=on_stale)


def lint(target, ledger: Optional[Ledger] = None) -> Findings:
    """Post-hoc R1-R7 over a run, a RunRecord, a declared dict, a Report,
    a Holdout, or a ledger path. One row per finding with the exact price
    and the one-line repair where one exists."""
    from .holdout import Holdout
    if isinstance(target, Findings):
        return target
    if isinstance(target, Holdout):
        return run_checks(target.runrecord(), ledger=target.ledger)
    if isinstance(target, Report):
        return run_checks(target.runrecord(), ledger=ledger)
    if isinstance(target, RunRecord):
        return run_checks(target, ledger=ledger)
    if isinstance(target, Ledger):
        return run_checks(runrecord_from_ledger(target), ledger=target)
    if isinstance(target, str) and os.path.isdir(target) and os.path.exists(os.path.join(target, Ledger.LEDGER_FILE)):
        led = Ledger(target)
        return run_checks(runrecord_from_ledger(led), ledger=led)
    if isinstance(target, Mapping):
        return run_checks(runrecord_from_dict(target), ledger=ledger)
    raise TypeError("lint accepts a Report, RunRecord, declared dict, Holdout, Ledger, or ledger path")


# ------------------------------------------------------------------ grade

@dataclass
class GradeReport:
    """The scorecard of one grading pass: per pre-committed cell, the
    when-fired evidence on fresh rows and a verdict at the graded bar.
    Budget-bearing: the graded level charges the book (or consumes the
    named reservation); the entry lands in the statement's grades
    section."""
    certificate_id: str
    admission_verdict: str
    level: float
    bar: float
    k: int
    reservation: Optional[str]
    cells: pd.DataFrame
    ledger_path: Optional[str]

    def summary(self) -> dict:
        v = self.cells["verdict"].value_counts().to_dict()
        return {"confirmed": int(v.get("CONFIRMED", 0)), "reversed": int(v.get("REVERSED", 0)), "not_confirmed": int(v.get("NOT CONFIRMED", 0))}

    def render(self) -> str:
        out = [f"-- grade: {self.certificate_id} --"]
        out.append(f"admission: {self.admission_verdict}; graded level {self.level:g} over k={self.k} cells; one-sided bar {self.bar:.4g} per cell" + (f" (reservation {self.reservation}, bar = alpha/|V|)" if self.reservation else " (bar = level/k)"))
        cols = [c for c in ("atom", "direction", "n_fired", "fire_rate", "mean_fired", "se", "agree", "p_confirm", "p_reverse", "verdict") if c in self.cells.columns]
        out.append(tabulate(self.cells[cols].values.tolist(), headers=cols, tablefmt="psql", floatfmt=".4g"))
        s = self.summary()
        out.append(f"scorecard: {s['confirmed']} CONFIRMED, {s['reversed']} REVERSED, {s['not_confirmed']} NOT CONFIRMED. REVERSED is significant evidence against the original direction at the graded bar, not merely absence of confirmation.")
        if self.ledger_path:
            out.append(f"book: entry appended to {self.ledger_path} (see the grades section of the statement); this grade is budget-bearing.")
        return "\n".join(out)

    def __str__(self):
        return self.render()


def grade(certificate: Certificate, X_new, y_new, ledger: Optional[str] = None, level: Optional[float] = None, reservation: Optional[str] = None, dataset_tag: str = "grade", clock=None) -> GradeReport:
    """Judge a certificate on fresh rows against its pre-committed
    directions. Admission runs first, so spent rows refuse and the pass
    level is quoted by the book; each certified cell is then tested
    one-sided in its committed direction on its own firings (the
    when-fired mean, deliberately marginal rather than
    the joint partial). Verdicts at the bar: CONFIRMED, REVERSED
    (significant evidence against the original direction), or NOT
    CONFIRMED. The entry is written to the book and charges it, unless
    it consumes a registered-vocabulary reservation, in which case the
    bar is alpha over the vocabulary size. On rows the certificate's
    proposal saw through a counted transcript (PEEKED, L bits) the cells
    are tested at level 2^-L and the book is charged the level itself,
    which is what the test then holds."""
    from scipy.stats import norm as _norm
    X_new = pd.DataFrame(X_new).reset_index(drop=True)
    yv = np.asarray(y_new, float).ravel()
    fam = certificate.guarantee.family
    names = fam.get("support_ordered")
    specs = fam.get("atoms_spec")
    dirs = fam.get("directions")
    if not names or not specs or not dirs:
        raise ValueError("certificate carries no pre-committed cell specs (support_ordered, atoms_spec, directions); only certificates issued by provable >= 0.10.0 are gradable")
    sid, aliases = source_identity(X_new)
    t0 = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if clock is not None:
        s0 = pd.to_datetime(pd.Series(clock[0])); s1 = pd.to_datetime(pd.Series(clock[1]))
        t0a, t1a = s0.min().to_pydatetime(), s1.max().to_pydatetime()
        t0a = t0a.replace(tzinfo=timezone.utc) if t0a.tzinfo is None else t0a
        t1a = t1a.replace(tzinfo=timezone.utc) if t1a.tzinfo is None else t1a
    else:
        t0a = t1a = t0
    rows = Lineage.of(Slice(source_id=sid, dataset_tag=str(dataset_tag), rows=frozenset(range(len(X_new))), t0=t0a, t1=t1a, aliases=aliases))
    led = None
    if ledger is not None:
        led = ledger if isinstance(ledger, Ledger) else Ledger(str(ledger))
        adm = led.admit(certs=[certificate], rows=rows, level=level, on_stale="refuse")
        nominal = led.budget if level is None else float(level)
    else:
        if level is None:
            raise ValueError("grade without a ledger needs an explicit level")
        adm = admit([certificate], rows, level=float(level), on_stale="refuse")
        nominal = float(level)
    if led is not None and level is None and reservation is None:
        rem = led.statement().remaining
        if rem <= 0:
            raise ValueError("book exhausted: no remaining budget to grade at; pass an explicit level or grade under a registered reservation")
        adm = led.admit(certs=[certificate], rows=rows, level=rem, on_stale="refuse")
        nominal = float(rem)
    alpha = float(adm.level)
    charge = nominal
    res_bar = None
    if reservation is not None:
        if led is None:
            raise ValueError("a reservation belongs to a ledger; pass the ledger it lives in")
        rops = [o for o in led._ops if o["op"] == "reserve" and o["id"] == reservation]
        if not rops or not rops[0].get("vocabulary"):
            raise ValueError(f"reservation {reservation} not found or carries no registered vocabulary")
        vocab = rops[0]["vocabulary"]
        missing = [n for n in names if n not in vocab]
        if missing:
            raise ValueError(f"cells outside the registered vocabulary cannot be graded under it: {missing[:3]}")
        alpha = float(rops[0]["level"])
        charge = alpha
        res_bar = alpha / len(vocab)
    k = len(names)
    bar = res_bar if res_bar is not None else alpha / k
    recs = []
    for nm, sp, d in zip(names, specs, dirs):
        f = eval_atom_spec(sp, X_new) > 0.5
        nf = int(f.sum())
        rec = {"atom": nm, "direction": int(d), "n_fired": nf, "fire_rate": nf / len(X_new)}
        if nf >= 2:
            a = yv[f]
            m, se = float(a.mean()), float(a.std(ddof=1) / np.sqrt(nf))
            z = m / se if se > 0 else 0.0
            rec.update({"mean_fired": m, "se": se, "agree": float((np.sign(a) == d).mean()), "p_confirm": float(_norm.sf(d * z)), "p_reverse": float(_norm.sf(-d * z))})
            rec["verdict"] = "CONFIRMED" if rec["p_confirm"] <= bar else ("REVERSED" if rec["p_reverse"] <= bar else "NOT CONFIRMED")
        else:
            rec.update({"mean_fired": float("nan"), "se": float("nan"), "agree": float("nan"), "p_confirm": float("nan"), "p_reverse": float("nan"), "verdict": "NOT CONFIRMED"})
        recs.append(rec)
    cells = pd.DataFrame(recs)
    rep = GradeReport(certificate_id=certificate.id, admission_verdict=adm.verdict.label, level=alpha, bar=float(bar), k=k, reservation=reservation, cells=cells, ledger_path=None if led is None else led.path)
    if led is not None:
        led.record_grade({"certificate": certificate.id, "level": charge, "graded_level": alpha, "bar": float(bar), "k": k, "summary": rep.summary(), "cells": cells.to_dict("records"), "data_sha": sid}, reservation=reservation)
    return rep


# ------------------------------------------------------------------ examine

@dataclass
class Anatomy:
    """One-call model anatomy: what the model is, where its structure
    sits, what each family would cost to remove, and what transports."""
    model: dict
    spectrum: dict
    families: pd.DataFrame
    display_top: int
    groups: Optional[pd.DataFrame]
    transport: Optional[dict]
    data_mode: str
    certificate_id: Optional[str]

    def render(self) -> str:
        out = ["-- examine: model anatomy --"]
        out.append(tabulate([[k, v] for k, v in self.model.items()], headers=["model", ""], tablefmt="psql"))
        out.append(tabulate([[k, v] for k, v in self.spectrum.items()], headers=["spectrum", ""], tablefmt="psql"))
        if len(self.families):
            out.append("-- families (magnitude = sum abs coefficient, sample-free ceiling; removal cost = mean squared family component on this data, priced before any edit; joint value = signed sum where all predicates fire) --")
            shown = self.families.head(self.display_top)
            out.append(tabulate(shown.values.tolist(), headers=list(shown.columns), tablefmt="psql", floatfmt=".4g"))
            if len(self.families) > self.display_top:
                out.append(f"({len(self.families) - self.display_top} further families enumerated; full table in .families)")
        if self.groups is not None and len(self.groups):
            out.append("-- groups --")
            out.append(tabulate(self.groups.values.tolist(), headers=list(self.groups.columns), tablefmt="psql", floatfmt=".4g"))
        if self.transport is not None:
            t = self.transport
            out.append(f"-- transport -- removal-cost share on supports outside the declared available set: {t['unavailable_share']:.1%} ({t['n_families_outside']} families); families entirely inside: {t['inside_share']:.1%}")
        if self.data_mode != "exact":
            out.append(f"DECLARED: prices computed under data mode '{self.data_mode}' (per-family complete rows; row counts in the families table)")
        return "\n".join(out)

    def __str__(self):
        return self.render()


def examine(model, X=None, groups: Optional[Mapping[str, str]] = None, available: Optional[Iterable[str]] = None, certificate: Optional[Certificate] = None, data: str = "exact", top: int = 15, min_order: int = 1) -> Anatomy:
    from .surface import load
    ad, spec = (model if isinstance(model, tuple) else load(model if isinstance(model, str) else None, booster=None if isinstance(model, str) else model))
    model_desc = ad.describe()
    spec_desc = spec.describe()
    fam = spec.families(X, by="removal_cost" if X is not None else "magnitude", top=10 ** 9, min_order=min_order, data=data)
    gdf = None
    if groups is not None and len(fam):
        rows = {}
        for _, r in fam.iterrows():
            feats = [f.strip() for f in str(r["support"]).split("*")]
            gset = sorted({groups.get(f, "other") for f in feats})
            key = " x ".join(gset)
            acc = rows.setdefault(key, {"group": key, "families": 0, "magnitude": 0.0, "removal_cost": 0.0})
            acc["families"] += 1
            acc["magnitude"] += float(r["magnitude"])
            if "removal_cost" in r and pd.notna(r["removal_cost"]):
                acc["removal_cost"] += float(r["removal_cost"])
        gdf = pd.DataFrame(sorted(rows.values(), key=lambda d: -d["removal_cost"]))
    transport = None
    if available is not None and X is not None and len(fam):
        avail = {str(a) for a in available}
        names = {v: k for k, v in ad.feature_names.items()} if isinstance(ad.feature_names, dict) else None
        outside_mask = []
        for _, r in fam.iterrows():
            feats = {f.strip() for f in str(r["support"]).split("*")}
            outside_mask.append(not feats <= avail)
        fam = fam.assign(outside=outside_mask)
        tot = float(fam["removal_cost"].sum()) or 1.0
        out_cost = float(fam.loc[fam["outside"], "removal_cost"].sum())
        transport = {"unavailable_share": out_cost / tot, "inside_share": 1.0 - out_cost / tot, "n_families_outside": int(fam["outside"].sum())}
    return Anatomy(model=model_desc, spectrum=spec_desc, families=fam, display_top=int(top), groups=gdf, transport=transport, data_mode=data, certificate_id=None if certificate is None else certificate.id)
