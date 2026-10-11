"""The reuse checks R1 to R6, with exact prices and source tags.

Every emitted row carries its check id and the statement of record it
enforces, per the rule-to-source map:

  R1 lineage validity            floors paper Lem 2.4; algebra Thm 3.1
  R2 fresh confirmations         algebra Prop 4.2
  R3 vacuous re-test             algebra Thm 5.2; floors Lem 6.1(c)
  R4 screen-then-gate            algebra Thm 5.3
  R5 no union at a common level  algebra Thm 6.5; Prop 6.3
  R6 refinement                  algebra Thm 7.2; Cor 7.6; Def 7.8
  R7 priced peek                 the transcript union bound (Rogers et al.
                                 2020, Thm 3.2): test rows seen through a
                                 counted transcript of L bits carry a test
                                 at level q to at most min(1, q 2^L); the
                                 exact repair tests at q 2^-L

Verdicts: PASS, VIOLATION, UNKNOWN (un-instrumented lineage; never a
silent pass), INFO (a priced observation that is not a violation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from tabulate import tabulate

from .admission import detect_screen_then_gate, verdict_for, Verdict
from .kit import confirmation_level, repair_pvalues, repaired_level, tau
from .lineage import identity_overlap, overlap_report
from .records import is_unknown, peeks_of, refines
from .runrecord import RunRecord

SOURCES = {
    "R1": "M2 Thm 3.1; U1 Lem 2.4",
    "R2": "M2 Prop 4.2",
    "R3": "M2 Thm 5.2; U1 Lem 6.1(c)",
    "R4": "M2 Thm 5.3",
    "R5": "M2 Thm 6.5; Prop 6.3",
    "R6": "M2 Thm 7.2; Cor 7.6; Def 7.8",
    "R7": "RRSSTW20 Thm 3.2",
}


@dataclass
class Finding:
    check: str
    source: str
    verdict: str
    price: str
    detail: str
    repair: Optional[dict] = None


@dataclass
class Findings:
    record_name: str
    rows: List[Finding] = field(default_factory=list)

    def add(self, check, verdict, price, detail, repair=None):
        self.rows.append(Finding(check=check, source=SOURCES[check], verdict=verdict, price=price, detail=detail, repair=repair))

    @property
    def violations(self) -> List[Finding]:
        return [r for r in self.rows if r.verdict == "VIOLATION"]

    @property
    def unknowns(self) -> List[Finding]:
        return [r for r in self.rows if r.verdict == "UNKNOWN"]

    def repair(self, check: str) -> dict:
        """The exact repair for a check, as numbers not prose."""
        for r in self.rows:
            if r.check == check and r.repair is not None:
                return r.repair
        raise KeyError(f"no repair recorded for {check}")

    def render(self) -> str:
        rows = [[r.check, r.verdict, r.price[:44], r.detail[:64], r.source] for r in self.rows]
        return "-- lint findings: " + self.record_name + " --\n" + tabulate(rows, headers=["check", "verdict", "price", "detail", "source of record"], tablefmt="psql")

    def __str__(self):
        return self.render()


def _fresh_state(a, b):
    if is_unknown(a) or is_unknown(b):
        return "UNKNOWN", []
    ov = identity_overlap(a, b)
    return ("VIOLATION" if ov else "PASS"), ov


def run_checks(rr: RunRecord, ledger=None) -> Findings:
    f = Findings(record_name=rr.name)

    # R1: every consumed certificate fresh for the test rows.
    if rr.consumed:
        for c in rr.consumed:
            v = verdict_for(c, rr.test_rows)
            if v.verdict == Verdict.UNKNOWN_LINEAGE:
                f.add("R1", "UNKNOWN", "lineage predates instrumentation", f"[{c.claim[:40]}] full lineage from the first instrumented run forward; UNKNOWN before", None)
            elif v.verdict == Verdict.NOT_FRESH:
                f.add("R1", "VIOLATION", "guarantee does not transport; demoted MISCALIBRATED unless refused (Cor 3.2)", f"[{c.claim[:40]}] {v.named_overlap()}", None)
            else:
                extra = " (dependence flag: cross-tag clock overlap)" if v.verdict == Verdict.FRESH_DEP_FLAG else ""
                if v.verdict == Verdict.PEEKED:
                    extra = f" (seen through a {v.bits:.2f}-bit transcript: priced under R7)"
                f.add("R1", "PASS", "guarantee preserved", f"[{c.claim[:40]}] identity-fresh for the test rows{extra}" if v.verdict == Verdict.PEEKED else f"[{c.claim[:40]}] fresh for the test rows{extra}", None)
    else:
        state, _ = _fresh_state(rr.proposal_rows, rr.test_rows)
        if state == "UNKNOWN":
            f.add("R1", "UNKNOWN", "lineage predates instrumentation", "proposal or test lineage un-instrumented", None)
        elif state == "PASS":
            f.add("R1", "PASS", "guarantee preserved", "proposal rows disjoint from test rows; any measurable search on them preserves the gate", None)
        else:
            _, ov = _fresh_state(rr.proposal_rows, rr.test_rows)
            f.add("R1", "VIOLATION", "gate guarantee not preserved", "proposal rows meet test rows: " + overlap_report(ov), None)

    # R3 before R2: a re-test on the certifying rows is vacuous.
    fresh_retests = []
    for rt in rr.retests:
        if is_unknown(rt.rows) or is_unknown(rr.test_rows):
            f.add("R3" if rt.on_certified else "R2", "UNKNOWN", "lineage predates instrumentation", f"re-test {rt.label or ''} lineage unknown", None)
            continue
        ov = identity_overlap(rt.rows, rr.test_rows)
        if rt.on_certified and ov:
            rep = None
            if rr.pvalues is not None and rr.q is not None and rr.certified_idx:
                p = np.asarray(rr.pvalues, float)
                taus = {int(i): tau(p, float(rr.q), int(i)) for i in rr.certified_idx}
                rep = {"confirmation_levels": {i: confirmation_level(rt.level, t) for i, t in taus.items()}, "rule": "confirm at alpha*tau_i (leave-one-in)"}
            f.add("R3", "VIOLATION", "re-test at alpha on the certifying rows certifies every selected member (empirical rate 1.0)", f"re-test {rt.label or ''} rows meet the certifying rows: {overlap_report(ov)}", rep)
        elif ov:
            f.add("R1", "VIOLATION", "gate guarantee not preserved", f"re-test {rt.label or ''} rows meet the certifying rows: {overlap_report(ov)}", None)
        else:
            fresh_retests.append(rt)

    # R2: fresh confirmations multiply error rates.
    if fresh_retests:
        prod = 1.0
        for rt in fresh_retests:
            prod *= float(rt.level)
        f.add("R2", "INFO", f"joint false-confirmation level = product = {prod:.3g}", f"{len(fresh_retests)} fresh confirmation(s) at levels {[rt.level for rt in fresh_retests]}", {"joint_level": prod})

    # R4: screen-then-gate on shared rows.
    if rr.screen is not None and rr.q is not None:
        if is_unknown(rr.screen.rows) or is_unknown(rr.test_rows):
            f.add("R4", "UNKNOWN", "lineage predates instrumentation", "screen lineage unknown", None)
        else:
            rep = detect_screen_then_gate(rr.screen.threshold, float(rr.q), rr.screen.rows, rr.test_rows)
            if rep is None:
                f.add("R4", "PASS", "screen rows disjoint from gate rows", f"screen |z|>{rr.screen.threshold:g} on separate rows", None)
            else:
                fix = {"repaired_level": rep.repaired_level, "rule": "gate at q*s, equivalently divide screened p-values by s"}
                if rr.pvalues is not None:
                    fix["repaired_pvalues"] = repair_pvalues(np.asarray(rr.pvalues, float), rep.null_pass_rate).tolist()
                f.add("R4", "VIOLATION", f"realized level min(1, q/s) = {rep.realized_level:.4f}; inflation {rep.inflation:.2f}x", f"screen |z|>{rep.screen_threshold:g} then gate at q={rep.nominal_level:g} on shared rows (s={rep.null_pass_rate:.4f})", fix)

    # R5: certificate sets do not union at a common level.
    for u in rr.union_claims:
        additive = float(sum(u.cert_levels))
        if u.claimed_level < additive - 1e-12:
            f.add("R5", "VIOLATION", f"valid statement is the additive level {additive:.4f}", f"{u.label or 'union'}: {len(u.cert_levels)} certificate sets claimed jointly at {u.claimed_level:g}", {"additive_level": additive})
        else:
            f.add("R5", "PASS", f"claimed level {u.claimed_level:g} >= additive {additive:.4f}", u.label or "union claim", None)

    # R6: refinement predicate.
    for rf in rr.refinements:
        ok = refines(rf.child, rf.parent, ledger=ledger)
        if ok:
            f.add("R6", "PASS", "refinement licensed at the charged multiplicity", rf.label or f"{rf.child.claim[:30]} refines {rf.parent.claim[:30]}", None)
        else:
            f.add("R6", "VIOLATION", "refinement predicate fails (kind, support containment, or multiplicity)", rf.label or f"{rf.child.claim[:30]} vs {rf.parent.claim[:30]}", None)

    # R7: test rows seen through a counted transcript.
    _priced_peek(rr, f)

    return f


def _priced_peek(rr: RunRecord, f: Findings) -> None:
    """R7. Declared peeks and the peeks carried by consumed certificates,
    one transcript per holdout at its largest bits, distinct holdouts
    adding. Touching the test rows, they inflate a test at level q to at
    most min(1, q 2^L); the run passes when its bar already priced L
    (priced_bits >= L), and the repair is the level q 2^-L."""
    peeks = [(p.holdout or f"declared peek {i + 1}", float(p.bits), p.rows, p.label) for i, p in enumerate(rr.peeks)]
    for c in rr.consumed:
        for p in peeks_of(c):
            peeks.append((p["holdout"], p["bits"], p["rows"], f"carried by {c.id}"))
    if not peeks:
        return
    if is_unknown(rr.test_rows) or any(is_unknown(rows) for _, _, rows, _ in peeks):
        f.add("R7", "UNKNOWN", "lineage predates instrumentation", "test or transcript lineage un-instrumented", None)
        return
    touching, named = {}, []
    for hid, bits, rows, lab in peeks:
        ov = identity_overlap(rows, rr.test_rows)
        if ov:
            touching[hid] = max(touching.get(hid, 0.0), bits)
            named.append(f"{hid} ({bits:.2f} bits{', ' + lab if lab else ''}): {overlap_report(ov)}")
    if not touching:
        f.add("R7", "PASS", "transcript rows disjoint from the test rows", f"{len(peeks)} transcript(s) on other rows", None)
        return
    L = float(sum(touching.values()))
    priced = float(rr.priced_bits)
    if L <= priced + 1e-9:
        f.add("R7", "PASS", f"priced: the bar already carries {priced:.2f} bits", f"test rows seen through {L:.2f} bits: " + "; ".join(named), None)
        return
    from .transcript import inflation as _inflation
    infl = _inflation(L - priced)
    rep = {"bits": L, "priced_bits": priced, "inflation": infl, "rule": "test at q 2^-L (the transcript union bound)"}
    if rr.q is not None:
        q = float(rr.q)
        rep["repaired_level"] = q * 2.0 ** (-L)
        price = f"realized level at most min(1, q 2^L) = {min(1.0, q * infl):.4f}; inflation {infl:.4g}x"
    else:
        price = f"inflation {infl:.4g}x; repair: test at q 2^-L"
    f.add("R7", "VIOLATION", price, f"test rows seen through {L:.2f} bits, {priced:.2f} priced: " + "; ".join(named), rep)
