"""RunRecord: the declared input schema for lint.

A RunRecord states what an analysis did: which rows tested, which rows
proposed, what was consumed, what was screened, what was re-tested, what
was claimed jointly, what was refined, and which rows its proposal saw
only through a counted transcript (peeks, with their bits) together with
the bits its own bar already priced. Three assemblers: from_dict (a
foreign pipeline declares itself and gets priced), from_ledger (the
book's own state), from_report (an instrumented run of this package).

Historical, un-instrumented material carries the UNKNOWN lineage
sentinel; checks over it return UNKNOWN, never a silent pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Mapping, Optional

from .lineage import Lineage, Slice
from .records import Certificate, unknown_lineage


@dataclass
class Retest:
    rows: Lineage
    level: float
    on_certified: bool = False
    label: str = ""


@dataclass
class ScreenDecl:
    threshold: float
    rows: Lineage
    label: str = ""


@dataclass
class UnionClaim:
    cert_levels: List[float]
    claimed_level: float
    label: str = ""


@dataclass
class RefinementDecl:
    child: Certificate
    parent: Certificate
    label: str = ""


@dataclass
class PeekDecl:
    """Rows the proposal saw only through a counted transcript of the
    given bits (a holdout's log), named by the holdout when known."""
    rows: Lineage
    bits: float
    holdout: str = ""
    label: str = ""


@dataclass
class RunRecord:
    name: str = "run"
    instrumented: bool = True
    test_rows: Lineage = field(default_factory=Lineage)
    proposal_rows: Lineage = field(default_factory=Lineage)
    consumed: List[Certificate] = field(default_factory=list)
    q: Optional[float] = None
    m: Optional[int] = None
    khat: Optional[int] = None
    pvalues: Optional[list] = None
    certified_idx: Optional[list] = None
    screen: Optional[ScreenDecl] = None
    retests: List[Retest] = field(default_factory=list)
    union_claims: List[UnionClaim] = field(default_factory=list)
    refinements: List[RefinementDecl] = field(default_factory=list)
    peeks: List[PeekDecl] = field(default_factory=list)
    priced_bits: float = 0.0


def _lineage_from_decl(d: Any) -> Lineage:
    """Accepts "unknown", a Lineage, or {"source_id", "dataset_tag",
    "rows": [..] or None, "t0", "t1"} (times ISO, optional)."""
    if isinstance(d, Lineage):
        return d
    if d in (None, "unknown", "UNKNOWN"):
        return unknown_lineage()
    t0 = datetime.fromisoformat(d["t0"]) if d.get("t0") else datetime(1970, 1, 1, tzinfo=timezone.utc)
    t1 = datetime.fromisoformat(d["t1"]) if d.get("t1") else t0
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=timezone.utc)
    rows = d.get("rows")
    return Lineage.of(Slice(source_id=str(d["source_id"]), dataset_tag=str(d.get("dataset_tag", d["source_id"])), rows=None if rows is None else frozenset(int(r) for r in rows), t0=t0, t1=t1))


def from_dict(d: Mapping[str, Any]) -> RunRecord:
    rr = RunRecord(name=str(d.get("name", "declared run")), instrumented=bool(d.get("instrumented", False)))
    rr.test_rows = _lineage_from_decl(d.get("test_rows", "unknown"))
    rr.proposal_rows = _lineage_from_decl(d.get("proposal_rows", "unknown"))
    rr.q = d.get("q")
    rr.m = d.get("m")
    rr.khat = d.get("khat")
    rr.pvalues = list(d["pvalues"]) if d.get("pvalues") is not None else None
    rr.certified_idx = list(d["certified_idx"]) if d.get("certified_idx") is not None else None
    if d.get("screen"):
        s = d["screen"]
        rr.screen = ScreenDecl(threshold=float(s["threshold"]), rows=_lineage_from_decl(s.get("rows", "unknown")), label=str(s.get("label", "")))
    for r in d.get("retests", []):
        rr.retests.append(Retest(rows=_lineage_from_decl(r.get("rows", "unknown")), level=float(r["level"]), on_certified=bool(r.get("on_certified", False)), label=str(r.get("label", ""))))
    for u in d.get("union_claims", []):
        rr.union_claims.append(UnionClaim(cert_levels=[float(x) for x in u["cert_levels"]], claimed_level=float(u["claimed_level"]), label=str(u.get("label", ""))))
    for pk in d.get("peeks", []):
        rr.peeks.append(PeekDecl(rows=_lineage_from_decl(pk.get("rows", "unknown")), bits=float(pk["bits"]), holdout=str(pk.get("holdout", "")), label=str(pk.get("label", ""))))
    rr.priced_bits = float(d.get("priced_bits", 0.0))
    return rr


def from_ledger(ledger) -> RunRecord:
    rr = RunRecord(name=f"ledger {ledger.path}", instrumented=True)
    rr.consumed = ledger.active()
    rr.test_rows = Lineage()
    return rr


def from_report(report) -> RunRecord:
    return report.runrecord()
