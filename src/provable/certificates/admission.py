"""Reuse admission.

The enforcement contract (certificate-algebra paper Thm 3.1, Cor 3.2):
a proposed reuse is admitted iff every consumed certificate is fresh for
the consuming rows; otherwise the ledger refuses by default, or demotes
under an explicit flag, marking the guarantee MISCALIBRATED and recording
the overlap.

Verdicts (enum identifiers distinct from printed labels): FRESH,
NOT_FRESH ("NOT FRESH"), FRESH_DEP_FLAG ("FRESH WITH DEPENDENCE FLAG",
identity-disjoint but clock windows overlap across dataset tags;
reported, enforcement per the ruled time-separation default at lag 0
remains report-only in v0.1), PEEKED (identity-disjoint, but the
consuming rows were seen by the certificate's proposal through a counted
transcript of L bits; admitted at inflation 2^L with pass level
level 2^-L, the Kraft union bound over transcripts), UNKNOWN_LINEAGE
("UNKNOWN LINEAGE", the item predates instrumentation; never a silent
pass, never budget-bearing).

An Admission carries: verdict, inflation (the error-rate multiplier the
reuse pattern incurs; 1.0 when clean), level (the pass level for the next
test after the admission), overlap (rows named), demotion (None, or the
miscalibrated record), and bits (log2 of the inflation; 0 unless PEEKED).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional

from .lineage import Lineage, clock_overlaps, identity_overlap, overlap_report
from .records import Certificate, is_unknown, peeks_of


class Verdict(Enum):
    FRESH = "FRESH"
    NOT_FRESH = "NOT FRESH"
    FRESH_DEP_FLAG = "FRESH WITH DEPENDENCE FLAG"
    PEEKED = "PEEKED"
    UNKNOWN_LINEAGE = "UNKNOWN LINEAGE"

    @property
    def label(self) -> str:
        return self.value


MISCALIBRATED = "MISCALIBRATED"
MODES = ("guidance", "object")


class FreshnessError(RuntimeError):
    pass


@dataclass
class Demotion:
    claim: str
    original_kind: str
    original_level: float
    status: str
    overlap: list


@dataclass
class ItemVerdict:
    cert_id: str
    claim: str
    verdict: Verdict
    overlaps: list = field(default_factory=list)
    dependence_flags: list = field(default_factory=list)
    peeks: list = field(default_factory=list)

    def named_overlap(self) -> str:
        return overlap_report(self.overlaps)

    @property
    def bits(self) -> float:
        """Transcript bits on the consuming rows: the sum over holdouts of
        each holdout's bits (one transcript per holdout)."""
        return float(sum(p["bits"] for p in self.peeks))


@dataclass
class Admission:
    """The reuse admission record. inflation is the exact error-rate
    multiplier of the admitted pattern (1.0 for a clean fresh reuse);
    level is the pass level for the next test; demotion is None unless
    the demote path was taken."""
    verdict: Verdict
    inflation: float
    level: float
    overlap: list
    demotion: Optional[List[Demotion]]
    mode: str
    items: List[ItemVerdict]
    bits: float = 0.0

    @property
    def admitted(self) -> bool:
        return self.verdict in (Verdict.FRESH, Verdict.FRESH_DEP_FLAG, Verdict.PEEKED)

    def flagged(self) -> List[ItemVerdict]:
        return [v for v in self.items if v.verdict == Verdict.FRESH_DEP_FLAG]

    def peeked(self) -> List[ItemVerdict]:
        return [v for v in self.items if v.verdict == Verdict.PEEKED]


def verdict_for(cert: Certificate, S: Lineage) -> ItemVerdict:
    lin = cert.full_lineage()
    if is_unknown(lin) or is_unknown(S):
        return ItemVerdict(cert_id=cert.id, claim=cert.claim, verdict=Verdict.UNKNOWN_LINEAGE)
    ov = identity_overlap(lin, S)
    if ov:
        return ItemVerdict(cert_id=cert.id, claim=cert.claim, verdict=Verdict.NOT_FRESH, overlaps=ov)
    flags = clock_overlaps(lin, S)
    seen = []
    for p in peeks_of(cert):
        if is_unknown(p["rows"]):
            return ItemVerdict(cert_id=cert.id, claim=cert.claim, verdict=Verdict.UNKNOWN_LINEAGE)
        pov = identity_overlap(p["rows"], S)
        if pov:
            seen.append({"holdout": p["holdout"], "bits": p["bits"], "overlap": pov})
    if seen:
        return ItemVerdict(cert_id=cert.id, claim=cert.claim, verdict=Verdict.PEEKED, dependence_flags=flags, peeks=seen)
    if flags:
        return ItemVerdict(cert_id=cert.id, claim=cert.claim, verdict=Verdict.FRESH_DEP_FLAG, dependence_flags=flags)
    return ItemVerdict(cert_id=cert.id, claim=cert.claim, verdict=Verdict.FRESH)


def _combine(items: List[ItemVerdict]) -> Verdict:
    vs = {v.verdict for v in items} or {Verdict.FRESH}
    for v in (Verdict.NOT_FRESH, Verdict.UNKNOWN_LINEAGE, Verdict.PEEKED, Verdict.FRESH_DEP_FLAG):
        if v in vs:
            return v
    return Verdict.FRESH


def peeked_bits(items: List[ItemVerdict]) -> float:
    """Bits on the consuming rows across admitted items: one transcript
    per holdout, priced at its largest bits; distinct holdouts add."""
    per = {}
    for v in items:
        if v.verdict != Verdict.PEEKED:
            continue
        for p in v.peeks:
            per[p["holdout"]] = max(per.get(p["holdout"], 0.0), float(p["bits"]))
    return float(sum(per.values()))


def admit(certs: Iterable[Certificate], rows: Lineage, level: float, mode: str = "guidance", on_stale: str = "refuse") -> Admission:
    """Thm 3.1 / Cor 3.2. on_stale is "refuse" (raise FreshnessError,
    default) or "demote" (admit with the guarantees marked MISCALIBRATED
    and the overlap recorded). A clean fresh admission has inflation 1.0
    and pass level = level; a PEEKED admission (the consuming rows seen
    only through counted transcripts of L bits in total) has inflation
    2^L and pass level level 2^-L; UNKNOWN LINEAGE admissions are
    returned, never raised, and never budget-bearing (the ledger excludes
    them)."""
    if mode not in MODES:
        raise ValueError(f"consumption mode {mode!r} not implemented; license is stored metadata only")
    if on_stale not in ("refuse", "demote"):
        raise ValueError("on_stale must be 'refuse' or 'demote'")
    certs = list(certs)
    items = [verdict_for(c, rows) for c in certs]
    stale = [(c, v) for c, v in zip(certs, items) if v.verdict == Verdict.NOT_FRESH]
    if stale and on_stale == "refuse":
        named = "; ".join(f"[{v.claim}] {v.named_overlap()}" for _, v in stale)
        raise FreshnessError(f"refused: certificate lineage meets the consuming rows. {named}")
    demotions = None
    if stale:
        demotions = [Demotion(claim=c.claim, original_kind=str(c.guarantee.kind.value), original_level=float(c.guarantee.level), status=MISCALIBRATED, overlap=v.overlaps) for c, v in stale]
    overlap = [o for v in items for o in v.overlaps]
    bits = peeked_bits(items)
    from .transcript import inflation as _inflation
    return Admission(verdict=_combine(items), inflation=_inflation(bits), level=float(level) * 2.0 ** (-bits), overlap=overlap, demotion=demotions, mode=mode, items=items, bits=bits)


@dataclass
class ScreenGateReport:
    screen_threshold: float
    null_pass_rate: float
    nominal_level: float
    realized_level: float
    inflation: float
    repaired_level: float
    overlap: list


def detect_screen_then_gate(screen_threshold: float, gate_level: float, screen_rows: Lineage, gate_rows: Lineage) -> Optional[ScreenGateReport]:
    """Screen at |z| > c then gate at level q on overlapping rows runs at
    min(1, q/s), s = 2 Phibar(c); inflation 1/s (3.15 at a one-sigma
    screen); exact repair gates at q*s. None when identity-disjoint."""
    from .kit import realized_level, repaired_level, screen_pass_rate
    ov = identity_overlap(screen_rows, gate_rows)
    if not ov:
        return None
    s = screen_pass_rate(screen_threshold)
    return ScreenGateReport(screen_threshold=float(screen_threshold), null_pass_rate=s, nominal_level=float(gate_level), realized_level=realized_level(gate_level, s), inflation=1.0 / s, repaired_level=repaired_level(gate_level, s), overlap=ov)
