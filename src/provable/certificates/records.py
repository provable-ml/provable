"""Certificate records.

The certificate tuple (Definition A.3 of the floors paper; Definition 2.8
of the certificate-algebra paper for lineage):
c = (claim, guarantee, ell_test, ell_prop, artifact, resolved support V,
Gram context G, timestamp), plus a kind and the consumed certificates.

The guarantee carries a kind (FDR, FWER, FCR, EXACT), a numeric level,
and the issuing family, which the artifact must reconstruct. EXACT marks
arithmetic facts of an artifact (no error budget; level 0).

Kinds: carrier (names a coordinate set carrying mass), existence
(detection without a carrier), artifact (an exact evidence bundle of the
engine's audit and enforce verbs, unified here; no lineage; identity is
the bundle's PV id), performance (a risk bound or a paired-gain bound for
one model, issued by a mediated holdout; FWER over the transcript family).
A conditional kind arrives later; loaders preserve unknown kind strings.

Peeks: rows a certificate's proposal saw only through a counted
transcript (a holdout's log of L bits) are recorded apart from its direct
lineage, in the additive field ell_peek, one entry per holdout with its
bits, and inherited through consumption at the largest bits per holdout.
A certificate without peeks serializes exactly as before.

Serialization is canonical JSON (sorted keys, fixed float repr). The
content identity of a certificate is the sha256 of its canonical bytes;
the hashes of consumed certificates sit inside the hashed content, so
hereditary lineage is a tamper-evident chain. Row-id sets are stored
in-record, compressed. The schema evolves additively only: unknown
fields are preserved on load and re-emitted on save.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Tuple

from .lineage import Lineage, Slice, hereditary_union

SCHEMA = "provable/certificate/v1"
UNKNOWN_SOURCE = "__uninstrumented__"


def eval_atom_spec(spec, X):
    """Evaluate one stored atom spec on a frame: the raw 0/1 indicator
    with the dictionary's own semantics (ge/le thresholds, eq one-hots,
    path conjunctions, conj pairs). Grade and any consumer of a
    certificate's family evaluate atoms through this one function."""
    import numpy as _np
    import pandas as _pd
    X = _pd.DataFrame(X)
    k = spec.get("kind")
    if k == "ge":
        return (_pd.to_numeric(X[spec["col"]], errors="coerce").to_numpy(dtype=float) >= float(spec["thr"])).astype(float)
    if k == "le":
        return (_pd.to_numeric(X[spec["col"]], errors="coerce").to_numpy(dtype=float) <= float(spec["thr"])).astype(float)
    if k == "eq":
        return (X[spec["col"]].astype(str).to_numpy() == str(spec["val"])).astype(float)
    if k == "path":
        out = _np.ones(len(X), dtype=float)
        for c0, t0, o0 in spec["conds"]:
            v = _pd.to_numeric(X[c0], errors="coerce").to_numpy(dtype=float)
            out *= (v <= float(t0)).astype(float) if o0 == "le" else (v >= float(t0)).astype(float)
        return out
    if k == "conj":
        a, b = spec["parts"]
        return eval_atom_spec(a, X) * eval_atom_spec(b, X)
    raise ValueError(f"unknown atom spec kind {k!r}")


class GuaranteeKind(str, Enum):
    FDR = "FDR"
    FWER = "FWER"
    FCR = "FCR"
    EXACT = "EXACT"


class CertificateKind(str, Enum):
    CARRIER = "carrier"
    EXISTENCE = "existence"
    ARTIFACT = "artifact"
    PERFORMANCE = "performance"


PEEK_FIELD = "ell_peek"


@dataclass(frozen=True)
class Guarantee:
    kind: GuaranteeKind
    level: float
    family: Mapping[str, Any]

    def __post_init__(self):
        lv = float(self.level)
        if self.kind == GuaranteeKind.EXACT:
            if lv != 0.0:
                raise ValueError("EXACT guarantees carry level 0 (no error budget)")
        elif not (0.0 < lv <= 1.0):
            raise ValueError("guarantee level must lie in (0, 1]")


def unknown_lineage() -> Lineage:
    """Sentinel lineage for un-instrumented history (full lineage from the
    first instrumented run forward; UNKNOWN before)."""
    t = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return Lineage.of(Slice(source_id=UNKNOWN_SOURCE, dataset_tag=UNKNOWN_SOURCE, rows=None, t0=t, t1=t))


def is_unknown(lin: Lineage) -> bool:
    return any(s.source_id == UNKNOWN_SOURCE for s in lin.slices)


@dataclass(frozen=True)
class Certificate:
    claim: str
    guarantee: Guarantee
    ell_test: Lineage
    ell_prop: Lineage
    artifact: Mapping[str, Any]
    support: frozenset
    gram_context: Mapping[str, Any]
    timestamp: datetime
    kind: CertificateKind
    consumed: Tuple["Certificate", ...] = ()
    consumption_mode: Optional[str] = None
    license_meta: Optional[Mapping[str, Any]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def full_lineage(self) -> Lineage:
        return self.ell_test.union(self.ell_prop)

    @property
    def id(self) -> str:
        """Content identity. Artifact-kind certificates keep the engine
        bundle's PV id; all others hash their canonical bytes."""
        if self.kind == CertificateKind.ARTIFACT:
            pv = self.artifact.get("certificate_id")
            if pv:
                return str(pv)
        return "CD-" + hashlib.sha256(canonical_bytes(self)).hexdigest()[:16].upper()

    def uninstrumented(self) -> bool:
        return is_unknown(self.ell_test) or is_unknown(self.ell_prop)


def issue(claim: str, guarantee: Guarantee, ell_test: Lineage, proposer_rows: Lineage, artifact: Mapping[str, Any], support: Iterable[str], gram_context: Mapping[str, Any], kind: CertificateKind, consumed: Iterable[Certificate] = (), consumption_mode: Optional[str] = None, license_meta: Optional[Mapping[str, Any]] = None, timestamp: Optional[datetime] = None, peeks: Iterable[Mapping[str, Any]] = ()) -> Certificate:
    """Issue with hereditary lineage: ell_prop = proposer rows united with
    the full lineage of every consumed certificate. peeks are the rows the
    proposal saw only through a counted transcript, each a mapping with
    holdout, bits and rows (a Lineage); they are united with the consumed
    certificates' peeks at the largest bits per holdout and stored in the
    additive field ell_peek, absent when there are none."""
    consumed = tuple(consumed)
    ell_prop = hereditary_union(proposer_rows, [c.full_lineage() for c in consumed])
    ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
    merged = merge_peeks(list(peeks) + [p for c in consumed for p in peeks_of(c)])
    extra = {PEEK_FIELD: [_peek_to_dict(p) for p in merged]} if merged else {}
    return Certificate(claim=str(claim), guarantee=guarantee, ell_test=ell_test, ell_prop=ell_prop, artifact=dict(artifact), support=frozenset(str(s) for s in support), gram_context=dict(gram_context), timestamp=ts, kind=kind, consumed=consumed, consumption_mode=consumption_mode, license_meta=None if license_meta is None else dict(license_meta), extra=extra)


def peeks_of(c: Certificate) -> list:
    """The certificate's transcript-seen rows: a list of {holdout, bits,
    rows (a Lineage)}, empty for a certificate without peeks."""
    raw = (c.extra or {}).get(PEEK_FIELD) or []
    return [{"holdout": str(p["holdout"]), "bits": float(p["bits"]), "rows": _lineage_from_list(p.get("rows"))} for p in raw]


def merge_peeks(peeks: Iterable[Mapping[str, Any]]) -> list:
    """One entry per holdout: the largest bits (one transcript per
    holdout, priced at its longest prefix) over the union of rows."""
    out = {}
    for p in peeks:
        hid = str(p["holdout"])
        rows = p["rows"] if isinstance(p["rows"], Lineage) else _lineage_from_list(p["rows"])
        if hid in out:
            out[hid] = {"holdout": hid, "bits": max(out[hid]["bits"], float(p["bits"])), "rows": out[hid]["rows"].union(rows)}
        else:
            out[hid] = {"holdout": hid, "bits": float(p["bits"]), "rows": rows}
    return [out[k] for k in sorted(out)]


def _peek_to_dict(p: Mapping[str, Any]) -> dict:
    return {"holdout": str(p["holdout"]), "bits": float(p["bits"]), "rows": _lineage_to_list(p["rows"])}


def from_evidence(bundle: Mapping[str, Any]) -> Certificate:
    """The unification: an engine evidence bundle (audit or enforce)
    becomes a Certificate of kind artifact with an EXACT guarantee, no
    lineage, and the bundle's PV id retained as its identity. The bundle
    itself is the artifact payload, byte-preserved; verify_id continues
    to hold on it."""
    verb = str(bundle.get("verb", "audit"))
    verdict = bundle.get("verdict", {}) or {}
    claim = str(verdict.get("statement", f"{verb} evidence bundle"))
    pol = bundle.get("policy", {}) or {}
    fam = {"policy": pol.get("name"), "policy_sha256": pol.get("sha256"), "verb": verb}
    sup = set()
    for f in bundle.get("findings", []) or []:
        cl = str(f.get("clause", ""))
        inner = cl[cl.find("(") + 1:cl.rfind(")")] if "(" in cl else ""
        for tok in inner.split(","):
            tok = tok.strip().split(" ")[0].strip()
            if tok and not tok.startswith("order"):
                sup.add(tok)
    ts_raw = bundle.get("created_utc")
    ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    g = Guarantee(kind=GuaranteeKind.EXACT, level=0.0, family=fam)
    return Certificate(claim=claim, guarantee=g, ell_test=Lineage(), ell_prop=Lineage(), artifact=dict(bundle), support=frozenset(sup), gram_context={}, timestamp=ts, kind=CertificateKind.ARTIFACT)


def refines(c_prime: Certificate, c: Certificate, ledger=None) -> bool:
    """Stored refinement predicate (not enforced in v0.1): c is existence
    kind, support of c_prime inside V of c, family charged at multiplicity
    |V| or at the ledger's certified-support-union multiplicity."""
    if c.kind != CertificateKind.EXISTENCE:
        return False
    if not c_prime.support <= c.support:
        return False
    fam = c_prime.guarantee.family
    mult = fam.get("multiplicity", fam.get("m"))
    if mult is None:
        return False
    allowed = {len(c.support)}
    if ledger is not None:
        allowed.add(len(ledger.certified_support_union()))
    return int(mult) in allowed


# ------------------------------------------------------------ serialization

def _rows_pack(rows) -> Optional[str]:
    if rows is None:
        return None
    arr = ",".join(str(i) for i in sorted(rows))
    return base64.b64encode(zlib.compress(arr.encode())).decode()


def _rows_unpack(blob) -> Optional[frozenset]:
    if blob is None:
        return None
    arr = zlib.decompress(base64.b64decode(blob)).decode()
    return frozenset(int(x) for x in arr.split(",")) if arr else frozenset()


def _slice_to_dict(s: Slice) -> dict:
    return {"source_id": s.source_id, "dataset_tag": s.dataset_tag, "rows": _rows_pack(s.rows), "t0": s.t0.isoformat(), "t1": s.t1.isoformat()}


def _slice_from_dict(d: Mapping) -> Slice:
    return Slice(source_id=d["source_id"], dataset_tag=d["dataset_tag"], rows=_rows_unpack(d.get("rows")), t0=datetime.fromisoformat(d["t0"]), t1=datetime.fromisoformat(d["t1"]))


def _lineage_to_list(l: Lineage) -> list:
    return sorted((_slice_to_dict(s) for s in l.slices), key=lambda d: (d["source_id"], d["dataset_tag"], d["t0"], d["t1"], d["rows"] or ""))


def _lineage_from_list(v) -> Lineage:
    return Lineage(frozenset(_slice_from_dict(d) for d in (v or [])))


def _num(x):
    import numpy as _np
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.ndarray,)):
        return x.tolist()
    if isinstance(x, frozenset):
        return sorted(x)
    raise TypeError(f"not serializable: {type(x)}")


def to_dict(c: Certificate) -> dict:
    d = {
        "schema": SCHEMA,
        "kind": c.kind.value if isinstance(c.kind, CertificateKind) else str(c.kind),
        "claim": c.claim,
        "guarantee": {"kind": c.guarantee.kind.value, "level": float(c.guarantee.level), "family": dict(c.guarantee.family)},
        "ell_test": _lineage_to_list(c.ell_test),
        "ell_prop": _lineage_to_list(c.ell_prop),
        "artifact": dict(c.artifact),
        "support": sorted(c.support),
        "gram_context": dict(c.gram_context),
        "timestamp": c.timestamp.isoformat(),
        "consumed_ids": [x.id for x in c.consumed],
        "consumption_mode": c.consumption_mode,
        "license_meta": None if c.license_meta is None else dict(c.license_meta),
    }
    for k, v in (c.extra or {}).items():
        if k not in d:
            d[k] = v
    return d


def canonical_bytes(c) -> bytes:
    d = to_dict(c) if isinstance(c, Certificate) else dict(c)
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=_num).encode()


def content_id(d: Mapping[str, Any]) -> str:
    if d.get("kind") == CertificateKind.ARTIFACT.value:
        pv = (d.get("artifact") or {}).get("certificate_id")
        if pv:
            return str(pv)
    body = json.dumps(dict(d), sort_keys=True, separators=(",", ":"), default=_num).encode()
    return "CD-" + hashlib.sha256(body).hexdigest()[:16].upper()


_KNOWN = {"schema", "kind", "claim", "guarantee", "ell_test", "ell_prop", "artifact", "support", "gram_context", "timestamp", "consumed_ids", "consumption_mode", "license_meta"}


def from_dict(d: Mapping[str, Any], resolve=None) -> Certificate:
    """Load a certificate record. Unknown top-level fields are preserved
    in .extra and re-emitted by to_dict (additive-only schema). resolve
    maps a consumed id to its Certificate when chain reconstruction is
    wanted; otherwise consumed stays empty and the ids remain in extra."""
    g = d["guarantee"]
    gu = Guarantee(kind=GuaranteeKind(g["kind"]), level=float(g["level"]), family=dict(g.get("family", {})))
    kind_raw = d.get("kind", "carrier")
    try:
        kind = CertificateKind(kind_raw)
    except ValueError:
        kind = kind_raw
    consumed = tuple(resolve(i) for i in d.get("consumed_ids", [])) if resolve else ()
    extra = {k: v for k, v in d.items() if k not in _KNOWN}
    if not resolve and d.get("consumed_ids"):
        extra.setdefault("consumed_ids", list(d["consumed_ids"]))
    ts = datetime.fromisoformat(d["timestamp"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return Certificate(claim=d["claim"], guarantee=gu, ell_test=_lineage_from_list(d.get("ell_test")), ell_prop=_lineage_from_list(d.get("ell_prop")), artifact=dict(d.get("artifact", {})), support=frozenset(d.get("support", [])), gram_context=dict(d.get("gram_context", {})), timestamp=ts, kind=kind, consumed=consumed, consumption_mode=d.get("consumption_mode"), license_meta=d.get("license_meta"), extra=extra)


def verify_chain(c: Certificate) -> bool:
    """Tamper evidence: every stored consumed id must equal the recomputed
    content id of the consumed certificate, recursively."""
    for x in c.consumed:
        if x.id != content_id(to_dict(x)):
            return False
        if not verify_chain(x):
            return False
    return c.id == content_id(to_dict(c))
