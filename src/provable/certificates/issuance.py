"""Issuance: the producer contract for instrumented runs.

An instrumented run exports: the claims as the claim; guarantee =
(FDR, q, family = core member ids with the full p-vector); test lineage
= fingerprinted certifying rows with clock keys; proposer lineage =
proposal rows (plus refit rows under train scope), hereditary through
consumed certificates; artifact = the run report; V = certified atom
names; a Gram descriptor over the certified support on the certifying
rows; a timestamp. One carrier certificate per certified run (the
guarantee is family-level; per-claim certificates would double-charge
the budget). Empty runs issue nothing in v0.1.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .lineage import Lineage, Slice, source_identity
from .records import Certificate, CertificateKind, Guarantee, GuaranteeKind, issue


def _slice_for(cd, rows, dataset_tag: str, clock=None) -> Slice:
    sid, aliases = source_identity(cd.X_)
    if clock is None:
        t = datetime(1970, 1, 1, tzinfo=timezone.utc)
        t0 = t1 = t
    else:
        s0 = pd.to_datetime(pd.Series(clock[0]).iloc[np.asarray(rows, int)])
        s1 = pd.to_datetime(pd.Series(clock[1]).iloc[np.asarray(rows, int)])
        t0, t1 = s0.min().to_pydatetime(), s1.max().to_pydatetime()
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
        if t1.tzinfo is None:
            t1 = t1.replace(tzinfo=timezone.utc)
    return Slice(source_id=sid, dataset_tag=str(dataset_tag), rows=frozenset(int(r) for r in rows), t0=t0, t1=t1, aliases=aliases)


def gram_descriptor(cd) -> dict:
    """Descriptor of the certified-support Gram on the certifying rows:
    sha256 of the correlation block rounded to 6 decimals."""
    cols = list(cd.certified_)
    if not cols:
        return {"n_support": 0, "descriptor": None}
    Z = cd.Z[np.ix_(np.asarray(cd.idx_s2, int), np.asarray(cols, int))].astype(np.float64)
    Zc = Z - Z.mean(axis=0)
    sd = Zc.std(axis=0)
    sd[sd == 0] = 1.0
    G = np.round((Zc / sd).T @ (Zc / sd) / len(Zc), 6)
    return {"n_support": len(cols), "descriptor": hashlib.sha256(G.tobytes()).hexdigest()[:16]}


def certificate_from_run(cd, dataset_tag: str = "panel", clock=None, consumed: Iterable[Certificate] = (), timestamp: Optional[datetime] = None) -> Optional[Certificate]:
    if cd.khat_ == 0 or not cd.certified_:
        return None
    q = float(cd.fdr_q)
    pv = [float(x) for x in np.asarray(cd.core_res_.pvalues)[1:]]
    eff = np.asarray(cd.core_res_.params)[1:]
    cert_pos = [int(p) for p in np.where(cd.cert_sel_)[0]]
    family = {"core_ids": [int(j) for j in cd.core_], "core_atoms": [cd.atom_names_[j] for j in cd.core_], "p": pv, "m": len(cd.core_), "khat": int(cd.khat_), "q": q,
              "support_ordered": [cd.atom_names_[cd.core_[p]] for p in cert_pos],
              "effects": [float(eff[p]) for p in cert_pos],
              "directions": [int(np.sign(eff[p])) or 1 for p in cert_pos],
              "atoms_spec": [cd.atom_spec_[cd.core_[p]] for p in cert_pos]}
    g = Guarantee(kind=GuaranteeKind.FDR, level=q, family=family)
    ell_test = Lineage.of(_slice_for(cd, cd.idx_s2, dataset_tag, clock))
    prop_rows = np.asarray(cd.idx_s1, int)
    if getattr(cd, "refit_scope", "train") == "train":
        prop_rows = np.asarray(cd.idx_train, int)
    ell_prop_raw = Lineage.of(_slice_for(cd, prop_rows, dataset_tag, clock))
    support = [cd.atom_names_[j] for j in cd.certified_]
    claim = f"certified carrier: {len(cd.certified_)} of {len(cd.core_)} core members at FDR q={q:g} (khat={cd.khat_})"
    return issue(claim=claim, guarantee=g, ell_test=ell_test, proposer_rows=ell_prop_raw, artifact={"report": cd.report_}, support=support, gram_context=gram_descriptor(cd), kind=CertificateKind.CARRIER, consumed=consumed, timestamp=timestamp)
