"""AdmissibilityGate: R1 lineage enforcement via metadata contract, refusals, attestations.

R1: Z admissible iff Z_j = zeta_j(W, E_j) with E exogenous, E independent of (Y, xi) given W,
and Z independent of Y given W (temporal + lineage; both clauses enforced). Unverifiable clauses
run on the attestation model (AC8a-P4). Refusal taxonomy (AC8a.1): REFUSE_LINEAGE (Y-descendant),
REFUSE_UNATTESTED (unknown lineage without attestation), NOGO_TWIN (attested world-twin,
AC5.4(i)), POST-HOC flag for candidates surfaced after the spec was seen.
"""


class AdmissibilityGate:
    def evaluate(self, meta):
        lineage = meta.get("lineage", "unknown")
        attested = bool(meta.get("attested", False))
        verdict, flags = "ADMISSIBLE", []
        if lineage == "y_descendant": verdict = "REFUSE_LINEAGE"
        elif lineage == "unknown" and not attested: verdict = "REFUSE_UNATTESTED"
        if bool(meta.get("twin_attested", False)): verdict = "NOGO_TWIN"
        if bool(meta.get("surfaced_after_spec", False)): flags.append("POST-HOC")
        if bool(meta.get("temporal_violation", False)): verdict = "REFUSE_LINEAGE"
        return {"verdict": verdict, "flags": flags, "lineage": lineage, "attested": attested}
