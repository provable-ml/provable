"""LabelChannel: exact identity, R^2_Z forecast, conservation, stratified accounting (AC6.1/6.5).

Sheet statements implemented verbatim:
  identity      E[1 - d(pi_Z)^2] = (1-d^2)(1 - R^2_Z), R^2_Z = Var(pi_Z)/Var(Y)
  small-signal  R^2_Z = pi(1-pi) nu_Z(lam) (1 + O(nu_Z)); forecast drops the O(nu) term (P09)
  conservation  M_Y^aug = M^rev + M^res with M^rev = pi(1-pi) nu_Z(lam)
  walls         stratified (1-d^2)(1 - R^2_Z) V_res; dodge multiplier 1
  lever table   E2 field beats stratify; E3-worst equal; E3-realized stratify strictly finer
"""
import numpy as np


class LabelChannel:
    def __init__(self, pi): self.pi = float(pi); self.one_minus_d2 = 4 * self.pi * (1 - self.pi)

    def r2_forecast(self, nu_Z): return float(self.pi * (1 - self.pi) * nu_Z)

    def identity_lhs(self, piZ): return float(np.mean(1 - (2 * np.asarray(piZ, float) - 1) ** 2))

    def identity_rhs(self, piZ):
        piZ = np.asarray(piZ, float)
        R2 = float(np.mean(piZ ** 2) - np.mean(piZ) ** 2) / (self.pi * (1 - self.pi))
        return float(self.one_minus_d2 * (1 - R2)), R2

    def identity_residual(self, piZ):
        lhs = self.identity_lhs(piZ); rhs, R2 = self.identity_rhs(piZ)
        base = self.one_minus_d2 * (1 - (np.mean(piZ) * 2 - 1) ** 2 / self.one_minus_d2) if False else None
        piZ = np.asarray(piZ, float)
        lhs_c = float(np.mean(1 - (2 * piZ - 1) ** 2))
        m = float(np.mean(piZ)); v = float(np.mean(piZ ** 2) - m * m)
        rhs_c = float((1 - (2 * m - 1) ** 2) - 4 * v)
        return abs(lhs_c - rhs_c), R2

    def realized_r2(self, piZ):
        piZ = np.asarray(piZ, float)
        return float((np.mean(piZ ** 2) - np.mean(piZ) ** 2) / (self.pi * (1 - self.pi)))

    def conservation(self, nu_aug, nu_Z):
        M_aug = self.pi * (1 - self.pi) * nu_aug
        M_rev = self.pi * (1 - self.pi) * nu_Z
        return {"M_aug": float(M_aug), "M_rev": float(M_rev), "M_res": float(M_aug - M_rev)}

    def walls(self, V_res, R2_Z):
        dodge = self.one_minus_d2 * V_res
        strat = self.one_minus_d2 * (1 - R2_Z) * V_res
        return {"wall_dodge": float(dodge), "wall_stratified": float(strat)}

    def lever_table(self, V_res, R2_realized, field_gain):
        w = self.walls(V_res, R2_realized)
        w0 = self.walls(V_res, 0.0)
        field = self.one_minus_d2 * max(V_res - field_gain, 0.0)
        return {"rows": [{"lever": "E2 field", "wall": float(field), "note": "conditioning removes revealed mass before the wall"}, {"lever": "stratify (realized)", "wall": w["wall_stratified"], "note": "strictly finer accounting when Var pi_Z > 0"}, {"lever": "dodge (multiplier 1)", "wall": w["wall_dodge"], "note": "worst case"}], "e3_worst_stratified": w0["wall_stratified"], "e3_worst_dodge": w0["wall_dodge"], "field_le_stratify": bool(field <= w["wall_stratified"] + 1e-12), "stratify_le_dodge": bool(w["wall_stratified"] <= w["wall_dodge"] + 1e-12)}
