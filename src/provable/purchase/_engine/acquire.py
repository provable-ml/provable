"""Acquisition: recovered-visibility identities, unlock, stack, reliability law, deadband, r_min.

All statements are the formula sheet verbatim under the P01 scale pin and P09 (o(1) -> 0):
  increment  Delta nu(S|z) = (Dzeta - c_S' G_SS^-1 Deps_S)^2 / (1 - c_S' G_SS^-1 c_S)
  unlock     certify iff lam*sqrt((1-rho) M_F) >= hbar + zdelta_pop
  guaranteed lamA^2 = (1-d^2)(hbar + zdelta)^2 / (4 (1-rho) M0)
  stack      lam^2 >= [(hbar + zdelta)^2 - v_W]_+ / ((1-rho) M_F)
  reliability lam*^2 = (1-d^2) hbar^2 (1 - M0/M*)_+ / (4 M0)
  attenuation nu_z(lam) = lam^2 nu_z(1); a(lam,R^2) = lam^2 (1-R^2)/(1-lam^2 R^2); cap 1/(1-lam^2)
  deadband   r <= s - kappa gives zero wall movement; r_min = smallest r with (1-d^2)Cap(s-r,kappa) < M0
Forecast per P10 is always a band, never a point.
"""
import numpy as np
from scipy.stats import norm


class Acquisition:
    def __init__(self, state, envelope): self.st, self.env = state, envelope

    def delta_nu(self, Deps_S, G_SS, c_S, Dzeta):
        G_SS = np.atleast_2d(G_SS); Deps_S = np.atleast_1d(Deps_S); c_S = np.atleast_1d(c_S)
        sol_D = np.linalg.solve(G_SS + 1e-8 * np.eye(len(Deps_S)), Deps_S)
        sol_c = np.linalg.solve(G_SS + 1e-8 * np.eye(len(c_S)), c_S)
        den = max(1.0 - float(c_S @ sol_c), 1e-12)
        num = (float(Dzeta) - float(c_S @ sol_D)) ** 2
        return num / den

    def conservation_check(self, nu_S, nu_aug, dnu): return abs(nu_aug - nu_S - dnu)

    def zdelta_pop(self, delta): return float(norm.ppf(1 - delta)) / self.st.s_n

    def unlock(self, lam, M_F, rho, delta):
        lhs = lam * np.sqrt(max((1 - rho) * M_F, 0.0))
        rhs = self.st.hbar + self.zdelta_pop(delta)
        return {"certify": bool(lhs >= rhs), "lhs": float(lhs), "rhs": float(rhs)}

    def lambda_A(self, M0, rho, delta):
        num = self.st.one_minus_d2 * (self.st.hbar + self.zdelta_pop(delta)) ** 2
        lam2 = num / (4 * (1 - rho) * M0)
        return {"lam2": float(lam2), "lam": float(np.sqrt(lam2)) if lam2 <= 1 else None, "feasible": bool(lam2 <= 1.0)}

    def stack_lambda2(self, v_W, M_F, rho, delta):
        num = max((self.st.hbar + self.zdelta_pop(delta)) ** 2 - v_W, 0.0)
        return float(num / max((1 - rho) * M_F, 1e-300))

    def reliability_law(self, M0, M_star):
        lam2 = self.st.one_minus_d2 * self.st.hbar ** 2 * max(1 - M0 / max(M_star, 1e-300), 0.0) / (4 * M0)
        return float(lam2)

    def attenuation(self, lam, R2=0.0):
        a = lam * lam * (1 - R2) / max(1 - lam * lam * R2, 1e-300)
        return {"nu_scale": lam * lam, "a": float(a), "suppressor_cap": float(1.0 / max(1 - lam * lam, 1e-300))}

    def n_req_rows(self, nu_deb, rho, delta, zq=None):
        zq = self.st.zq if zq is None else zq
        zd = float(norm.ppf(1 - delta))
        return int(np.ceil((zq + zd) ** 2 / ((1 - rho) * max(nu_deb, 1e-9) * self.st.pihat * (1 - self.st.pihat))))

    def deadband_check(self, s, kappa, r): return bool(r <= s - kappa)

    def wall_sequence(self, support_eigs, s, kappa, B, r_max=None):
        eigs = np.sort(np.asarray(support_eigs, float))[::-1][:s]
        r_max = s if r_max is None else r_max
        return [self.env.wall(eigs[r:], kappa, B) for r in range(r_max + 1)]

    def r_min(self, support_eigs, s, kappa, B, M0):
        walls = self.wall_sequence(support_eigs, s, kappa, B)
        for r, w in enumerate(walls):
            if w < M0: return {"r_min": r, "walls": walls}
        return {"r_min": None, "walls": walls}

    def forecast(self, lam, cos2, M0=None, rho=None, delta=None, MhatF=None):
        st = self.st
        MhatF = st.MhatF if MhatF is None else MhatF
        margin = st.s_n * lam * np.sqrt(max(MhatF, 0.0) * cos2) - st.zq
        b_nu = lam * np.sqrt(max(cos2, 0.0)) * st.s_n * st.band_nu / (2 * np.sqrt(max(MhatF, 1e-9)))
        b_pi = abs((1 - 2 * st.pihat) / (2 * st.pihat * (1 - st.pihat))) * 1.96 * np.sqrt(st.pihat * (1 - st.pihat) / st.n) * (abs(margin) + st.zq)
        B = max(float(b_nu + b_pi), 0.05)
        p_mid = float(norm.cdf(margin) + norm.cdf(-margin - 2 * st.zq))
        lo = float(norm.cdf(margin - B) + norm.cdf(-(margin - B) - 2 * st.zq))
        hi = float(norm.cdf(margin + B) + norm.cdf(-(margin + B) - 2 * st.zq))
        return {"margin_z": float(margin), "B_z": B, "p_mid": p_mid, "p_lo": min(lo, hi), "p_hi": max(lo, hi)}

    def certify_column(self, z, Y):
        z = np.asarray(z, float); zs = (z - z.mean()) / max(z.std(ddof=1), 1e-12)
        dz = zs[Y == 1].mean() - zs[Y == 0].mean()
        zcol = self.st.s_n * dz
        return {"z_col": float(zcol), "certify": bool(abs(zcol) >= self.st.zq), "Dzeta_hat": float(dz), "nu_z_hat": float(dz * dz)}
