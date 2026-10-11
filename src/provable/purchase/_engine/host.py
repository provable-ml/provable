"""SyntheticHost: factor host w = A F + xi, planted contrast masks, candidate columns (R2 model).

Host law (AC0-P1..P4): F | Y=y ~ N(m_y, Sigma_F) with Sigma_F common across sections (P4),
w = A F + xi, so w | Y=y ~ N(A m_y, G) with G = A Sigma_F A^T + Sigma_xi. Binary Y prior pi,
d(pi) = |2 pi - 1|, 1 - d^2 = 4 pi (1-pi). Candidate columns follow R2 exactly:
Z = lambda <v, F> + sqrt(1-lambda^2) eps with v Sigma_F-unit and independent instrument noises.
"""
import numpy as np


class SyntheticHost:
    def __init__(self, P, eigs, s2, pi, Dm, seed, a_mode="random"):
        self.P, self.K, self.s2, self.pi = int(P), len(eigs), float(s2), float(pi)
        self.eigs = np.asarray(eigs, float)
        self.Sigma_F = np.diag(self.eigs)
        self.Dm = np.asarray(Dm, float)
        self.m0 = np.zeros(self.K); self.m1 = self.Dm.copy()
        self.rng0 = np.random.default_rng(seed)
        self.A = self._build_A(a_mode)
        self.G = self.A @ self.Sigma_F @ self.A.T + self.s2 * np.eye(self.P)
        self.Deps = self.A @ self.Dm

    def _build_A(self, a_mode):
        if a_mode == "flat_first":
            signs = self.rng0.choice([-1.0, 1.0], size=self.P); a1 = signs / np.sqrt(self.P)
            M = self.rng0.standard_normal((self.P, self.K)); M[:, 0] = a1
            Q, _ = np.linalg.qr(M); Q[:, 0] *= np.sign(Q[:, 0] @ a1)
            return Q
        Q, _ = np.linalg.qr(self.rng0.standard_normal((self.P, self.K)))
        return Q

    def M_F(self): return float(self.Dm @ np.linalg.solve(self.Sigma_F, self.Dm))

    def nu_full(self): return float(self.Dm @ np.linalg.solve(self.Sigma_F + self.s2 * np.eye(self.K), self.Dm))

    def oracle_v(self):
        v = np.linalg.solve(self.Sigma_F, self.Dm); return v / np.sqrt(max(v @ self.Sigma_F @ v, 1e-300))

    def sigma_unit(self, v):
        v = np.asarray(v, float); return v / np.sqrt(max(v @ self.Sigma_F @ v, 1e-300))

    def sample(self, n, rng):
        Y = (rng.random(n) < self.pi).astype(int)
        F = np.where(Y[:, None] == 1, self.m1[None, :], self.m0[None, :]) + rng.standard_normal((n, self.K)) * np.sqrt(self.eigs)[None, :]
        W = F @ self.A.T + np.sqrt(self.s2) * rng.standard_normal((n, self.P))
        return Y, W, F

    def candidate_column(self, F, v, lam, rng):
        v = self.sigma_unit(v)
        return lam * (F @ v) + np.sqrt(max(1.0 - lam * lam, 0.0)) * rng.standard_normal(F.shape[0])

    def panel_col_zmeans(self, n):
        return np.sqrt(n * self.pi * (1 - self.pi)) * self.Deps / np.sqrt(np.diag(self.G))

    def ceiling_auc(self): return float(_Phi(np.sqrt(self.M_F() / 2.0)))

    def ceiling_acc(self):
        m = np.sqrt(self.M_F()); c = np.log((1 - self.pi) / self.pi)
        err = self.pi * _Phi((c - m * m / 2) / m) + (1 - self.pi) * _Phi((-c - m * m / 2) / m)
        return float(1.0 - err)

    def pi_Z_true(self, z, lam, v):
        v = self.sigma_unit(v); g = lam * float(v @ self.Dm)
        f1 = _phi(z - g); f0 = _phi(z)
        return self.pi * f1 / np.maximum(self.pi * f1 + (1 - self.pi) * f0, 1e-300)


def _Phi(x):
    from scipy.stats import norm; return norm.cdf(x)


def _phi(x):
    from scipy.stats import norm; return norm.pdf(x)
