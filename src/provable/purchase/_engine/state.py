"""SieveState: fitted state of record for a synthetic panel (renamed from the folded module ledger.py; "ledger" is reserved for provable.certificates.Ledger).

Standardization of record (amendment A1): columns scaled by pooled within-class sd; Ghat is the
pooled within-class covariance of class-centered residuals (unit diagonal by construction).
Holds Ghat, Depshat, per-column t's, pihat, n, P, debiased norms with bands (P05), eta (P04),
the BH marginal bar (P02), and the rider refusal (P03). Consumes provable (P08): one
CertifiedDiscovery.run per panel of record supplies khat, certified atoms and the claims table.
Synthetic panels only.
"""
import numpy as np
from scipy.stats import norm


class SieveState:
    N_MIN_RATIO, JITTER, C_ETA, DELTA_EST = 4, 1e-8, 2.0, 0.05

    def __init__(self, q=0.10): self.q = float(q); self.refusal = None; self.provable_block = None

    def fit(self, Y, W, run_provable=False, provable_seed=0, provable_kwargs=None):
        Y = np.asarray(Y, int); W = np.asarray(W, float)
        self.n, self.P = W.shape
        if self.n < self.N_MIN_RATIO * self.P:
            self.refusal = f"REFUSE_RIDER: n={self.n} < {self.N_MIN_RATIO}*P={self.N_MIN_RATIO*self.P} (AC2-P2 hard refusal, pin P03)"
            return self
        self.pihat = float(np.clip(Y.mean(), 1e-6, 1 - 1e-6))
        self.d2hat = (2 * self.pihat - 1) ** 2; self.one_minus_d2 = 4 * self.pihat * (1 - self.pihat)
        self.s_n = float(np.sqrt(self.n * self.pihat * (1 - self.pihat)))
        mu = W.mean(axis=0)
        R = W - np.where(Y[:, None] == 1, W[Y == 1].mean(axis=0)[None, :], W[Y == 0].mean(axis=0)[None, :])
        sd = np.sqrt((R * R).sum(axis=0) / (self.n - 2)); sd = np.where(sd < 1e-12, 1.0, sd)
        self.Ws = (W - mu) / sd
        Rs = R / sd
        self.Ghat = (Rs.T @ Rs) / (self.n - 2)
        self.Depshat = self.Ws[Y == 1].mean(axis=0) - self.Ws[Y == 0].mean(axis=0)
        self.sigma_D2 = 1.0 / (self.n * self.pihat) + 1.0 / (self.n * (1 - self.pihat))
        self.t = self.Depshat / np.sqrt(self.sigma_D2)
        self.eta = self.C_ETA * (np.sqrt(self.P / self.n) + self.P / self.n + np.sqrt(np.log(1 / self.DELTA_EST) / self.n))
        self.normD2_deb = float(self.Depshat @ self.Depshat - self.P * self.sigma_D2)
        self.sd_norm = float(np.sqrt(4 * self.sigma_D2 * float(self.Depshat @ (self.Ghat @ self.Depshat)) + 2 * self.sigma_D2 ** 2 * float(np.sum(self.Ghat * self.Ghat))))
        Gj = self.Ghat + self.JITTER * np.eye(self.P)
        self.nu_raw = float(self.Depshat @ np.linalg.solve(Gj, self.Depshat))
        self.nu_deb = self.nu_raw - self.P * self.sigma_D2
        self.sd_nu = float(np.sqrt(4 * self.sigma_D2 * max(self.nu_deb, 0.0) + 2 * self.P * self.sigma_D2 ** 2) + self.eta * max(self.nu_deb, 0.0))
        self.band_nu = 1.96 * self.sd_nu; self.band_norm = 1.96 * self.sd_norm
        self.MhatF = max(self.nu_deb, 0.0)
        evals, evecs = np.linalg.eigh((self.Ghat + self.Ghat.T) / 2)
        order = np.argsort(evals)[::-1]
        self.evals, self.evecs = evals[order], evecs[:, order]
        self.Khat = max(1, int(np.sum(self.evals > 1.0 + 2 * np.sqrt(self.P / self.n))))
        self.khat = 0
        if run_provable: self._consume_provable(Y, W, provable_seed, provable_kwargs or {})
        self.zq = self.zstar(self.khat)
        self.hbar = self.zq / self.s_n
        return self

    def _consume_provable(self, Y, W, seed, kw):
        import pandas as pd
        from provable.sieve import CertifiedDiscovery
        base = dict(fdr_q=self.q, seed=int(seed), verbose=False, rounds=1, interactions=False, n_harvest=12, entry_cap=60, core_cap=30)
        base.update(kw)
        cd = CertifiedDiscovery(**base)
        X = pd.DataFrame(W, columns=[f"c{j}" for j in range(self.P)])
        out = cd.run(X, pd.Series(Y))
        self.khat = int(getattr(cd, "khat_", 0))
        claims = getattr(cd, "claims_", None)
        self.provable_block = {"version": _provable_version(), "khat": self.khat, "certified": list(getattr(cd, "certified_", [])), "n_claims": 0 if claims is None else int(len(claims)), "claims_head": [] if claims is None else claims.head(8).to_dict("records"), "elapsed_s": float(getattr(cd, "elapsed_", float("nan")))}

    def zstar(self, k): return float(norm.ppf(1 - self.q * max(int(k), 1) / (2 * self.P)))

    def zstar_peff(self, k, P_eff): return float(norm.ppf(1 - self.q * max(int(k), 1) / (2 * int(P_eff))))

    def pihat_ci(self): return 1.96 * np.sqrt(self.pihat * (1 - self.pihat) / self.n)

    def summary(self):
        if self.refusal: return {"refusal": self.refusal, "n": self.n, "P": self.P}
        return {"n": self.n, "P": self.P, "pihat": self.pihat, "khat": self.khat, "zq": self.zq, "hbar": self.hbar, "eta": self.eta, "normD2_deb": self.normD2_deb, "band_norm": self.band_norm, "nu_deb": self.nu_deb, "band_nu": self.band_nu, "Khat": self.Khat}


def _provable_version():
    try:
        import provable; return getattr(provable, "__version__", "?")
    except Exception:
        return "unavailable"
