"""Localizer: realized localization with the estimation rider.

Feasible-dark output per AC2.3 + P-8.2 under pin P07: projector-valued tolerance set (top-J
Ghat-eigenspace holding >= 1-rho of the debiased contrast energy) or, iff J == 1 with a certified
spectral gap gammahat > 2*eta, a frame with the Davis-Kahan sin-theta <= 2*eta/gammahat quote.
Weyl eigenvalue intervals lamhat_k +/- eta ride every quote. theta*(M0, rho) per P17 is
display-only: theta* = pihat(1-pihat)*normD2_deb/(rho*M0).
"""
import numpy as np


class Localizer:
    def __init__(self, state): self.st = state

    def run(self, M0, rho):
        st = self.st
        proj = st.evecs.T @ st.Depshat
        energy = proj ** 2
        total = max(st.normD2_deb, 1e-9)
        csum = np.cumsum(energy)
        need = (1 - rho) * total
        J = int(np.searchsorted(csum, need) + 1)
        J = min(max(J, 1), st.Khat)
        gammahat = float(st.evals[0] - st.evals[1]) if st.P > 1 else float("inf")
        frame_ok = (J == 1) and (gammahat > 2 * st.eta)
        Pj = st.evecs[:, :J]
        out = {"type": "frame" if frame_ok else "tolerance", "J": J, "projector": Pj, "share_target": 1 - rho, "share_captured": float(min(csum[J - 1] / total, 1.0)) if total > 0 else 1.0, "gammahat": gammahat, "eta": st.eta, "eig_intervals": [(float(l - st.eta), float(l + st.eta)) for l in st.evals[:J]], "theta_star": float(st.pihat * (1 - st.pihat) * max(st.normD2_deb, 0.0) / (rho * M0)), "dk_sin_bound": float(min(2 * st.eta / gammahat, 1.0)) if frame_ok else None, "normD2_deb": st.normD2_deb, "band_norm": st.band_norm, "nu_deb": st.nu_deb, "band_nu": st.band_nu}
        return out

    def retention(self, loc, direction):
        d = np.asarray(direction, float); d = d / max(np.linalg.norm(d), 1e-300)
        Pj = loc["projector"]
        return float(np.sum((Pj.T @ d) ** 2))
