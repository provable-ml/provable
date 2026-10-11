"""Envelope: water-filling V, sharp conditional envelope, residual walls (E3 two channels).

Water-filling per the formula sheet with the P06 kappa-sparse binding: fill eigenvalues
descending; each filled direction yields hbar^2 at amplitude cost hbar^2/lambda_k, per-direction
cost capped at B^2, at most kappa directions, total budget Q = kappa*B^2; V = K* hbar^2 +
lambda_(K*+1)(Q - spent), zero beyond the support. Sharp conditional envelope (common support,
AC3-P1): M_Y <= (1-d^2) V(lambda; hbar^2, kappa B^2), attained by the antisymmetric two-level
fill R_k = hbar/sqrt(lambda_k). E3 channels per AC5.3 and P13:
(N) residual = Cap_nu(unpurchased dodge) + sum_j hbar^2/lambda_j^2, smallest purchased BINDING;
(P) Delta M* = (1-d^2) hbar^2 sum_j lambda_j^2/(tau (tau + lambda_j^2)).
"""
import numpy as np


class Envelope:
    def __init__(self, hbar, one_minus_d2): self.hbar, self.one_minus_d2 = float(hbar), float(one_minus_d2)

    def water_fill(self, eigs, kappa, B):
        eigs = np.sort(np.asarray(eigs, float))[::-1]
        h2, Q = self.hbar ** 2, kappa * B * B
        spent, V, filled = 0.0, 0.0, 0
        for lam in eigs:
            if filled >= kappa: break
            c = h2 / lam
            if c > B * B: continue
            if spent + c <= Q: spent += c; V += h2; filled += 1
            else:
                V += lam * (Q - spent); spent = Q; filled += 1; break
        return {"V": float(V), "K_star": int(filled), "spent": float(spent), "Q": float(Q)}

    def cap_nu(self, eigs, kappa, B): return self.water_fill(eigs, kappa, B)["V"]

    def envelope_MY(self, eigs, kappa, B): return self.one_minus_d2 * self.cap_nu(eigs, kappa, B)

    def two_level_fill(self, eigs, kappa, B):
        eigs = np.sort(np.asarray(eigs, float))[::-1]
        wf = self.water_fill(eigs, kappa, B)
        amps = np.zeros(len(eigs)); signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(eigs))])
        for k in range(wf["K_star"]): amps[k] = self.hbar / np.sqrt(eigs[k])
        return signs * amps

    def wall(self, remaining_eigs, kappa, B): return self.one_minus_d2 * self.cap_nu(remaining_eigs, kappa, B)

    def e3_N_residual(self, unpurchased_eigs, kappa, B, purchased_lams):
        lams = np.asarray(purchased_lams, float)
        cap = self.cap_nu(unpurchased_eigs, kappa, B)
        floors = (self.hbar ** 2) / np.maximum(lams ** 2, 1e-300)
        binding = int(np.argmin(lams)) if len(lams) else None
        return {"residual": float(cap + floors.sum()), "cap_dodge": float(cap), "floors": floors.tolist(), "binding_index": binding, "binding_lam": float(lams[binding]) if binding is not None else None}

    def e3_P_delta(self, purchased_lams, tau):
        lams = np.asarray(purchased_lams, float)
        return float(self.one_minus_d2 * self.hbar ** 2 * np.sum(lams ** 2 / (tau * (tau + lams ** 2))))
