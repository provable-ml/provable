"""Currency: columns-rows exchange, quadrichotomy lines, trichotomy, gate-before-dial (AC7, P-8.6).

Sheet statements implemented verbatim:
  n_eff(lam) = n (1 + lam^2 / (tau_v (1 - lam^2))) per direction (precision channel)
  nameability channel: no finite rows substitute; after purchase the direction re-prices as rows
  quadrichotomy: rounds / rows / columns / never; r = 0 collapses the columns line into geometry
  trichotomy: Dark_op = G_prop + M*_coh + M*_amp; M*_amp = (1-d^2) kappa_eff min(B^2, hbar^2/(n tau_eff))
  gate-before-dial invariant: never quote a row price on a gated (nameability) direction without
  the gate line first (enforced by render index in the order object).
"""
import numpy as np


class Currency:
    def __init__(self, state): self.st = state

    def n_eff(self, lam, tau_v): return float(self.st.n * (1 + lam * lam / (tau_v * max(1 - lam * lam, 1e-300))))

    def row_equivalent(self, lam, tau_v): return float(self.n_eff(lam, tau_v) - self.st.n)

    def m_star_amp(self, kappa_eff, B, tau_eff):
        h2 = self.st.hbar ** 2
        return float(self.st.one_minus_d2 * kappa_eff * min(B * B, h2 / (self.st.n * tau_eff)))

    def trichotomy(self, G_prop, M_coh, kappa_eff, B, tau_eff):
        amp = self.m_star_amp(kappa_eff, B, tau_eff)
        return {"G_prop": float(G_prop), "M_star_coh": float(M_coh), "M_star_amp": amp, "Dark_op": float(G_prop + M_coh + amp)}

    def quadrichotomy_lines(self, r, items):
        lines = []
        for it in items:
            bucket = it.get("bucket")
            if r == 0 and bucket == "columns": bucket = "geometry"
            lines.append({"line": it.get("line"), "bucket": bucket, "price": it.get("price"), "gated": bool(it.get("gated", False)), "gate_line": it.get("gate_line")})
        return lines

    def price_lines(self, purchased, tau_v):
        rows = []
        for pj in purchased:
            lam = float(pj["lam"])
            gated = bool(pj.get("nameability_gated", False))
            if gated: rows.append({"kind": "gate", "direction": pj["direction"], "text": f"NAMEABILITY GATE: direction {pj['direction']} has no finite-rows substitute pre-purchase (gate-before-dial)"})
            rows.append({"kind": "price", "direction": pj["direction"], "lam": lam, "row_equivalents": self.row_equivalent(lam, tau_v), "text": f"direction {pj['direction']}: lam={lam:.4f} re-prices as {self.row_equivalent(lam, tau_v):.1f} row-equivalents (n_eff={self.n_eff(lam, tau_v):.1f})"})
        return rows
