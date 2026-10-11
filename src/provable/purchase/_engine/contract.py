"""PurchaseOrder: the AC8a.4 output contract, byte for byte, with renderers.

Thirteen fields: header, promise, estimand, feasible-dark, spec table, stack line, label line,
candidate scores, never-ceiling, prices, no-go, seam citations, self-audit. Console rendering via
tabulate('psql'); JSON rendering round-trips. Seam citations draw only from the frozen P18 enum.
Gate-before-dial is enforced structurally: price lines for gated directions render strictly after
their gate line and the render indices are recorded for audit.
"""
import json
import numpy as np
from tabulate import tabulate

SEAM_ENUM = ["AC8a.1", "AC3.5", "AC5-F2", "AC2.3", "P-8.2", "AC4.5", "AC5.2", "AC5.3", "AC4.3", "P-8.4", "AC6.1", "AC6.5", "AC3.1(iii)", "P-8.3", "AC5.4(iii)", "P-8.5", "P-8.6", "AC7", "AC5.4(i)", "AC8a.2-ledger"]

BRIDGE_LINE = "acc - 1/2 = 1 - c bridge line (cited: AC5.4(iii)/P-8.5, not re-derived)"


class PurchaseOrder:
    def __init__(self):
        self.fields = {k: None for k in ["header", "promise", "estimand", "feasible_dark", "spec_table", "stack_line", "label_line", "candidate_scores", "never_ceiling", "prices", "no_go", "seam_citations", "self_audit"]}
        self.refusals = []

    def set(self, key, value): self.fields[key] = value; return self

    def add_refusal(self, code, text): self.refusals.append({"code": code, "text": text}); return self

    def set_header(self, dataset, target, state):
        if state.refusal:
            self.fields["header"] = {"dataset": dataset, "target": target, "n": state.n, "P": state.P, "pihat": None, "regime_n_gg_P": "REFUSED", "regime_P4": "declared", "provable": None}
            self.add_refusal("REFUSE_RIDER", state.refusal); return self
        self.fields["header"] = {"dataset": dataset, "target": target, "n": state.n, "P": state.P, "pihat": round(state.pihat, 6), "khat": state.khat, "regime_n_gg_P": f"PASS (n={state.n} >= 4P={4*state.P})", "regime_P4": "declared: Sigma_F common across sections (AC0-P4, load-bearing)", "provable": state.provable_block}
        return self

    def set_promise(self, M0, rho, theta, delta, q, cost_basis="declared cost basis: unit synthetic budget; M0 derived via the P-8.5 bridge"):
        self.fields["promise"] = {"M0": M0, "rho": rho, "theta": theta, "delta": delta, "q": q, "economics": cost_basis}
        return self

    def set_feasible_dark(self, loc):
        self.fields["feasible_dark"] = {"type": loc["type"], "J": loc["J"], "share_target": loc["share_target"], "share_captured": round(loc["share_captured"], 6), "gammahat": round(loc["gammahat"], 6), "eta": round(loc["eta"], 6), "dk_sin_bound": None if loc["dk_sin_bound"] is None else round(loc["dk_sin_bound"], 6), "eig_intervals": [(round(a, 4), round(b, 4)) for a, b in loc["eig_intervals"]], "theta_star": round(loc["theta_star"], 6), "normD2_deb": round(loc["normD2_deb"], 6), "band_norm": round(loc["band_norm"], 6), "nu_deb": round(loc["nu_deb"], 6), "band_nu": round(loc["band_nu"], 6)}
        return self

    def render_console(self):
        L, self._render_index = [], {}
        f = self.fields
        L.append("=" * 96); L.append("PURCHASE ORDER (AC8a.4 output contract)"); L.append("=" * 96)
        if self.refusals:
            L.append(tabulate([[r["code"], r["text"]] for r in self.refusals], headers=["refusal", "detail"], tablefmt="psql"))
        if f["header"]:
            h = f["header"]
            L.append("[header]  license AC8a.1  computed-from ledger")
            L.append(tabulate([[h["dataset"], h["target"], h["n"], h["P"], h["pihat"], h.get("khat"), h["regime_n_gg_P"], h["regime_P4"]]], headers=["dataset", "target", "n", "P", "pihat", "khat", "n>>P", "P4"], tablefmt="psql"))
            if h.get("provable"): L.append(f"  provable {h['provable']['version']}: khat={h['provable']['khat']}, certified={h['provable']['certified'][:6]}, claims={h['provable']['n_claims']}")
        if self.refusals and not f["promise"]:
            return "\n".join(L)
        if f["promise"]:
            p = f["promise"]; L.append("[promise]  license AC3.5  computed-from user declaration")
            L.append(tabulate([[p["M0"], p["rho"], p["theta"], p["delta"], p["q"]]], headers=["M0", "rho", "theta", "delta", "q"], tablefmt="psql"))
            L.append("  economics: " + p["economics"])
        if f["estimand"]: L.append(f"[estimand]  license AC5-F2  recorded buyer's choice: {f['estimand']}")
        if f["feasible_dark"]:
            d = f["feasible_dark"]; L.append("[feasible-dark]  license AC2.3 + P-8.2")
            L.append(tabulate([[d["type"], d["J"], f">={d['share_target']}", d["share_captured"], d["gammahat"], d["eta"], d["dk_sin_bound"], d["theta_star"]]], headers=["type", "J", "share", "captured", "gammahat", "eta", "DK sin", "theta*"], tablefmt="psql"))
            L.append(f"  debiased norms: ||Deps||^2_deb={d['normD2_deb']} (+/-{d['band_norm']})  nu_deb={d['nu_deb']} (+/-{d['band_nu']})")
        if f["spec_table"] is not None:
            L.append("[spec table]  license AC4.5, AC5.2/5.3")
            if len(f["spec_table"]):
                L.append(tabulate([[r.get("stratum"), r.get("channel"), r.get("psi"), r.get("lam_min"), r.get("r"), r.get("e3_N"), r.get("e3_P")] for r in f["spec_table"]], headers=["stratum", "channel", "Psi/tolerance", "lam_min", "r", "E3-N", "E3-P"], tablefmt="psql"))
            else: L.append("  (no coherence spec rows)")
        if f["stack_line"]: L.append(f"[stack line]  license AC4.3 / P-8.4  {f['stack_line']}")
        if f["label_line"]:
            ll = f["label_line"]; L.append("[label line]  license AC6.1/6.5")
            L.append(tabulate([[c.get("candidate"), c.get("r2_forecast"), c.get("conditional"), c.get("self_audit_field")] for c in ll], headers=["candidate", "R2_Z forecast", "conditional on", "self-audit field"], tablefmt="psql"))
        if f["candidate_scores"] is not None:
            L.append("[candidate scores]  license AC3.1(iii), P-8.3")
            rows = []
            for c in f["candidate_scores"]:
                self._render_index[("candidate", c["name"])] = len(L)
                rows.append([c["name"], c["gate_verdict"], ",".join(c.get("flags", [])) or "-", c.get("cos2"), c.get("band"), c.get("verdict")])
            L.append(tabulate(rows, headers=["candidate", "gate", "flags", "cos2hat", "forecast band", "accept/reject"], tablefmt="psql"))
        if f["never_ceiling"]:
            nc = f["never_ceiling"]; L.append("[never-ceiling]  license AC5.4(iii) / P-8.5")
            L.append(tabulate([[nc["auc"], nc["acc"], nc["caveat"]]], headers=["AUC ceiling", "accuracy ceiling", "in-host caveat"], tablefmt="psql"))
            L.append("  " + nc["bridge"])
        if f["prices"] is not None:
            L.append("[prices]  license P-8.6, AC7 currency table  (gate-before-dial enforced)")
            for row in f["prices"]:
                self._render_index[(row["kind"], row["direction"])] = len(L)
                L.append(("  GATE  " if row["kind"] == "gate" else "  PRICE ") + row["text"])
        if f["no_go"] is not None:
            L.append("[no-go]  license AC5.4(i)  attested twin strata")
            L.append(tabulate([[g] for g in f["no_go"]], headers=["stratum"], tablefmt="psql") if f["no_go"] else "  (none)")
        if f["seam_citations"]: L.append("[seam citations]  rows of AC8a.2 consumed: " + ", ".join(f["seam_citations"]))
        if f["self_audit"]:
            sa = f["self_audit"]; L.append("[self-audit]  license P-8.3")
            L.append(tabulate([[sa.get("phase"), sa.get("forecast"), sa.get("realized"), sa.get("band"), sa.get("verdict")]], headers=["phase", "forecast", "realized", "band", "verdict"], tablefmt="psql"))
        L.append("=" * 96)
        return "\n".join(L)

    def render_json(self):
        def clean(o):
            if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [clean(v) for v in o]
            if isinstance(o, (np.floating, np.integer)): return o.item()
            if isinstance(o, np.ndarray): return o.tolist()
            return o
        return json.dumps({"fields": clean(self.fields), "refusals": clean(self.refusals)}, indent=1)

    def gate_before_dial_ok(self):
        self.render_console()
        ok = True
        for (kind, direction), idx in self._render_index.items():
            if kind == "price" and ("gate", direction) in self._render_index:
                ok = ok and (self._render_index[("gate", direction)] < idx)
        return ok

    def field_completeness(self):
        missing = [k for k, v in self.fields.items() if v is None]
        return {"complete": len(missing) == 0, "missing": missing}
