"""Folded AC-series acquisition check suite (the build of record).

Synthetic worlds W-A..W-R exercising the acquisition engine end to end. The
numeric lock targets are the sealed build's, carried verbatim; the registration
seal text and its hash-echo lock (provenance) are not reproduced here. Grader().
grade_all() returns the scoreboard.
"""
import numpy as np
from scipy.stats import norm
from tabulate import tabulate
from provable.purchase._engine.host import SyntheticHost
from provable.purchase._engine.state import SieveState
from provable.purchase._engine.localize import Localizer
from provable.purchase._engine.envelope import Envelope
from provable.purchase._engine.acquire import Acquisition
from provable.purchase._engine.label import LabelChannel
from provable.purchase._engine.currency import Currency
from provable.purchase._engine.gate import AdmissibilityGate
from provable.purchase._engine.contract import PurchaseOrder, SEAM_ENUM, BRIDGE_LINE
from provable.purchase._engine.audit import SelfAudit

ROOT = 20260910
ZDEL = 1.281552


class _PopState:
    def __init__(self, n, pi, zq):
        self.n, self.pihat = n, pi
        self.one_minus_d2 = 4 * pi * (1 - pi); self.d2hat = (2 * pi - 1) ** 2
        self.s_n = float(np.sqrt(n * pi * (1 - pi)))
        self.zq = zq; self.hbar = zq / self.s_n
        self.MhatF, self.band_nu, self.nu_deb, self.normD2_deb = 0.0, 0.0, 0.0, 0.0


class Grader:
    def __init__(self): self.rows, self.band_widths, self.renders = [], [], {}

    def lock(self, lock, world, target, observed, ok, mechanism=""):
        self.rows.append({"lock": lock, "world": world, "target": target, "observed": observed, "verdict": "PASS" if ok else "MISS", "mechanism": "" if ok else mechanism})

    def _collect_bands(self, fcs):
        for fc in fcs: self.band_widths.append(fc["p_hi"] - fc["p_lo"])

    def world_A(self):
        seed = ROOT + 1
        host = SyntheticHost(P=60, eigs=[2.5, 1.4, 1.0, 0.8, 0.65, 0.5, 0.4, 0.3], s2=0.04, pi=0.40, Dm=np.sqrt(1.5) * np.eye(8)[0], seed=seed, a_mode="flat_first")
        M0, rho, delta, q, kappa, B, tau_v = 0.06, 0.10, 0.10, 0.10, 1, 1.5, 1.0
        rng = np.random.default_rng(seed * 1000)
        Y, W, F = host.sample(900, rng)
        st = SieveState(q=q).fit(Y, W, run_provable=True, provable_seed=seed)
        loc = Localizer(st).run(M0, rho)
        env = Envelope(st.hbar, st.one_minus_d2)
        acq = Acquisition(st, env)
        lam_res = acq.lambda_A(M0, rho, delta)
        vstar = host.oracle_v()
        z_at = host.candidate_column(F, vstar, 0.634815, rng)
        cert_at = acq.certify_column(z_at, Y)
        gate = AdmissibilityGate()
        g_at = gate.evaluate({"lineage": "exogenous", "attested": True})
        g_post = gate.evaluate({"lineage": "exogenous", "attested": True, "surfaced_after_spec": True})
        fc_at = acq.forecast(0.634815, 1.0)
        fc_post = acq.forecast(0.30, 1.0)
        self._collect_bands([fc_at, fc_post])
        lc = LabelChannel(st.pihat)
        r2f = lc.r2_forecast(0.634815 ** 2 * st.MhatF)
        cur = Currency(st)
        prices = cur.price_lines([{"direction": "dark-1", "lam": 0.634815, "nameability_gated": True}], tau_v)
        aud = SelfAudit().reconcile([{"quantity": "z-margin", "forecast": fc_at["margin_z"], "realized": cert_at["z_col"] - st.zq, "band": max(fc_at["B_z"], 2.0)}])
        vW = 0.0
        if st.provable_block and st.provable_block["khat"] > 0: vW = float(np.max(st.Depshat ** 2))
        po = PurchaseOrder()
        po.set_header("W-A synthetic camouflage panel", "Y binary declared", st)
        po.set_promise(M0, rho, 1.0, delta, q)
        po.set("estimand", "E2-realized")
        po.set_feasible_dark(loc)
        po.set("spec_table", [{"stratum": "common", "channel": "coherence", "psi": f"{loc['type']} J={loc['J']}", "lam_min": round(lam_res["lam"], 6), "r": 1, "e3_N": "-", "e3_P": "-"}])
        po.set("stack_line", f"best nameable-penumbra discount: v_W={vW:.6f}, lam^2 required {acq.stack_lambda2(vW, st.MhatF, rho, delta):.6f} vs unassisted {acq.stack_lambda2(0.0, st.MhatF, rho, delta):.6f}")
        po.set("label_line", [{"candidate": "c_at", "r2_forecast": round(r2f, 6), "conditional": "candidate model R2 (AC6.1)", "self_audit_field": "post-purchase R2 reconciliation pending"}])
        po.set("candidate_scores", [{"name": "c_at", "gate_verdict": g_at["verdict"], "flags": g_at["flags"], "cos2": 1.0, "band": f"[{fc_at['p_lo']:.4f},{fc_at['p_hi']:.4f}]", "verdict": "ACCEPT vs spec" if 0.634815 >= lam_res["lam"] - 1e-9 else "REJECT_VS_SPEC"}, {"name": "c_posthoc", "gate_verdict": g_post["verdict"], "flags": g_post["flags"], "cos2": 1.0, "band": f"[{fc_post['p_lo']:.4f},{fc_post['p_hi']:.4f}]", "verdict": "REJECT_VS_SPEC"}])
        po.set("never_ceiling", {"auc": round(float(norm.cdf(np.sqrt(st.MhatF / 2))), 6), "acc": round(_acc_ceiling(st.MhatF, st.pihat), 6), "caveat": "in-host ceiling (factor host of record)", "bridge": BRIDGE_LINE})
        po.set("prices", prices)
        po.set("no_go", [])
        po.set("seam_citations", ["AC8a.1", "AC3.5", "AC5-F2", "AC2.3", "P-8.2", "AC4.5", "AC5.2", "AC5.3", "AC4.3", "P-8.4", "AC6.1", "AC6.5", "AC3.1(iii)", "P-8.3", "AC5.4(iii)", "P-8.5", "P-8.6", "AC7", "AC5.4(i)", "AC8a.2-ledger"])
        po.set("self_audit", aud)
        self.renders["W-A"] = po.render_console()
        self.po_A = po
        ret_hits = 0
        for i in range(50):
            r = np.random.default_rng(seed * 1000 + 1 + i)
            Yi, Wi, Fi = host.sample(900, r)
            sti = SieveState(q=q).fit(Yi, Wi)
            loci = Localizer(sti).run(M0, rho)
            if Localizer(sti).retention(loci, host.A[:, 0]) >= 0.85: ret_hits += 1
        self.lock("A1", "W-A", ">=45/50 retention>=0.85", f"{ret_hits}/50", ret_hits >= 45, "localizer misses the planted dark direction; check P07 energy rule or eigvec estimation")
        lamA_formula = float(np.sqrt((st.zstar(st.khat) + ZDEL) ** 2 / (900 * (1 - rho) * M0)))
        diff = abs(lam_res["lam"] - lamA_formula)
        self.lock("A2", "W-A", "<=1e-6", f"{diff:.2e}", diff <= 1e-6, "lambda_A implementation deviates from the frozen guaranteed law at the record khat")
        hits_at, hits_mid, hits_lo, fmid, flo, bw = 0, 0, 0, [], [], []
        for i in range(200):
            r = np.random.default_rng(seed * 1000 + 100 + i)
            Yi, Wi, Fi = host.sample(900, r)
            sti = SieveState(q=q).fit(Yi, Wi)
            acqi = Acquisition(sti, Envelope(sti.hbar, sti.one_minus_d2))
            for lam, sink_hits, sink_f in ((0.634815, "at", None), (0.276171, "mid", fmid), (0.15, "lo", flo)):
                zi = host.candidate_column(Fi, vstar, lam, r)
                ci = acqi.certify_column(zi, Yi)
                if sink_hits == "at": hits_at += int(ci["certify"])
                if sink_hits == "mid": hits_mid += int(ci["certify"])
                if sink_hits == "lo": hits_lo += int(ci["certify"])
                if sink_f is not None:
                    fci = acqi.forecast(lam, 1.0); sink_f.append(fci["p_mid"])
                    if sink_hits == "mid": bw.append(fci["B_z"])
            if i < 3: self._collect_bands([acqi.forecast(0.276171, 1.0)])
        rate_at, rate_mid, rate_lo = hits_at / 200, hits_mid / 200, hits_lo / 200
        self.lock("A3", "W-A", ">=0.95", f"{rate_at:.3f}", rate_at >= 0.95, "at-spec column fails to certify at the promised rate; check P12 statistic or unlock scale")
        okA4 = abs(rate_mid - np.mean(fmid)) <= 0.10 and 1.491313 <= np.mean(bw) <= 4.473939
        self.lock("A4", "W-A", "|obs-fc|<=0.10 and mean B_z in [1.491313,4.473939]", f"|{rate_mid:.3f}-{np.mean(fmid):.3f}|={abs(rate_mid-np.mean(fmid)):.3f}, mean_Bz={np.mean(bw):.3f}", okA4, "mid-probe forecast miscalibrated or band degenerate/inflated; check P10")
        self.lock("A5", "W-A", "|obs-fc|<=0.08", f"|{rate_lo:.3f}-{np.mean(flo):.3f}|={abs(rate_lo-np.mean(flo)):.3f}", abs(rate_lo - np.mean(flo)) <= 0.08, "below-spec failure rate off forecast; two-sided term or scale error in P10")
        self.lock("A6", "W-A", "POST-HOC flag on c_posthoc", str(g_post["flags"]), "POST-HOC" in g_post["flags"], "gate does not flag candidates surfaced after the spec was seen")
        return st

    def world_B(self):
        seed = ROOT + 2
        host = SyntheticHost(P=40, eigs=[2.0, 1.2, 0.8, 0.6, 0.4], s2=0.04, pi=0.50, Dm=np.sqrt(0.2) * np.eye(5)[0], seed=seed)
        M0, rho, delta, q = 0.02, 0.10, 0.10, 0.10
        rng = np.random.default_rng(seed * 1000)
        Y, W, F = host.sample(500, rng)
        st = SieveState(q=q).fit(Y, W, run_provable=True, provable_seed=seed)
        env = Envelope(st.hbar, st.one_minus_d2); acq = Acquisition(st, env)
        lam_res = acq.lambda_A(M0, rho, delta)
        n_req = acq.n_req_rows(st.nu_deb, rho, delta)
        lowsnr = st.band_nu > 0.5 * max(st.nu_deb, 1e-9)
        gate = AdmissibilityGate(); g_t = gate.evaluate({"lineage": "exogenous", "attested": True})
        fc_t = acq.forecast(0.9, 1.0); self._collect_bands([fc_t])
        verdict_t = "REJECT_VS_SPEC" if (not lam_res["feasible"] or 0.9 < (lam_res["lam"] or 2)) else "ACCEPT vs spec"
        po = PurchaseOrder()
        po.set_header("W-B synthetic amplitude-only panel", "Y binary declared", st)
        po.set_promise(M0, rho, 1.0, delta, q)
        po.set("estimand", "E2-guaranteed")
        po.set_feasible_dark(Localizer(st).run(M0, rho))
        po.set("spec_table", [])
        rows_line = f"ROWS route: INFEASIBLE_LAMBDA (lamA^2={lam_res['lam2']:.6f} > 1); n_req={n_req}" + (" LOWSNR" if lowsnr else "")
        po.set("stack_line", rows_line)
        po.set("label_line", [{"candidate": "-", "r2_forecast": "-", "conditional": "-", "self_audit_field": "-"}])
        po.set("candidate_scores", [{"name": "c_tempt(lam=0.9)", "gate_verdict": g_t["verdict"], "flags": g_t["flags"], "cos2": 1.0, "band": f"[{fc_t['p_lo']:.4f},{fc_t['p_hi']:.4f}]", "verdict": verdict_t}])
        po.set("never_ceiling", {"auc": round(float(norm.cdf(np.sqrt(st.MhatF / 2))), 6), "acc": round(_acc_ceiling(st.MhatF, st.pihat), 6), "caveat": "in-host ceiling", "bridge": BRIDGE_LINE})
        po.set("prices", [{"kind": "price", "direction": "rows", "text": rows_line}])
        po.set("no_go", [])
        po.set("seam_citations", ["AC8a.1", "AC3.5", "AC5-F2", "P-8.6", "AC7", "P-8.3"])
        po.set("self_audit", {"phase": "pre-purchase", "forecast": None, "realized": None, "band": None, "verdict": "NO_PURCHASE"})
        self.renders["W-B"] = po.render_console()
        self.po_B = po
        no_coh = len(po.fields["spec_table"]) == 0 and (not lam_res["feasible"])
        self.lock("B1", "W-B", "zero coherence rows and INFEASIBLE_LAMBDA", f"rows={len(po.fields['spec_table'])}, lamA^2={lam_res['lam2']:.4f}", no_coh, "module emitted a coherence spec where no admissible reliability exists (false purchase)")
        okB2 = (336 <= n_req <= 2523) or lowsnr
        self.lock("B2", "W-B", "n_req in [336,2523] or LOWSNR", f"n_req={n_req}, LOWSNR={lowsnr}", okB2, "rows quote outside the honest band without a LOWSNR flag; check P14")
        self.lock("B3", "W-B", "REJECT_VS_SPEC", verdict_t, verdict_t == "REJECT_VS_SPEC", "tempting candidate accepted under an infeasible guaranteed spec")

    def world_C(self):
        seed = ROOT + 3
        c = np.sqrt(0.288)
        host = SyntheticHost(P=30, eigs=[1.8, 1.2, 0.9, 0.6], s2=0.04, pi=0.30, Dm=c * np.ones(4), seed=seed)
        rho, delta, q = 0.10, 0.10, 0.10
        rng = np.random.default_rng(seed * 1000)
        Y, W, F = host.sample(3000, rng)
        st = SieveState(q=q).fit(Y, W, run_provable=True, provable_seed=seed)
        auc = float(norm.cdf(np.sqrt(st.MhatF / 2))); acc = _acc_ceiling(st.MhatF, st.pihat)
        okC1 = abs(auc - 0.780711) <= 0.025 and abs(acc - 0.757957) <= 0.025
        self.lock("C1", "W-C", "|AUC-0.780711|<=0.025 and |acc-0.757957|<=0.025", f"AUC={auc:.6f}, acc={acc:.6f}", okC1, "ceiling quote off the frozen truth; check debias or the ceiling closed forms")
        ceiling_M = st.pihat * (1 - st.pihat) * st.nu_deb
        refused = 0.30 > ceiling_M
        self.lock("C2", "W-C", "INFEASIBLE_M0 fires at M0=0.30", f"M0=0.30 vs pihat(1-pihat)nu_deb={ceiling_M:.4f} -> refusal={refused}", refused, "promise above the reachability ceiling not refused (AC8a.1)")
        gate = AdmissibilityGate()
        g_twin = gate.evaluate({"lineage": "exogenous", "attested": True, "twin_attested": True})
        po = PurchaseOrder()
        po.set_header("W-C synthetic never-ceiling panel", "Y binary declared", st)
        po.set_promise(0.10, rho, 1.0, delta, q)
        po.set("estimand", "E2-realized")
        po.set_feasible_dark(Localizer(st).run(0.10, rho))
        env = Envelope(st.hbar, st.one_minus_d2); acq = Acquisition(st, env)
        lamr = acq.lambda_A(0.10, rho, delta)
        po.set("spec_table", [{"stratum": "common", "channel": "coherence", "psi": "tolerance", "lam_min": round(lamr["lam"], 6) if lamr["feasible"] else None, "r": 1, "e3_N": "-", "e3_P": "-"}] if lamr["feasible"] else [])
        po.set("stack_line", "no certified penumbra; unassisted quote")
        po.set("label_line", [{"candidate": "-", "r2_forecast": "-", "conditional": "-", "self_audit_field": "-"}])
        po.set("candidate_scores", [{"name": "c_twin", "gate_verdict": g_twin["verdict"], "flags": g_twin["flags"], "cos2": None, "band": "-", "verdict": "NO-GO"}])
        po.set("never_ceiling", {"auc": round(auc, 6), "acc": round(acc, 6), "caveat": "in-host ceiling (factor host of record)", "bridge": BRIDGE_LINE})
        po.set("prices", [])
        po.set("no_go", ["c_twin (attested world-twin: zero visibility for every admissible Z)"])
        po.set("seam_citations", ["AC8a.1", "AC5.4(i)", "AC5.4(iii)", "P-8.5"])
        po.set("self_audit", {"phase": "pre-purchase", "forecast": None, "realized": None, "band": None, "verdict": "NO_PURCHASE"})
        self.renders["W-C"] = po.render_console()
        okC3 = g_twin["verdict"] == "NOGO_TWIN" and len(po.fields["no_go"]) == 1
        self.lock("C3", "W-C", "attested twin listed under no-go", f"verdict={g_twin['verdict']}, no_go={len(po.fields['no_go'])}", okC3, "twin attestation not routed to the no-go section")
        okC4 = "acc - 1/2 = 1 - c" in self.renders["W-C"]
        self.lock("C4", "W-C", "bridge line rendered", str(okC4), okC4, "never-ceiling section omits the cited bridge line")
        okC5 = len(po.refusals) == 0 and po.field_completeness()["complete"]
        self.lock("C5", "W-C", "M0=0.10 completes without refusal", f"refusals={len(po.refusals)}, complete={po.field_completeness()['complete']}", okC5, "feasible promise wrongly refused or contract left incomplete")

    def world_D(self):
        eigs = np.array([3.0, 2.4, 2.0, 1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4])
        s, kappa, B, pi, n, M0, tau = 7, 3, 0.8, 0.45, 2000, 0.03, 1.0
        zq = float(norm.ppf(1 - 0.10 * 1 / (2 * s)))
        st = _PopState(n, pi, zq)
        env = Envelope(st.hbar, st.one_minus_d2); acq = Acquisition(st, env)
        dead = acq.deadband_check(s, kappa, 4)
        walls = acq.wall_sequence(eigs[:s], s, kappa, B)
        move04 = walls[0] - walls[4]
        okD1 = dead and abs(move04) <= 1e-12
        self.lock("D1", "W-D", "DEADBAND refusal at r=4 and wall(0)-wall(4)=0", f"deadband={dead}, movement={move04:.2e}", okD1, "deadband not enforced or wall moves inside the dead zone; check P06 kappa-sparse fill")
        rres = acq.r_min(eigs[:s], s, kappa, B, M0)
        self.lock("D2", "W-D", "r_min==5", str(rres["r_min"]), rres["r_min"] == 5, "r_min law deviates from the frozen kappa-sparse capacity")
        lams = [0.9, 0.8, 0.7, 0.6, 0.5]
        zd_pop = ZDEL / st.s_n
        spec_rows = []
        for j in range(5):
            lam_min = min(1.0, (st.hbar + zd_pop) / np.sqrt(0.9 * eigs[j]))
            spec_rows.append({"stratum": f"support-{j+1}", "channel": "coherence(E3)", "psi": f"eig={eigs[j]}", "lam_min": round(lam_min, 6), "r": 1, "e3_N": None, "e3_P": None})
        e3n = env.e3_N_residual(eigs[5:s], kappa, B, lams)
        e3p = env.e3_P_delta(lams, tau)
        for row in spec_rows: row["e3_N"] = round(e3n["residual"], 8); row["e3_P"] = round(e3p, 8)
        okD3i = all(0 < r["lam_min"] <= 1 for r in spec_rows) and len(spec_rows) == 5 and not dead if False else (all(0 < r["lam_min"] <= 1 for r in spec_rows) and len(spec_rows) == 5)
        okD3ii = abs(e3p - 0.01958399) <= 1e-6
        okD3iii = abs(e3n["residual"] - 0.16510678) <= 1e-6 and abs(e3n["binding_lam"] - 0.5) <= 1e-12
        seq_ok = all(walls[r + 1] <= walls[r] + 1e-12 for r in range(s)) and all(walls[r + 1] < walls[r] - 1e-12 for r in (4, 5, 6))
        okD3 = okD3i and okD3ii and okD3iii and seq_ok
        self.lock("D3", "W-D", "5 rows lam_min in (0,1]; E3-P=0.01958399; E3-N=0.16510678 binding lam=0.5; wall monotone", f"rows={len(spec_rows)}, E3-P={e3p:.8f}, E3-N={e3n['residual']:.8f}, binding={e3n['binding_lam']}, monotone={seq_ok}", okD3, "E3 channel constants or wall monotonicity off the frozen values")
        self.wD = {"walls": walls, "spec_rows": spec_rows, "e3n": e3n, "e3p": e3p}

    def world_E(self):
        seed = ROOT + 5
        host = SyntheticHost(P=30, eigs=[1.8, 1.2, 0.9, 0.6], s2=0.04, pi=0.40, Dm=1.2 * np.eye(4)[0], seed=seed)
        lam, q = 0.6, 0.10
        vstar = host.oracle_v()
        lc = LabelChannel(0.40)
        r2f_true = lc.r2_forecast(lam * lam * host.M_F())
        r2_real, ident_max = [], 0.0
        for i in range(100):
            r = np.random.default_rng(seed * 1000 + i)
            Y, W, F = host.sample(2000, r)
            z = host.candidate_column(F, vstar, lam, r)
            piZ = host.pi_Z_true(z, lam, vstar)
            r2_real.append(lc.realized_r2(piZ))
            res, _ = lc.identity_residual(piZ)
            ident_max = max(ident_max, res)
        gapE1 = abs(r2f_true - float(np.mean(r2_real)))
        self.lock("E1", "W-E", "<=0.01201829", f"{gapE1:.6f} (forecast {r2f_true:.6f} vs realized {np.mean(r2_real):.6f})", gapE1 <= 0.01201829, "small-signal R2 forecast off the truth-model realization beyond the frozen tolerance")
        self.lock("E2", "W-E", "<=1e-9", f"{ident_max:.2e}", ident_max <= 1e-9, "exact identity broken between the two code paths (AC6.1)")
        v = host.sigma_unit(vstar)
        cS = lam * host.A @ host.Sigma_F @ v
        Dz = lam * float(v @ host.Dm)
        G_aug = np.block([[host.G, cS[:, None]], [cS[None, :], np.array([[1.0]])]])
        D_aug = np.concatenate([host.Deps, [Dz]])
        nu_aug = float(D_aug @ np.linalg.solve(G_aug + 1e-10 * np.eye(31), D_aug))
        stE = _PopState(2000, 0.40, 3.0)
        acq = Acquisition(stE, Envelope(stE.hbar, stE.one_minus_d2))
        dnu = acq.delta_nu(host.Deps, host.G, cS, Dz)
        resE3 = acq.conservation_check(host.nu_full(), nu_aug, dnu)
        self.lock("E3", "W-E", "<=1e-8", f"{resE3:.2e}", resE3 <= 1e-8, "block conservation identity violated; check delta_nu Schur form")
        lt = lc.lever_table(V_res=0.05, R2_realized=float(np.mean(r2_real)), field_gain=0.02)
        e3eq = abs(lt["e3_worst_stratified"] - lt["e3_worst_dodge"])
        okE4 = lt["field_le_stratify"] and lt["stratify_le_dodge"] and e3eq <= 1e-9
        self.lock("E4", "W-E", "field<=stratify<=dodge and E3-worst equal", f"field_le={lt['field_le_stratify']}, strat_le={lt['stratify_le_dodge']}, |eq|={e3eq:.2e}", okE4, "lever table comparators broken")
        Yr = np.random.default_rng(seed * 1000 + 999)
        Y, W, F = host.sample(2000, Yr)
        z = host.candidate_column(F, vstar, lam, Yr)
        st = SieveState(q=q).fit(Y, W)
        acq2 = Acquisition(st, Envelope(st.hbar, st.one_minus_d2))
        c2 = min(1.0, acq2.certify_column(z, Y)["nu_z_hat"] / max(lam * lam * st.MhatF, 1e-9))
        r2f_mod = st.pihat * (1 - st.pihat) * lam * lam * st.MhatF * c2
        zs = (z - z.mean()) / max(z.std(ddof=1), 1e-12)
        mu1, mu0 = zs[Y == 1].mean(), zs[Y == 0].mean()
        f1, f0 = norm.pdf(zs - mu1), norm.pdf(zs - mu0)
        piZ_fit = st.pihat * f1 / np.maximum(st.pihat * f1 + (1 - st.pihat) * f0, 1e-300)
        r2_realized_mod = LabelChannel(st.pihat).realized_r2(piZ_fit)
        bandR2 = st.pihat * (1 - st.pihat) * lam * lam * st.band_nu + 0.01
        aud = SelfAudit().reconcile([{"quantity": "R2_Z", "forecast": r2f_mod, "realized": r2_realized_mod, "band": bandR2}])
        numeric = abs(r2_realized_mod - r2f_mod) <= bandR2
        okO8 = (aud["verdict"] == "RECONCILED") == numeric
        self.lock("O8", "W-E", "audit verdict equals its own numeric comparison", f"verdict={aud['verdict']}, numeric={numeric}", okO8, "self-audit verdict inconsistent with the comparison it quotes")

    def world_N(self):
        seed = ROOT + 6
        host = SyntheticHost(P=50, eigs=[3.0, 2.0, 1.3, 0.9, 0.7, 0.5], s2=0.04, pi=0.45, Dm=np.sqrt(2.1) * np.eye(6)[0], seed=seed)
        vstar = host.oracle_v()
        grid = [(300, 0.431489), (1000, 0.236336), (3000, 0.136449), (10000, 0.074736)]
        true_eigs = np.array([10.88018769, 9.41284752, 6.27481467, 4.66972795, 4.01630823, 3.17180570])
        n4_pass, frame_ok, frame_tot = 0, 0, 0
        per_rung = []
        for n, lam_probe in grid:
            hits1 = hits2 = hits3 = cert = 0
            plos, phis = [], []
            for i in range(60):
                r = np.random.default_rng(seed * 1000 + n + i)
                Y, W, F = host.sample(n, r)
                st = SieveState(q=0.10).fit(Y, W)
                err = float(np.max(np.abs(st.evals[:6] - true_eigs)))
                hits1 += int(err <= st.eta)
                hits2 += int(abs(st.normD2_deb - 7.43065562) <= st.band_norm)
                hits3 += int(abs(st.nu_deb - 0.69078947) <= st.band_nu)
                loc = Localizer(st).run(0.05, 0.10)
                should_frame = (loc["J"] == 1) and (loc["gammahat"] > 2 * st.eta)
                frame_ok += int((loc["type"] == "frame") == should_frame); frame_tot += 1
                acq = Acquisition(st, Envelope(st.hbar, st.one_minus_d2))
                z = host.candidate_column(F, vstar, lam_probe, r)
                cert += int(acq.certify_column(z, Y)["certify"])
                fc = acq.forecast(lam_probe, 1.0)
                plos.append(fc["p_lo"]); phis.append(fc["p_hi"])
                if i < 2: self._collect_bands([fc])
            rate = cert / 60; sdb = 2 * np.sqrt(0.25 / 60)
            inband = (np.mean(plos) - sdb) <= rate <= (np.mean(phis) + sdb)
            n4_pass += int(inband)
            per_rung.append((n, hits1 / 60, hits2 / 60, hits3 / 60, rate, np.mean(plos), np.mean(phis), inband))
        okN1 = all(r[1] >= 0.85 for r in per_rung)
        okN2 = all(r[2] >= 0.85 for r in per_rung)
        okN3 = all(r[3] >= 0.85 for r in per_rung)
        self.lock("N1", "W-N", ">=0.85 each rung", str([f"{r[1]:.2f}" for r in per_rung]), okN1, "eta bound dishonest at some n; Weyl constant C_ETA or eigen estimator off")
        self.lock("N2", "W-N", ">=0.85 each rung", str([f"{r[2]:.2f}" for r in per_rung]), okN2, "norm-debias band dishonest; P05 variance form off")
        self.lock("N3", "W-N", ">=0.85 each rung", str([f"{r[3]:.2f}" for r in per_rung]), okN3, "nu-debias band dishonest; P05 variance or eta term off")
        self.lock("N4", "W-N", ">=3 of 4 rungs in band", f"{n4_pass}/4 " + str([(r[0], f"{r[4]:.2f} in [{r[5]:.2f},{r[6]:.2f}]") for r in per_rung]), n4_pass >= 3, "forecast bands not honest across the ladder; P10 band terms off")
        self.lock("N5", "W-N", ">=0.98 conformance", f"{frame_ok/frame_tot:.4f}", frame_ok / frame_tot >= 0.98, "frame emitted off the P07 rule")
        self.per_rung = per_rung

    def world_R(self):
        seed = ROOT + 7
        host = SyntheticHost(P=60, eigs=[2.0, 1.0], s2=0.04, pi=0.5, Dm=np.array([0.5, 0.0]), seed=seed)
        rng = np.random.default_rng(seed * 1000)
        Y, W, F = host.sample(80, rng)
        st = SieveState(q=0.10).fit(Y, W)
        po = PurchaseOrder().set_header("W-R rider refusal panel", "Y binary declared", st)
        self.renders["W-R"] = po.render_console()
        okR1 = st.refusal is not None and "REFUSE_RIDER" in self.renders["W-R"]
        self.lock("R1", "W-R", "REFUSE_RIDER raised and rendered", f"refusal={'set' if st.refusal else 'absent'}", okR1, "rider regime violation not refused (P03)")

    def contract_locks(self):
        comp = self.po_A.field_completeness()
        self.lock("O1", "W-A", "all 13 fields populated", f"missing={comp['missing']}", comp["complete"], "AC8a.4 contract incomplete")
        okO2 = self.po_A.gate_before_dial_ok()
        self.lock("O2", "W-A", "gate line precedes price line for gated directions", str(okO2), okO2, "gate-before-dial ordering violated (P-8.6)")
        okO3 = all(w > 0 for w in self.band_widths) and len(self.band_widths) > 0
        self.lock("O3", "all", f"all {len(self.band_widths)} collected forecast bands have width > 0", f"min width={min(self.band_widths):.4g}", okO3, "a forecast collapsed to a point (P10 violated)")
        cits = set(self.po_A.fields["seam_citations"]) | set(self.po_B.fields["seam_citations"])
        okO5 = len(cits) > 0 and cits.issubset(set(SEAM_ENUM))
        self.lock("O5", "W-A/W-B", "seam citations nonempty subset of P18 enum", f"{len(cits)} cited, subset={cits.issubset(set(SEAM_ENUM))}", okO5, "citation outside the frozen enum")
        import json as _json
        okO6 = True
        try:
            _json.loads(self.po_A.render_json()); _json.loads(self.po_B.render_json())
        except Exception as e:
            okO6 = False
        self.lock("O6", "W-A/W-B", "JSON round-trips", str(okO6), okO6, "JSON renderer emits non-parseable output")
        okO7 = "P-8.5 bridge" in self.po_A.fields["promise"]["economics"]
        self.lock("O7", "W-A", "economics line present (P-8.5 bridge)", str(okO7), okO7, "promise block missing the economics line")

    def grade_all(self):
        self.world_A(); self.world_B(); self.world_C(); self.world_D(); self.world_E(); self.world_N(); self.world_R(); self.contract_locks()
        return self.report()

    def report(self):
        rows = [[r["lock"], r["world"], r["target"], r["observed"], r["verdict"]] for r in self.rows]
        table = tabulate(rows, headers=["lock", "world", "target", "observed", "verdict"], tablefmt="psql")
        misses = [r for r in self.rows if r["verdict"] == "MISS"]
        n_pass = sum(1 for r in self.rows if r["verdict"] == "PASS")
        lines = ["ACQUISITION ENGINE GRADING (folded AC-series check suite; scientific locks only)", "", f"Locks: {len(self.rows)} graded, {n_pass} PASS, {len(misses)} MISS", "", table, ""]
        if misses:
            lines += ["MISS MECHANISMS", tabulate([[m["lock"], m["mechanism"]] for m in misses], headers=["lock", "mechanism"], tablefmt="psql"), ""]
        lines += ["W-N LADDER DETAIL", tabulate([[n, f"{a:.3f}", f"{b:.3f}", f"{c:.3f}", f"{d:.3f}", f"[{e:.3f},{f:.3f}]", g] for n, a, b, c, d, e, f, g in self.per_rung], headers=["n", "eig cov", "norm cov", "nu cov", "probe rate", "forecast band", "in band"], tablefmt="psql"), ""]
        lines += ["ORDER OF RECORD: W-A", self.renders["W-A"], "", "ORDER OF RECORD: W-B", self.renders["W-B"], "", "ORDER OF RECORD: W-R (refusal)", self.renders["W-R"]]
        text = "\n".join(lines)
        return {"pass": n_pass, "miss": len(misses), "rows": self.rows, "text": text}


def _acc_ceiling(M_F, pi):
    m = np.sqrt(max(M_F, 1e-12)); c = np.log((1 - pi) / pi)
    err = pi * norm.cdf((c - m * m / 2) / m) + (1 - pi) * norm.cdf((-c - m * m / 2) / m)
    return float(1 - err)
