import numpy as np
import pytest
from scipy.stats import norm
from provable.purchase._engine.host import SyntheticHost
from provable.purchase._engine.state import SieveState
from provable.purchase._engine.localize import Localizer
from provable.purchase._engine.envelope import Envelope
from provable.purchase._engine.acquire import Acquisition
from provable.purchase._engine.label import LabelChannel
from provable.purchase._engine.currency import Currency
from provable.purchase._engine.gate import AdmissibilityGate
from provable.purchase._engine.contract import PurchaseOrder, SEAM_ENUM


def make_host(seed=77701):
    return SyntheticHost(P=20, eigs=[2.0, 1.0, 0.5], s2=0.05, pi=0.4, Dm=np.array([1.0, 0.3, 0.0]), seed=seed)


def test_host_geometry():
    h = make_host()
    assert np.allclose(h.A.T @ h.A, np.eye(3), atol=1e-10)
    nu_direct = float(h.Deps @ np.linalg.solve(h.G, h.Deps))
    assert abs(nu_direct - h.nu_full()) < 1e-10
    v = h.oracle_v()
    assert abs(v @ h.Sigma_F @ v - 1.0) < 1e-12
    rng = np.random.default_rng(77702)
    Y, W, F = h.sample(60000, rng)
    z = h.candidate_column(F, v, 0.7, rng)
    assert abs(np.var(z[Y == 0]) - 1.0) < 0.03 and abs(np.var(z[Y == 1]) - 1.0) < 0.03


def test_envelope_waterfill_and_channels():
    env = Envelope(hbar=0.1, one_minus_d2=0.96)
    wf = env.water_fill([2.0, 1.0, 0.5], kappa=2, B=1.0)
    assert wf["K_star"] == 2 and abs(wf["V"] - 0.02) < 1e-12
    wf2 = env.water_fill([2.0, 1.0], kappa=2, B=0.05)
    assert wf2["K_star"] == 0 and wf2["V"] == 0.0
    tl = env.two_level_fill([2.0, 1.0, 0.5], kappa=2, B=1.0)
    yields = np.array([2.0, 1.0, 0.5]) * tl ** 2
    assert np.allclose(sorted(yields[yields > 0]), [0.01, 0.01])
    e3n = env.e3_N_residual([1.0, 0.5], kappa=2, B=1.0, purchased_lams=[0.8, 0.4])
    assert abs(e3n["residual"] - (0.02 + 0.01 / 0.64 + 0.01 / 0.16)) < 1e-12
    assert e3n["binding_lam"] == 0.4
    e3p = env.e3_P_delta([0.5], tau=1.0)
    assert abs(e3p - 0.96 * 0.01 * 0.25 / 1.25) < 1e-12


def test_acquire_schur_and_laws():
    h = make_host()
    v = h.oracle_v()
    cS = 0.6 * h.A @ h.Sigma_F @ v
    Dz = 0.6 * float(v @ h.Dm)
    G_aug = np.block([[h.G, cS[:, None]], [cS[None, :], np.array([[1.0]])]])
    D_aug = np.concatenate([h.Deps, [Dz]])
    nu_aug = float(D_aug @ np.linalg.solve(G_aug, D_aug))
    class St: pass
    st = St(); st.n, st.pihat = 1000, 0.4
    st.one_minus_d2 = 0.96; st.s_n = np.sqrt(1000 * 0.24); st.zq = 3.0; st.hbar = 3.0 / st.s_n
    st.MhatF, st.band_nu, st.nu_deb = 0.5, 0.05, 0.5
    acq = Acquisition(st, Envelope(st.hbar, st.one_minus_d2))
    dnu = acq.delta_nu(h.Deps, h.G, cS, Dz)
    assert abs(nu_aug - h.nu_full() - dnu) < 1e-8
    la = acq.lambda_A(M0=0.05, rho=0.1, delta=0.1)
    reduced = (st.zq + norm.ppf(0.9)) ** 2 / (1000 * 0.9 * 0.05)
    assert abs(la["lam2"] - reduced) < 1e-12
    fc = acq.forecast(0.5, 1.0)
    assert fc["p_hi"] > fc["p_lo"] and fc["B_z"] >= 0.05
    rng = np.random.default_rng(77703)
    Y = (rng.random(4000) < 0.4).astype(int)
    z = rng.standard_normal(4000) + 0.8 * Y
    st.s_n = np.sqrt(4000 * 0.24)
    assert acq.certify_column(z, Y)["certify"]


def test_ledger_refusal_and_bars():
    rng = np.random.default_rng(77704)
    st = SieveState().fit((rng.random(30) < 0.5).astype(int), rng.standard_normal((30, 10)))
    assert st.refusal and "REFUSE_RIDER" in st.refusal
    h = make_host()
    Y, W, _ = h.sample(400, rng)
    st2 = SieveState(q=0.1).fit(Y, W)
    assert st2.refusal is None
    assert abs(st2.zstar(0) - st2.zstar(1)) < 1e-12
    assert st2.band_nu > 0 and st2.eta > 0


def test_localizer_retention_and_rule():
    h = make_host(77705)
    rng = np.random.default_rng(77706)
    Y, W, _ = h.sample(3000, rng)
    st = SieveState(q=0.1).fit(Y, W)
    lz = Localizer(st)
    loc = lz.run(M0=0.05, rho=0.1)
    assert loc["type"] in ("frame", "tolerance") and 1 <= loc["J"] <= st.Khat
    full = np.eye(st.P)[:, 0]
    locJ = dict(loc); locJ["projector"] = np.eye(st.P)
    assert abs(lz.retention(locJ, full) - 1.0) < 1e-12


def test_label_identity_and_walls():
    lc = LabelChannel(0.4)
    rng = np.random.default_rng(77707)
    piZ = np.clip(0.4 + 0.1 * rng.standard_normal(5000), 0.01, 0.99)
    res, R2 = lc.identity_residual(piZ)
    assert res < 1e-12
    w = lc.walls(V_res=0.1, R2_Z=0.2)
    assert w["wall_stratified"] < w["wall_dodge"]
    lt = lc.lever_table(0.1, 0.2, 0.03)
    assert lt["field_le_stratify"] and lt["stratify_le_dodge"]
    assert abs(lt["e3_worst_stratified"] - lt["e3_worst_dodge"]) < 1e-12


def test_currency_and_gate():
    class St: pass
    st = St(); st.n = 1000; st.hbar = 0.1; st.one_minus_d2 = 0.96
    cur = Currency(st)
    assert abs(cur.n_eff(0.5, 1.0) - 1000 * (1 + 0.25 / 0.75)) < 1e-9
    assert cur.m_star_amp(2, 0.5, 1.0) == 0.96 * 2 * min(0.25, 0.01 / 1000)
    lines = cur.price_lines([{"direction": "d1", "lam": 0.5, "nameability_gated": True}], 1.0)
    assert lines[0]["kind"] == "gate" and lines[1]["kind"] == "price"
    g = AdmissibilityGate()
    assert g.evaluate({"lineage": "y_descendant"})["verdict"] == "REFUSE_LINEAGE"
    assert g.evaluate({"lineage": "unknown"})["verdict"] == "REFUSE_UNATTESTED"
    assert g.evaluate({"lineage": "exogenous", "attested": True, "twin_attested": True})["verdict"] == "NOGO_TWIN"
    assert "POST-HOC" in g.evaluate({"lineage": "exogenous", "attested": True, "surfaced_after_spec": True})["flags"]


def test_order_contract():
    po = PurchaseOrder()
    comp = po.field_completeness()
    assert not comp["complete"] and len(comp["missing"]) == 13
    po.set("prices", [{"kind": "price", "direction": "d1", "text": "price first"}, {"kind": "gate", "direction": "d1", "text": "gate after"}])
    po.set("header", {"dataset": "t", "target": "Y", "n": 100, "P": 10, "pihat": 0.5, "khat": 0, "regime_n_gg_P": "PASS", "regime_P4": "declared", "provable": None})
    po.set("promise", {"M0": 0.1, "rho": 0.1, "theta": 1.0, "delta": 0.1, "q": 0.1, "economics": "P-8.5 bridge"})
    po.set("estimand", "E2-realized")
    po.set("feasible_dark", {"type": "tolerance", "J": 1, "share_target": 0.9, "share_captured": 0.95, "gammahat": 1.0, "eta": 0.3, "dk_sin_bound": None, "eig_intervals": [(1.0, 1.6)], "theta_star": 5.0, "normD2_deb": 0.4, "band_norm": 0.1, "nu_deb": 0.3, "band_nu": 0.1})
    po.set("spec_table", []); po.set("stack_line", "s"); po.set("label_line", [])
    po.set("candidate_scores", []); po.set("never_ceiling", {"auc": 0.7, "acc": 0.65, "caveat": "in-host", "bridge": "acc - 1/2 = 1 - c"})
    po.set("no_go", []); po.set("seam_citations", ["AC7"]); po.set("self_audit", {"phase": "pre", "forecast": None, "realized": None, "band": None, "verdict": "NO_PURCHASE"})
    assert not po.gate_before_dial_ok()
    po.set("prices", [{"kind": "gate", "direction": "d1", "text": "gate first"}, {"kind": "price", "direction": "d1", "text": "price after"}])
    assert po.gate_before_dial_ok()
    import json
    json.loads(po.render_json())
    assert set(["AC7"]).issubset(set(SEAM_ENUM))
