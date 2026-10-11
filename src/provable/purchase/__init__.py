"""The purchase layer: price what data would move a result, in the data
currency, quoted before any purchase.

Two legs. The rows leg (purchase_order(rep)) prices the certifying rows
that would hold each certified atom at the claims bar; the weakest atom's
requirement is the report's N3_hat. The columns leg (purchase_order(rep,
promise=...)) prices the feature a dataset lacks: given a declared promise
(a target dark mass M0 to make certifiable at localization tolerance rho),
it locates the feasible-dark subspace, quotes the reliability a new column
must carry (guaranteed and reliability-law), the column count r_min at a
declared amplitude budget, the rows-route exchange, and guards the promise
against the reachability ceiling (the terminal-remainder guard, PG Thm 7.4).
The columns leg runs the folded acquisition engine (AC-series) on the
report's own panel; it issues nothing and charges nothing. Every number is
est, plug-in.
"""
from dataclasses import dataclass, field
from typing import Optional, Mapping, Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from ._engine import SieveState, Localizer, Acquisition, Envelope, LabelChannel, Currency, AdmissibilityGate, OrderContract, SEAM_ENUM


class _NoColumnsState:
    """po.state before a columns leg is priced. Its nu is the estimated-Gram
    corrected debiased visibility, available only once a promise is priced."""

    @property
    def nu(self):
        raise AttributeError("po.state.nu is the columns leg's estimated-Gram corrected visibility; pass promise= to purchase_order to price it")


@dataclass
class PurchasePromise:
    """The buyer's declared target for the columns leg. M0 = dark mass to make
    certifiable (M_Y scale); rho = localization tolerance (capture >= 1-rho of
    the debiased contrast energy); delta = unlock confidence; kappa, B = the
    sparse amplitude budget (at most kappa directions, per-direction amplitude
    cost capped at B^2) used to price r_min and the walls, optional; estimand =
    recorded buyer's choice; q defaults to the report's q."""
    M0: float
    rho: float = 0.10
    delta: float = 0.10
    kappa: Optional[int] = None
    B: Optional[float] = None
    theta: Optional[float] = None
    estimand: str = "E2-realized"


@dataclass
class ColumnsLeg:
    """The priced columns leg. reachability_ceiling = pihat(1-pihat)*nu_deb is
    the total mass the augmented panel could make legible; terminal_remainder
    fires when M0 exceeds it (PG Thm 7.4: the excess stays dark under any
    admissible acquisition). Pricing fields are quoted only when the promise is
    reachable."""
    promise: PurchasePromise
    refusal: Optional[str]
    feasible_dark: Optional[dict]
    reachability_ceiling: float
    terminal_remainder: dict
    guaranteed_reliability: Optional[dict]
    reliability_law_lam2: Optional[float]
    r_min: Optional[dict]
    rows_route: Optional[int]
    attenuation: Optional[dict]
    ceilings: Optional[dict]

    def render(self) -> str:
        p = self.promise
        L = [f"-- purchase order (columns leg; est, plug-in; promise M0={p.M0:g}, rho={p.rho:g}, delta={p.delta:g}, estimand={p.estimand}) --"]
        if self.refusal:
            L.append(f"columns leg REFUSED: {self.refusal}")
            return "\n".join(L)
        d = self.feasible_dark
        L.append(tabulate([[d["type"], d["J"], f">={d['share_target']:g}", round(d["share_captured"], 4), round(d["gammahat"], 4), round(d["eta"], 4), None if d["dk_sin_bound"] is None else round(d["dk_sin_bound"], 4)]], headers=["feasible-dark", "J", "share", "captured", "gammahat", "eta", "DK sin"], tablefmt="psql"))
        L.append(f"debiased visibility: nu_deb={d['nu_deb']:.6g} (+/-{d['band_nu']:.4g}); ||Deps||^2_deb={d['normD2_deb']:.6g} (+/-{d['band_norm']:.4g})")
        tr = self.terminal_remainder
        L.append(f"reachability ceiling pihat(1-pihat)nu_deb = {self.reachability_ceiling:.6g}; M0 = {p.M0:g}")
        if tr["fires"]:
            L.append(f"TERMINAL-REMAINDER GUARD FIRED (PG Thm 7.4): M0 exceeds the ceiling by {tr['remainder']:.6g}; that excess is terminally dark, no admissible column resolves it. Acquisition NOT priced.")
            return "\n".join(L)
        gr = self.guaranteed_reliability
        rows = [["guaranteed reliability lam_A (unlock at 1-delta)", f"lam^2={gr['lam2']:.4g}", "feasible" if gr["feasible"] else "INFEASIBLE_LAMBDA (lam^2>1)", None if gr["lam"] is None else f"lam={gr['lam']:.4g}"],
                ["reliability-law lam* (fixed hbar, M* = reachability ceiling)", f"lam*^2={self.reliability_law_lam2:.4g}", "", None if self.reliability_law_lam2 > 1 else f"lam*={np.sqrt(self.reliability_law_lam2):.4g}"]]
        L.append(tabulate(rows, headers=["reliability price", "value", "feasibility", "lam"], tablefmt="psql"))
        if self.r_min is not None:
            rm = self.r_min["r_min"]
            L.append(f"column count r_min at kappa={p.kappa}, B={p.B}: " + ("NONE within the support (no admissible count clears M0 at this budget)" if rm is None else f"{rm} columns"))
        else:
            L.append("column count r_min: not priced (declare kappa and B in the promise to price it)")
        if self.rows_route is not None:
            L.append(f"rows-route (columns<->rows exchange): {self.rows_route:,} rows would carry the same feasible visibility at rho={p.rho:g}, delta={p.delta:g}")
        if self.ceilings is not None:
            L.append(f"never-ceiling (plug-in from nu_deb): AUC<={self.ceilings['auc']:.4f}, accuracy<={self.ceilings['acc']:.4f} (in-panel estimate)")
        return "\n".join(L)

    def __str__(self):
        return self.render()


@dataclass
class PurchaseOrder:
    """A priced data order. Rows leg: per certified atom, the certifying-row
    count at which its |t| would sit at the claims bar (rows_for_level =
    zbar_core^2 * n2 / t^2, the engine's phase-III plug-in; the weakest atom's
    value is the report's N3_hat), and the order = the additional rows against
    N_train. Columns leg (present when a promise was passed): the priced feature
    the panel lacks, with po.state.nu = the estimated-Gram corrected visibility."""
    kind: str
    n_train: int
    n_s2: int
    claims_bar: float
    q: float
    rows: pd.DataFrame
    weakest: dict
    requirement: float
    additional_rows: int
    columns: Optional[ColumnsLeg] = None
    state: object = field(default_factory=_NoColumnsState)

    def render(self) -> str:
        head = f"-- purchase order (rows leg; est, plug-in at the claims bar {self.claims_bar:.4g}, q={self.q:g}) --"
        tab = tabulate([[r.atom, f"{r.t:.2f}", f"{r.rows_for_level:.4g}", f"{r.additional:.4g}"] for r in self.rows.itertuples()], headers=["atom", "|t|", "rows_for_level", "additional vs N_train"], tablefmt="psql")
        lines = [head, tab,
                 f"weakest certified: {self.weakest['atom']} at |t| {self.weakest['t']:.2f}; requirement {self.requirement:.4g} rows vs N_train = {self.n_train} (order: {self.additional_rows:,} additional rows)",
                 "requirement currency: certifying-row scale (engine phase-III plug-in), compared against N_train per the engine convention"]
        if self.columns is not None:
            lines += ["", self.columns.render()]
        else:
            lines.append("columns leg: pass promise= to purchase_order to price the feature the panel lacks")
        return "\n".join(lines)

    def __str__(self):
        return self.render()


def _acc_ceiling(M_F, pi):
    m = np.sqrt(max(M_F, 1e-12)); c = np.log((1 - pi) / pi)
    from scipy.stats import norm
    err = pi * norm.cdf((c - m * m / 2) / m) + (1 - pi) * norm.cdf((-c - m * m / 2) / m)
    return float(1 - err)


def _auc_ceiling(M_F):
    from scipy.stats import norm
    return float(norm.cdf(np.sqrt(max(M_F, 0.0) / 2.0)))


def _columns_leg(cd, promise: PurchasePromise) -> ColumnsLeg:
    y = np.asarray(cd.y, float).ravel()
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("columns leg requires a binary target (the acquisition engine is binary-Y by construction); the certified target is not 0/1. Binarizing it is a registration decision, not one the purchase layer makes.")
    W = pd.get_dummies(cd.X_, dtype=float).to_numpy()
    st = SieveState(q=float(cd.fdr_q)).fit(y.astype(int), W, run_provable=False)
    if st.refusal:
        return ColumnsLeg(promise=promise, refusal=st.refusal, feasible_dark=None, reachability_ceiling=float("nan"), terminal_remainder={"fires": False, "remainder": float("nan"), "M0": promise.M0, "reachability_ceiling": float("nan")}, guaranteed_reliability=None, reliability_law_lam2=None, r_min=None, rows_route=None, attenuation=None, ceilings=None)
    loc = Localizer(st).run(M0=promise.M0, rho=promise.rho)
    ceiling = float(st.pihat * (1 - st.pihat) * max(st.nu_deb, 0.0))
    remainder = max(promise.M0 - ceiling, 0.0)
    fires = promise.M0 > ceiling
    tr = {"fires": bool(fires), "remainder": float(remainder), "M0": float(promise.M0), "reachability_ceiling": ceiling}
    fd = {k: loc[k] for k in ("type", "J", "share_target", "share_captured", "gammahat", "eta", "dk_sin_bound", "eig_intervals", "theta_star", "normD2_deb", "band_norm", "nu_deb", "band_nu")}
    if fires:
        return ColumnsLeg(promise=promise, refusal=None, feasible_dark=fd, reachability_ceiling=ceiling, terminal_remainder=tr, guaranteed_reliability=None, reliability_law_lam2=None, r_min=None, rows_route=None, attenuation=None, ceilings=None)
    acq = Acquisition(st, Envelope(st.hbar, st.one_minus_d2))
    gr = acq.lambda_A(M0=promise.M0, rho=promise.rho, delta=promise.delta)
    rl = acq.reliability_law(M0=promise.M0, M_star=ceiling)
    n_req = acq.n_req_rows(st.nu_deb, promise.rho, promise.delta)
    atten = acq.attenuation(gr["lam"]) if gr["lam"] is not None else None
    r_min = None
    if promise.kappa is not None and promise.B is not None:
        support = st.evals
        r_min = acq.r_min(support_eigs=support, s=len(support), kappa=int(promise.kappa), B=float(promise.B), M0=promise.M0)
    ceilings = {"auc": _auc_ceiling(st.MhatF), "acc": _acc_ceiling(st.MhatF, st.pihat)}
    return ColumnsLeg(promise=promise, refusal=None, feasible_dark=fd, reachability_ceiling=ceiling, terminal_remainder=tr, guaranteed_reliability=gr, reliability_law_lam2=float(rl), r_min=r_min, rows_route=int(n_req), attenuation=atten, ceilings=ceilings)


def purchase_order(rep, promise=None) -> PurchaseOrder:
    """Price the rows that would hold every certified atom at the claims bar,
    and, when a promise is passed, the feature the panel lacks. Consumes a
    certify Report; issues nothing and charges nothing. promise is a
    PurchasePromise (or a mapping with at least M0)."""
    cd = rep.cd
    d = cd.diag_
    if not len(cd.certified_):
        raise ValueError("purchase_order: nothing certified; there is no level to hold (khat=0)")
    zbar = float(d["zbar_core"])
    n2 = len(cd.idx_s2)
    ntr = len(cd.idx_train)
    cl = rep.claims[["atom", "t"]].copy()
    cl["t"] = cl["t"].abs()
    cl["rows_for_level"] = zbar ** 2 * n2 / cl["t"] ** 2
    cl["additional"] = np.maximum(0.0, cl["rows_for_level"] - ntr)
    cl = cl.sort_values("rows_for_level", ascending=False).reset_index(drop=True)
    w = cl.iloc[0]
    req = float(w["rows_for_level"])
    assert np.isclose(req, float(d["N3_hat"]), rtol=1e-9), "rows leg drifted from the engine's N3_hat"
    po = PurchaseOrder(kind="rows", n_train=ntr, n_s2=n2, claims_bar=zbar, q=float(cd.fdr_q), rows=cl, weakest={"atom": str(w["atom"]), "t": float(w["t"])}, requirement=req, additional_rows=int(round(max(0.0, req - ntr))))
    if promise is not None:
        if not isinstance(promise, PurchasePromise):
            promise = PurchasePromise(**dict(promise))
        cols = _columns_leg(cd, promise)
        po.kind = "rows+columns"
        po.columns = cols
        po.state = _ColumnsState(cols)
    return po


class _ColumnsState:
    """po.state once a columns leg is priced. nu is the estimated-Gram
    corrected debiased visibility of record for the panel (SieveState.nu_deb)."""

    def __init__(self, cols: ColumnsLeg):
        self._cols = cols

    @property
    def nu(self):
        if self._cols.feasible_dark is None:
            raise AttributeError("po.state.nu unavailable: the columns leg refused (rider regime)")
        return float(self._cols.feasible_dark["nu_deb"])


def purchase_refit(certificate, X, y, expect_manifest, *, q: float = 0.10, seed: int = 0, dataset_tag: str = "panel", **certify_kwargs):
    """PD-1 option (a): re-derive the fitted state for a real-data purchase
    WITHOUT touching the book. Runs certify with ledger=None (no register op,
    no charge) and a hard manifest match against the accepted baseline; raises
    ManifestMismatch on any drift. When certificate is supplied, also asserts
    the refit reproduces its certified support, so the purchase prices the
    accepted result and not a silent re-fit. Returns the ledger-free Report."""
    from ..facade import certify
    rep = certify(X, y, q=q, ledger=None, seed=seed, dataset_tag=dataset_tag, expect_manifest=expect_manifest, **certify_kwargs)
    # ledger=None issues no certificate, so cross-check the refit's certified
    # atoms (rep.claims) against the accepted baseline certificate's support.
    if certificate is not None:
        base = certificate.guarantee.family.get("support_ordered")
        got = [] if rep.claims is None else list(rep.claims["atom"])
        if base is not None:
            if len(base) != len(got) or set(map(str, base)) != set(map(str, got)):
                raise ValueError("purchase_refit: ledger-free refit did not reproduce the baseline certificate's certified support; the manifest matched but the certified set differs (investigate before pricing)")
    return rep
