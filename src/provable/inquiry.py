"""The inquiry layer: play the Twenty Questions charter on a certified
target's archive shadow (Reading a Missing Page, Theorem 9.5 and Table 4).

Three verbs, all free of acquisition, all descriptive: they issue nothing,
charge nothing, and write nothing to any book.

  shadow(rep, X, y, clock)        -> Shadow      the certified target's owned
                                                 residual on a declared
                                                 out-of-fit slice
  characterize(sh, ...)           -> Characterization   basis interrogation
                                                 (Thm 7.3), the lag ledger
                                                 (Thm 8.3, 8.5, 8.6(iv)), the
                                                 charter filing (Table 4)
  screen(sh, candidates, ...)     -> Screening   the screen of Section 4
                                                 (Defs 4.3-4.4, Thm 4.9)

Terminology, kept explicit in the C-track pages: the per-row e this layer
stores is the paper's owned residual u = Y - f_D, whose empirical second
moment estimates the answerable aggregate Var(u) = Mhat + Mbar of
Theorem 2.4(iii); the atom profiles computed from it estimate the archive
shadow's profiles, with the hypothesis-free bands of Proposition 2.8 on
deterministic frame atoms. Every printed number carries a grade (exact /
est / plug-in) and the statement it comes from.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from tabulate import tabulate

from .certificates.lineage import source_identity

XI_MAX = float(np.sqrt(2.0) * np.arcsin(1.0 / np.sqrt(2.0)))


# ---------------------------------------------------------------- scales

def xi(r):
    """Stabilized coupling scale (Lem 4.8): xi(r) = sqrt2 asin(r/sqrt2)."""
    r = np.clip(np.asarray(r, float), 0.0, 1.0)
    return np.sqrt(2.0) * np.arcsin(r / np.sqrt(2.0))


def xi_inv(x):
    x = np.clip(np.asarray(x, float), 0.0, XI_MAX)
    return np.sqrt(2.0) * np.sin(x / np.sqrt(2.0))


def _a(r):
    return r * r * (2.0 - r * r)


def _C(r, J):
    return 2.0 * r * r * (2.0 - r * r) + 2.0 * (1.0 - r * r) ** 2 + 2.0 * (J - 1)


def n_z(rho, z, Delta, J):
    """One-sided requirement of Theorem 4.9(a)."""
    a = _a(rho)
    return (z * z * a + z * np.sqrt(z * z * a * a + Delta * Delta * _C(rho, J))) / (Delta * Delta)


def m_req(lam_min, dlam, J=1, eps_miss=0.10, eps_clear=0.10):
    """The operating requirement of Definition 4.4, achieved by the
    threshold test at t* = lam_min^2 + dlam^2 (Theorem 4.9(a); the band
    rule inherits it to first order, Theorem 4.9(f)). Returns (m, corner);
    corner=True when lam_min <= dlam (Theorem 4.9(e): the clear side
    degenerates to exact chi-square null testing)."""
    if lam_min <= 0:
        return float("inf"), True
    Delta = 2.0 * lam_min * dlam
    zm, zc = norm.ppf(1 - eps_miss), norm.ppf(1 - eps_clear)
    rp, rm = min(lam_min + dlam, 0.999999), max(lam_min - dlam, 0.0)
    m = 1.0 + max(n_z(rp, zm, Delta, J), n_z(rm, zc, Delta, J))
    return float(m), bool(lam_min <= dlam)


# ------------------------------------------------------------ design effect

def cluster_ids(clock, cluster: str = "day") -> np.ndarray:
    c = pd.to_datetime(pd.Series(clock)).dt.floor("D" if cluster == "day" else cluster)
    return pd.factorize(c)[0]


def deff_cluster(psi, cid) -> float:
    """Design effect of the mean of psi under the within-cluster-uniform,
    across-cluster-zero kernel: the cluster special case of the dependence
    factor of Theorem 8.5(iii). Ratio of the clustered to the iid variance
    of the mean; plug-in."""
    x = np.asarray(psi, float) - float(np.mean(psi))
    denom = float(np.sum(x * x))
    if denom <= 0:
        return 1.0
    s = pd.Series(x).groupby(np.asarray(cid)).sum().to_numpy()
    return float(np.sum(s * s) / denom)


# ---------------------------------------------------------------- algebra

def _span_of(rows: pd.DataFrame) -> np.ndarray:
    """The empirical licensed span this layer residualizes against
    (Def 7.1): the intercept and, when the record carries it, the shadow's
    own p_hat. Frame members enter as their own residuals, so their span
    is the intercept."""
    n = len(rows)
    if "p_hat" in rows.columns:
        return np.column_stack([np.ones(n), rows["p_hat"].to_numpy()])
    return np.ones((n, 1))


def _residualize(M: np.ndarray, span: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(span)
    return M - Q @ (Q.T @ M)


def _complete(v, verb: str, what: str) -> np.ndarray:
    """Every verb that reads a column takes it complete. A NaN or an infinity
    would otherwise pass into a least-squares fit or a logistic fit and either
    stop it deep inside a library or, worse, come back out as a nan statistic;
    how a gap is filled is a registration decision made before the verb."""
    v = np.asarray(v, float)
    bad = ~np.isfinite(v)
    if bad.any():
        raise ValueError(f"{verb} refused: {what} carries {int(bad.sum())} missing or non-finite value(s) in {v.shape[0]} rows; the verbs take complete columns, and how a gap is filled (imputation, a missingness indicator, dropping the rows) is a registration decision made before the verb")
    return v


def _orthonormalize(M: np.ndarray, n: int):
    """Empirical-inner-product orthonormalization within a family
    (Def 7.1): columns q_b with n^-1 sum q_b^2 = 1, numerically null atoms
    dropped, and the conditioning norm of the change of basis quoted with
    any transformed band (Thm 7.3(ii))."""
    G = (M.T @ M) / n
    d, V = np.linalg.eigh(G)
    keep = d > 1e-10 * max(float(d.max()) if len(d) else 1.0, 1.0)
    if not keep.any():
        return np.zeros((n, 0)), 1.0, []
    d, V = d[keep], V[:, keep]
    Q = M @ (V / np.sqrt(d))
    for j in range(Q.shape[1]):
        k = int(np.argmax(np.abs(V[:, j])))
        if V[k, j] < 0:
            Q[:, j] = -Q[:, j]
    cond = float(np.sqrt(d.max() / d.min()))
    return Q, cond, list(np.where(keep)[0])


def _orth_slice(W: np.ndarray) -> np.ndarray:
    """Orthonormalize the aim-block columns in the empirical inner product
    of one slice, sign-aligned to the incoming columns so the J = 1 sign
    gate reads the same orientation on both slices."""
    n = len(W)
    Wc = W - W.mean(axis=0)
    Q, _ = np.linalg.qr(Wc)
    Q = Q * np.sqrt(n)
    for j in range(Q.shape[1]):
        if float(Q[:, j] @ W[:, j]) < 0:
            Q[:, j] = -Q[:, j]
    return Q


def _fourier(n: int, period_steps: float) -> Tuple[np.ndarray, List[str]]:
    t = np.arange(n)
    cols, lab = [], []
    for nm, v in (("sin", np.sin(2 * np.pi * t / period_steps)), ("cos", np.cos(2 * np.pi * t / period_steps))):
        r = float(np.sqrt(np.mean(v * v)))
        if r > 1e-9:
            cols.append(v / r)
            lab.append(nm)
    return (np.column_stack(cols) if cols else np.zeros((n, 0))), lab


def _period_steps(period: str, step: pd.Timedelta) -> float:
    named = {"day": pd.Timedelta("1D"), "week": pd.Timedelta("7D")}
    length = named[period] if period in named else pd.Timedelta(period)
    return max(length / step, 2.0)


def _hf_radius(K: int, level: float, n: float) -> float:
    """Hypothesis-free archive radius, uniform over K atoms: Prop 2.8 with
    a union bound (Thm 7.3(ii)). Exact finite-sample, no plug-in."""
    return float(np.sqrt(2.0 * np.log(2.0 * max(K, 1) / level) / n))


# ================================================================= Shadow

@dataclass
class Shadow:
    """The archive-shadow record of one certified target on a declared
    out-of-fit slice: per-row (clock, y, p_hat, e), e = y - f_D(x) the
    owned residual of the Report's own certified-class predictor; the fit
    manifest consumed; the slice era (Ass 5.2: an estimation warranty,
    never an identification clause); the mean shift; ||e||^2_n (est of the
    answerable aggregate, Thm 2.4(iii)); the DEFF of e. The slice's state
    design in the Report's own encoding rides along hashed and unprinted,
    for the aim block of a screen."""

    rows: pd.DataFrame
    manifest: dict
    dataset_tag: str
    estimand: str
    era: Tuple[pd.Timestamp, pd.Timestamp]
    khat: int
    base_rate: Optional[float]
    var: float
    mean_shift: float
    deff: float
    m: int
    m_eff: float
    cluster: str
    clock_step: pd.Timedelta
    tag: str = "unit"
    members: Optional[List["Shadow"]] = None
    _design: Optional[np.ndarray] = field(default=None, repr=False)
    _design_cols: Optional[List[str]] = field(default=None, repr=False)
    _rep: Any = field(default=None, repr=False)

    @property
    def is_frame(self) -> bool:
        return self.members is not None

    @staticmethod
    def join(shadows: Sequence["Shadow"], agg: str = "mean") -> "Shadow":
        """The multi-unit frame: the shadows of several targets aligned on
        the clock key, aggregated to the coarsest declared clock (mean),
        common rows only; every member's manifest and the frame clause
        (shared clock, overlapping eras) that the cross-sectional row of
        the charter requires (Cor 7.7)."""
        if agg != "mean":
            raise ValueError("join: only agg='mean' is defined (the coarsest-clock mean)")
        if len(shadows) < 2:
            raise ValueError("join: a frame needs at least two members")
        for s in shadows:
            if s.is_frame:
                raise ValueError(f"join refused: member '{s.tag}' is already a frame; join units only")
        steps = [s.clock_step for s in shadows]
        coarse = max(steps)
        bad = [s.tag for s, st in zip(shadows, steps) if (coarse.value % st.value) != 0]
        if bad:
            raise ValueError("join refused: mixed clocks; the step of member(s) " + ", ".join(bad) + " does not nest into the coarsest declared clock")
        eras = [s.era for s in shadows]
        if max(t0 for t0, _ in eras) > min(t1 for _, t1 in eras):
            raise ValueError("join refused: member eras do not overlap; the frame clause of Cor 7.7 requires overlapping eras on a shared clock")
        cols = {}
        for i, s in enumerate(shadows):
            g = s.rows.assign(_c=pd.to_datetime(s.rows["clock"]).dt.floor(coarse)).groupby("_c")["e"].mean()
            cols[f"e{i}"] = g
        aligned = pd.DataFrame(cols).dropna(axis=0, how="any")
        if not len(aligned):
            raise ValueError("join refused: no common rows on the coarsest clock")
        rows = aligned.reset_index().rename(columns={"_c": "clock"})
        E = rows[[f"e{i}" for i in range(len(shadows))]].to_numpy()
        cid = cluster_ids(rows["clock"], shadows[0].cluster)
        d = float(np.mean([deff_cluster(E[:, j], cid) for j in range(E.shape[1])]))
        m = len(rows)
        return Shadow(rows=rows, manifest={"members": [dict(s.manifest) for s in shadows]},
                      dataset_tag=shadows[0].dataset_tag, estimand=shadows[0].estimand,
                      era=(rows["clock"].min(), rows["clock"].max()), khat=-1, base_rate=None,
                      var=float(np.mean(E * E)), mean_shift=float(np.mean(E)), deff=d, m=m,
                      m_eff=m / max(d, 1.0), cluster=shadows[0].cluster, clock_step=coarse,
                      tag="frame", members=list(shadows))

    def render(self) -> str:
        L = [f"-- Shadow ({('frame, %d members' % len(self.members)) if self.is_frame else 'unit ' + self.tag}; estimand {self.estimand}; free of acquisition, nothing issued or charged) --"]
        L.append(f"rows m = {self.m}; m_eff = {self.m_eff:.1f} (DEFF {self.deff:.3f}, {self.cluster}-cluster kernel; plug-in, Thm 8.5(iii)); clock step {self.clock_step}")
        L.append(f"era {self.era[0]} .. {self.era[1]}  [Ass 5.2: an estimation warranty, never an identification clause]")
        if self.is_frame:
            L.append(f"common rows on the coarsest clock: {self.m}; every member manifest carried")
        else:
            man = {k: self.manifest.get(k) for k in ("x_sha", "y_sha", "dictionary_sha", "config_sha", "seed")}
            L.append("fit manifest consumed: " + ", ".join(f"{k}={v}" for k, v in man.items()))
            if self.khat == 0:
                L.append(f"khat = 0: the certified class is empty; the predictor is the training base rate {self.base_rate:.4f} and every number below reads against it")
        L.append(f"mean shift = {self.mean_shift:+.6g} [exact on the slice; ranked rules are immune to it, any p > 0.5 rule is not]")
        L.append(f"shadow second moment ||e||^2_n = {self.var:.6g} [est of the answerable aggregate Var(u) = Mhat + Mbar, Thm 2.4(iii)]")
        return "\n".join(L)

    def __str__(self):
        return self.render()


def _cert_slices(rep):
    out = []
    cert = getattr(rep, "certificate", None)
    if cert is None:
        return out
    for lin in (getattr(cert, "ell_test", None), getattr(cert, "proposer_rows", None)):
        if lin is None:
            continue
        sls = getattr(lin, "slices", None)
        if sls is None:
            sls = [lin] if hasattr(lin, "t0") else []
        out.extend(list(sls) if isinstance(sls, (list, tuple, set, frozenset)) else [sls])
    return out


def _design_for(rep, X):
    ref = pd.get_dummies(rep.cd.X_, dtype=float)
    new = pd.get_dummies(pd.DataFrame(X), dtype=float).reindex(columns=ref.columns, fill_value=0.0)
    return new.to_numpy(), list(ref.columns)


def shadow(rep, X, y, clock, estimand: str = "E2-realized", cluster: str = "day", dataset_tag: Optional[str] = None) -> Shadow:
    """The archive shadow of the Report's certified target on a declared
    slice: e_t = y_t - f_D(x_t), f_D the Report's own certified-class
    predictor through its stored dictionary, scored on rows disjoint from
    every row the Report was fitted on. Disjointness is asserted by row
    identity (source fingerprint; and the clock envelope of the Report's
    lineage under the same dataset tag); overlap refuses by name. Runs on
    a certify Report or on the ledger-free Report of purchase_refit. When
    khat = 0 the predictor is the training base rate and the Shadow says
    so."""
    if estimand != "E2-realized":
        raise ValueError(f"shadow: estimand '{estimand}' is not defined; only E2-realized is")
    yv = np.asarray(y, float).ravel()
    if not set(np.unique(yv)) <= {0.0, 1.0}:
        raise ValueError("shadow: this layer's certified target is binary; the slice target is not 0/1. Binarizing it is a registration decision, not one the inquiry layer makes.")
    Xf = pd.DataFrame(X).reset_index(drop=True)
    ck = pd.to_datetime(pd.Series(clock)).reset_index(drop=True)
    if len(Xf) != len(yv) or len(ck) != len(yv):
        raise ValueError("shadow: X, y and clock must align row for row; a clock key is mandatory")
    tag = dataset_tag if dataset_tag is not None else rep.dataset_tag
    fp, fp_aliases = source_identity(Xf)
    if rep.manifest.get("x_sha") in fp_aliases:
        raise ValueError("shadow refused: overlap by row identity; the slice source fingerprint equals the Report's x_sha, so these are the fitted rows, not an out-of-fit slice")
    t0, t1 = ck.min(), ck.max()
    for sl in _cert_slices(rep):
        try:
            same_tag = sl.dataset_tag == tag
            inter = (pd.Timestamp(sl.t0) <= t1) and (pd.Timestamp(sl.t1) >= t0)
        except Exception:
            continue
        if same_tag and inter:
            raise ValueError(f"shadow refused: overlap by row identity; the slice clock window [{t0} .. {t1}] intersects the Report's fitted lineage envelope [{sl.t0} .. {sl.t1}] under the same dataset tag '{tag}' (the conservative clock taint of the row identity)")
    khat = int(getattr(rep.cd, "khat_", 0))
    if khat == 0:
        base = float(np.mean(np.asarray(rep.cd.y, float)))
        p_hat = np.full(len(yv), base)
    else:
        base = None
        p_hat = np.clip(np.asarray(rep.core_fit(Xf), float).ravel(), 0.0, 1.0)
    e = yv - p_hat
    rows = pd.DataFrame({"clock": ck, "y": yv, "p_hat": p_hat, "e": e}).sort_values("clock").reset_index(drop=True)
    diffs = rows["clock"].diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    step = diffs.median() if len(diffs) else pd.Timedelta("1h")
    cid = cluster_ids(rows["clock"], cluster)
    W, wcols = _design_for(rep, Xf)
    man = {k: rep.manifest.get(k) for k in ("x_sha", "y_sha", "dictionary_sha", "config_sha", "seed")}
    man["slice_sha"] = fp
    man["state_design_sha"] = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:16]
    ev = rows["e"].to_numpy()
    d = deff_cluster(ev, cid)
    return Shadow(rows=rows, manifest=man, dataset_tag=tag, estimand=estimand,
                  era=(rows["clock"].iloc[0], rows["clock"].iloc[-1]), khat=khat, base_rate=base,
                  var=float(np.mean(ev * ev)), mean_shift=float(np.mean(ev)), deff=d, m=len(rows),
                  m_eff=len(rows) / max(d, 1.0), cluster=cluster, clock_step=step,
                  tag=str(tag), _design=W, _design_cols=wcols, _rep=rep)


# ============================================================ Characterization

@dataclass
class Characterization:
    """Basis interrogation (Thm 7.3), the lag ledger (Thm 8.3, 8.5,
    8.6(iv)) and the charter filing (Table 4). Descriptive, priced at zero
    on the certification ledger (Def 7.2). Carries no field for a
    candidate identity: the absolute rule of the charter (any identity
    test sits at exactly one half, Thm 3.2) is enforced at the type
    level. On a frame, one charter per member plus the shared frame
    block."""

    profiles: pd.DataFrame
    ledger: pd.DataFrame
    persistence: pd.DataFrame
    h_star: Optional[int]
    verdict_persistence: str
    gram: Optional[dict]
    scope_shares: Optional[dict]
    charter: List[dict]
    member_charters: Optional[List[List[dict]]]
    joint: dict
    lam_arch: dict
    level: float
    caveats: List[str]

    def _charter_tab(self, ch):
        rows = [[r["q"], r["status"], "" if r["value"] is None else r["value"], r["cite"]] for r in ch]
        return tabulate(rows, headers=["question", "status", "value and band", "law"], tablefmt="psql", maxcolwidths=[30, 24, 46, 18])

    def render(self) -> str:
        L = ["-- Characterization (interrogation is priced at zero on the certification ledger, Def 7.2) --"]
        led = [[r.member, r.family, r.K, f"{r.explained:.6g}", f"{r.share:.4f}", r.grade] for r in self.ledger.itertuples()]
        L.append(tabulate(led, headers=["member", "family", "K'", "explained mass", "share of ||e||^2_n", "band"], tablefmt="psql"))
        L.append(f"joint share over all families = {self.joint['share']:.4f} [exact ledger, Thm 7.3(i)]; remainder behind the wall = {self.joint['remainder_share']:.4f} of ||e||^2_n [Thm 7.3(iii); the external part of it is walled, Thm 2.11]")
        st = self.profiles[self.profiles["strongest"]]
        L.append(tabulate([[r.member, r.family, r.atom, f"{r.beta:+.4f}", f"+/-{r.band:.4f}", r.grade] for r in st.itertuples()],
                          headers=["member", "family", "strongest atom", "beta_hat", "band", "grade"], tablefmt="psql"))
        L.append("[a strongest-atom quote is a maximum over K selections; an honest widened quote adds sqrt(2 log K) noise scale, Thm 5.6(iv) / Thm 5.7]")
        L.append(tabulate([[int(r.lag), f"{r.rho:+.4f}", f"[{r.lo:+.4f}, {r.hi:+.4f}] (b={int(r.bw)})"] for r in self.persistence.itertuples()],
                          headers=["lag", "autocorr of e", "band (uniform over the set; long-run plug-in at the printed Bartlett bandwidth, Thm 8.5)"], tablefmt="psql"))
        hs = "beyond the declared set" if self.h_star is None else str(self.h_star)
        L.append(f"persistence verdict: {self.verdict_persistence} (half-life h* = {hs} by band inversion, Thm 8.6(iv))")
        if self.gram is not None:
            L.append(f"frame block: top residual-Gram eigenvalue {self.gram['top_eig']:.4f} vs independence reference {self.gram['reference']:.4f}; verdict {self.gram['verdict']} [Cor 7.7]; shares frame-wide {self.scope_shares['frame_wide']:.2f} / group {self.scope_shares['group']:.2f} / idiosyncratic {self.scope_shares['idiosyncratic']:.2f}")
        la = self.lam_arch
        L.append(f"archive aim ceiling lam_arch = {la['value']:.4f} (+/-{la['band']:.4f}) on the residual-variance scale [est, Def 4.14]" + (f"; {la['aim_value']:.4f} on the aim scale over nu_aim [est, Prop 4.2(ii)]" if la.get("aim_value") is not None else ""))
        L.append("")
        if self.member_charters is None:
            L.append("-- the Twenty Questions charter, filed (Table 4; Thm 9.5) --")
            L.append(self._charter_tab(self.charter))
        else:
            for i, ch in enumerate(self.member_charters):
                L.append(f"-- charter, member {i} (Table 4; Thm 9.5) --")
                L.append(self._charter_tab(ch))
        for c in self.caveats:
            L.append(c)
        return "\n".join(L)

    def __str__(self):
        return self.render()


def _family_block(name, M, labels, e_r, n, cid, level, deterministic, span):
    Mr = _residualize(np.asarray(M, float), span)
    Q, cond, kept = _orthonormalize(Mr, n)
    K = Q.shape[1]
    if K == 0:
        return None
    beta = (Q.T @ e_r) / n
    hf = _hf_radius(K, level, n) * cond if deterministic else None
    z = norm.ppf(1 - level / 2)
    rows = []
    strongest = int(np.argmax(np.abs(beta)))
    for j in range(K):
        pj = z * float(np.std(Q[:, j] * e_r)) * np.sqrt(max(deff_cluster(Q[:, j] * e_r, cid), 1.0)) / np.sqrt(n)
        rows.append(dict(family=name, atom=labels[kept[j]] if kept[j] < len(labels) else f"q{j}",
                         beta=float(beta[j]), band=float(hf if hf is not None else pj),
                         grade="exact" if deterministic else "plug-in",
                         exceeds=bool(abs(beta[j]) > (hf if hf is not None else pj)), strongest=(j == strongest)))
    grade = f"exact hypothesis-free (Prop 2.8; radius x conditioning {cond:.2f})" if deterministic else "plug-in (clustered, Thm 8.5(iii))"
    return dict(name=name, Q=Q, beta=beta, explained=float(np.sum(beta ** 2)), K=K, cond=cond, hf=hf,
                rows=rows, grade=grade, deterministic=deterministic)


def deff_bartlett(psi, bw: int) -> float:
    """Long-run-variance dependence factor of the influence sequence with
    the Bartlett kernel at the printed bandwidth (Thm 8.5(i) and (iii));
    plug-in. The within-cluster-uniform kernel of deff_cluster is the
    cluster special case."""
    x = np.asarray(psi, float) - float(np.mean(psi))
    n = len(x)
    c0 = float(np.mean(x * x))
    if c0 <= 0 or n < 3:
        return 1.0
    acc = 0.0
    for k in range(1, min(bw, n - 1) + 1):
        acc += (1.0 - k / (bw + 1.0)) * float(np.mean(x[:-k] * x[k:]))
    return max(1.0 + 2.0 * acc / c0, 1e-6)


def _persistence(e_r, cid, lags, level):
    n = len(e_r)
    v = float(np.mean(e_r ** 2))
    out = []
    z = norm.ppf(1 - level / (2 * max(len(lags), 1)))
    bw = max(int(np.ceil(4.0 * (n / 100.0) ** (2.0 / 9.0))), 1)
    for ell in sorted(set(int(l) for l in lags)):
        if not (0 < ell < n - 2):
            continue
        a, b = e_r[:-ell], e_r[ell:]
        rho = float(np.mean(a * b)) / max(v, 1e-300)
        psi = a * b
        d = max(deff_bartlett(psi, bw), deff_cluster(psi, np.asarray(cid)[ell:]), 1.0)
        sd = float(np.std(psi)) * np.sqrt(d) / np.sqrt(n - ell)
        half = z * sd / max(v, 1e-300)
        out.append(dict(lag=ell, rho=rho, lo=rho - half, hi=rho + half, bw=bw))
    df = pd.DataFrame(out, columns=["lag", "rho", "lo", "hi", "bw"])
    h_star = None
    if len(df):
        half_level = 0.5 * abs(float(df.iloc[0]["rho"]))
        for r in df.itertuples():
            if abs(r.rho) <= half_level:
                h_star = int(r.lag)
                break
        last = df.iloc[-1]
        verdict = "persistent" if (last["lo"] > 0 or last["hi"] < 0) else "fast-decaying"
    else:
        verdict = "fast-decaying"
    return df, h_star, verdict


def _curve_gap(rows, fF):
    if "y" not in rows.columns or fF is None or fF["K"] == 0:
        return None
    n = len(rows)
    e = rows["e"].to_numpy()
    y = rows["y"].to_numpy()

    def cum(v):
        b = (fF["Q"].T @ (v - v.mean())) / n
        msq = np.sort(b ** 2)[::-1]
        return np.cumsum(msq) / max(msq.sum(), 1e-300)

    return float(np.max(np.abs(cum(e) - cum(y))))


def _unit_pipeline(rows, var_e, cid, clock_step, atoms, level, lags, periods, holidays, member):
    n = len(rows)
    e = rows["e"].to_numpy()
    span = _span_of(rows)
    e_r = _residualize(e[:, None], span)[:, 0]
    fams, caveats = [], []
    Fs, Flab = [], []
    for p in periods:
        B, nm = _fourier(n, _period_steps(p, clock_step))
        Fs.append(B)
        Flab += [f"{p}:{x}" for x in nm]
    blk = _family_block("temporal (Fourier)", np.column_stack(Fs) if Fs else np.zeros((n, 0)), Flab, e_r, n, cid, level, True, span)
    if blk:
        fams.append(blk)
    Ls, Llab = [], []
    for ell in lags:
        if 0 < ell < n:
            Ls.append(np.concatenate([np.zeros(ell), e[:-ell]]))
            Llab.append(f"lag {ell}")
    if Ls:
        blk = _family_block("temporal (own lags)", np.column_stack(Ls), Llab, e_r, n, cid, level, False, span)
        if blk:
            fams.append(blk)
    wd = pd.get_dummies(pd.to_datetime(rows["clock"]).dt.day_of_week, dtype=float)
    Clab = [f"weekday {int(c)}" for c in wd.columns]
    wdM = wd.to_numpy()
    if len(holidays):
        hol = pd.to_datetime(rows["clock"]).dt.floor("D").isin(pd.to_datetime(pd.Series(list(holidays)))).astype(float).to_numpy()[:, None]
        wdM = np.column_stack([wdM, hol])
        Clab.append("declared holiday")
    blk = _family_block("calendar", wdM, Clab, e_r, n, cid, level, True, span)
    if blk:
        fams.append(blk)
    cross = atoms.get("cross_members")
    if cross is not None and cross.shape[1] > 0:
        blk = _family_block("cross-members", cross, [f"member {j}" for j in atoms.get("cross_labels", range(cross.shape[1]))], e_r, n, cid, level, False, span)
        if blk:
            fams.append(blk)
    arch = atoms.get("archive", {})
    if arch:
        A, Alab = [], []
        for name, spec in arch.items():
            if "lag_class" not in spec and "clock_offset" not in spec:
                raise ValueError(f"characterize refused: archive atom '{name}' carries no lag class or clock offset; post-decision quantities must declare theirs")
            A.append(np.asarray(spec["values"], float).ravel())
            Alab.append(str(name))
        blk = _family_block("archive", np.column_stack(A), Alab, e_r, n, cid, level, False, span)
        if blk:
            fams.append(blk)
    rotation = None
    stt = atoms.get("state", {})
    if stt and fams:
        strong = max(fams, key=lambda f: f["explained"])
        qcol = strong["Q"][:, int(np.argmax(np.abs(strong["beta"])))]
        name, spec = next(iter(stt.items()))
        v = np.asarray(spec["values"], float).ravel()
        qs = sorted(spec.get("quantiles", (0.25, 0.5, 0.75)))
        edges = np.quantile(v, [0.0] + list(qs) + [1.0])
        b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
        loads, vshare, rshare = [], [], []
        for k in range(len(edges) - 1):
            mk = b == k
            if mk.sum() < 5:
                loads.append(np.nan); vshare.append(0.0); rshare.append(float(mk.mean())); continue
            lo = float(np.mean(qcol[mk] * e_r[mk]) / max(np.mean(qcol[mk] ** 2), 1e-12))
            loads.append(lo); vshare.append(float(np.sum((qcol[mk] * lo) ** 2))); rshare.append(float(mk.mean()))
        vs, rs = np.asarray(vshare), np.asarray(rshare)
        order = np.argsort(-vs)
        topk = order[: max(1, len(vs) // 2)]
        conc = float(vs[topk].sum() / max(vs.sum(), 1e-12)) / max(rs[topk].sum(), 1e-12)
        signs = {np.sign(x) for x in loads if np.isfinite(x) and x != 0}
        rotation = dict(state=str(name), buckets=len(edges) - 1, loads=loads, sign_flip=bool(len(signs) > 1), concentration=conc)
        caveats.append("[rotation grid: the atom count multiplies across the product grid and the multiplicity radius scales with it, stated once here (Cor 7.8)]")
    led, joint_explained = [], 0.0
    for f in fams:
        proj = f["Q"] @ ((f["Q"].T @ e_r) / n)
        drop = float(np.mean(e_r ** 2)) - float(np.mean((e_r - proj) ** 2))
        chk = abs(f["explained"] - drop)
        assert chk <= 1e-12 * max(1.0, var_e), f"exact ledger identity failed for family {f['name']}: |sum beta^2 - fitted drop| = {chk:.3e}"
        joint_explained += f["explained"]
        led.append(dict(member=member, family=f["name"], K=f["K"], explained=f["explained"], share=f["explained"] / max(var_e, 1e-300), grade=f["grade"]))
    prof = pd.DataFrame([dict(member=member, **r) for f in fams for r in f["rows"]]) if fams else pd.DataFrame(columns=["member", "family", "atom", "beta", "band", "grade", "exceeds", "strongest"])
    pers, h_star, verdict = _persistence(e_r, cid, lags, level)
    joint = dict(share=joint_explained / max(var_e, 1e-300), remainder_share=1.0 - joint_explained / max(var_e, 1e-300))
    return dict(fams=fams, ledger=led, profiles=prof, persistence=pers, h_star=h_star, verdict=verdict,
                joint=joint, rotation=rotation, caveats=caveats, e_r=e_r, span=span)


def _file_charter(pipe, rows, var_e, la, frame_block):
    fF = next((f for f in pipe["fams"] if f["name"].startswith("temporal (Fourier)")), None)
    fL = next((f for f in pipe["fams"] if "own lags" in f["name"]), None)
    ch = []
    if fF is not None and fF["K"]:
        j = int(np.argmax(np.abs(fF["beta"])))
        v1 = f"strongest line {fF['rows'][j]['atom']}, beta {fF['beta'][j]:+.4f} +/- {fF['hf']:.4f} [exact]"
        gap = _curve_gap(rows, fF)
        v2 = None if gap is None else f"residual band-mass curve vs the outcome's own curve on the declared grid: sup gap {gap:.3f} [est]"
    else:
        v1, v2 = "no nondegenerate line on the declared grid", None
    ch.append(dict(q="Which temporal band does the shadow load", status="answerable", value=v1, cite="Cor 7.6"))
    ch.append(dict(q="Does it vary as often as the outcome", status="answerable", value=v2 if v2 is not None else "two identified curves compared; outcome column not carried on this frame", cite="Cor 7.6"))
    if frame_block is not None:
        g, sc = frame_block
        v3 = f"verdict {g['verdict']}; shares frame-wide {sc['frame_wide']:.2f} / group {sc['group']:.2f} / idiosyncratic {sc['idiosyncratic']:.2f}; top eig {g['top_eig']:.3f} vs ref {g['reference']:.3f} [plug-in]"
        ch.append(dict(q="Frame-wide, group, or idiosyncratic", status="answerable (multi-unit frame)", value=v3, cite="Cor 7.7"))
    else:
        ch.append(dict(q="Frame-wide, group, or idiosyncratic", status="not applicable (single-unit shadow; the frame clause)", value=None, cite="Cor 7.7"))
    rot = pipe["rotation"]
    if rot is not None:
        ch.append(dict(q="Does its factor loading rotate", status="answerable", value=f"loading over {rot['buckets']} declared buckets of one state column: sign flip {rot['sign_flip']}; concentration {rot['concentration']:.2f} [est, grid-priced]", cite="Cor 7.8"))
    else:
        ch.append(dict(q="Does its factor loading rotate", status="answerable when a state column is declared; none this run", value=None, cite="Cor 7.8"))
    pers, h_star, verdict = pipe["persistence"], pipe["h_star"], pipe["verdict"]
    if len(pers):
        r0 = pers.iloc[0]
        v5 = f"{verdict}; autocorr at lag {int(r0['lag'])} = {r0['rho']:+.3f} in [{r0['lo']:+.3f}, {r0['hi']:+.3f}]; h* = {h_star if h_star is not None else 'beyond the set'} [plug-in, uniform over the set]"
    else:
        v5 = "lag set empty on this slice"
    ch.append(dict(q="Fast-decaying or persistent", status="answerable exactly for the shadow; to the envelope for the score", value=v5, cite="Thm 8.6(iv)"))
    v6 = (f"own-lag family explained share {fL['explained'] / max(var_e, 1e-300):.4f} at the archive tier [plug-in]; the world tier stays a hypothesis" if fL is not None else "no lag inside the slice")
    ch.append(dict(q="Does past surprise feed it", status="answerable at the archive tier; hypothesis at the world tier", value=v6, cite="Thm 8.3, Rem 8.4"))
    v7 = f"lam_arch = {la['value']:.4f} (+/-{la['band']:.4f}) on the residual-variance scale" + (f"; {la['aim_value']:.4f} on the aim scale over nu_aim" if la.get("aim_value") is not None else "") + " [est]"
    ch.append(dict(q="How much of it is buildable from the archive", status="answerable; lower-boundable by exhibition", value=v7, cite="Def 4.14, Cor 4.15"))
    ch.append(dict(q="Is the un-archived part noise or signal", status="walled; only the sum answerable (the remainder line above)", value=None, cite="Thm 2.11"))
    ch.append(dict(q="What is its name", status="walled at every rung; any test at exactly one half", value=None, cite="Thm 3.2"))
    ch.append(dict(q="Its own dynamics beyond the shadow pathway", status="walled before; measured upon acquisition", value=None, cite="Thm 9.4(ii)"))
    ch.append(dict(q="Can any purchasable column reach the specification", status="walled above the archive ceiling; rows 7 and the Screening carry the answerable part", value=None, cite="Thm 3.4"))
    ch.append(dict(q="Is the external component serially structured", status="walled in both directions at innovative lags; white by theorem otherwise", value=None, cite="Lem 7.4, Thm 7.5"))
    return ch


def characterize(sh: Shadow, atoms: Optional[Mapping[str, Any]] = None, po=None, level: float = 0.05,
                 lags: Sequence[int] = (1, 2, 7, 14), periods: Sequence[str] = ("day", "week"),
                 holidays: Sequence = ()) -> Characterization:
    """Basis interrogation, the persistence ledger and the charter filing
    on the shadow. Each atom family is declared, residualized against the
    licensed span (the intercept and the shadow's own p_hat) and
    orthonormalized within the family (Def 7.1). Deterministic frame
    families (Fourier lines, calendar) carry the exact hypothesis-free
    band of Prop 2.8, radius scaled by the quoted conditioning norm
    (Thm 7.3(ii)); data families (own lags, cross-members, archive) carry
    the clustered plug-in band (Thm 8.5(iii)). Leak guard: an atom that is
    the target's own realized outcome, or any column whose declared
    lineage descends from it, is refused by name. On a frame, one charter
    per member (the other members enter as its cross-sectional family)
    plus the shared frame block (Cor 7.7)."""
    atoms = dict(atoms or {})
    target_ids = {str(sh.manifest.get("y_sha")), "target"} if not sh.is_frame else {"target"} | {str(m.get("y_sha")) for m in sh.manifest.get("members", []) if isinstance(m, dict)}
    yv = sh.rows["y"].to_numpy() if (not sh.is_frame and "y" in sh.rows.columns) else None
    for fam in ("archive", "state"):
        for name, spec in dict(atoms.get(fam, {})).items():
            lin = set(map(str, spec.get("lineage", ())))
            if lin & target_ids:
                raise ValueError(f"characterize refused: atom '{name}' descends from the target by declared lineage; the target's own realized outcome is not an atom (leak guard)")
            v = _complete(spec["values"], "characterize", f"atom '{name}'").ravel()
            if yv is not None and len(v) == len(yv):
                vv, yy = v - v.mean(), yv - yv.mean()
                nv, ny = np.linalg.norm(vv), np.linalg.norm(yy)
                if nv > 0 and ny > 0 and abs(float(vv @ yy) / (nv * ny)) > 1 - 1e-9:
                    raise ValueError(f"characterize refused: atom '{name}' is affinely identical to the target's realized outcome on the aligned rows (leak guard)")
    caveats: List[str] = []
    if not sh.is_frame:
        cid = cluster_ids(sh.rows["clock"], sh.cluster)
        pipe = _unit_pipeline(sh.rows, sh.var, cid, sh.clock_step, atoms, level, lags, periods, holidays, member="unit")
        la = _lam_arch_of(pipe, sh, po, level)
        charter = _file_charter(pipe, sh.rows, sh.var, la, None)
        return Characterization(profiles=pipe["profiles"], ledger=pd.DataFrame(pipe["ledger"]),
                                persistence=pipe["persistence"], h_star=pipe["h_star"],
                                verdict_persistence=pipe["verdict"], gram=None, scope_shares=None,
                                charter=charter, member_charters=None, joint=pipe["joint"], lam_arch=la,
                                level=level, caveats=caveats + pipe["caveats"])
    # frame: shared block once, then one charter per member
    ecols = [c for c in sh.rows.columns if c.startswith("e")]
    E = sh.rows[ecols].to_numpy()
    n, M = E.shape
    cid = cluster_ids(sh.rows["clock"], sh.cluster)
    Er = _residualize(E, np.ones((n, 1)))
    dg = np.sqrt(np.clip((Er * Er).mean(axis=0), 1e-12, None))
    Cg = (Er.T @ Er) / n / np.outer(dg, dg)
    eig = np.linalg.eigvalsh(Cg)
    ref = (1 + np.sqrt(M / n)) ** 2
    P = np.linalg.pinv(Cg)
    dp = np.sqrt(np.clip(np.diag(P), 1e-12, None))
    partial = -P / np.outer(dp, dp)
    zb = norm.ppf(1 - level / (M * (M - 1)))
    pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            pr = float(partial[i, j])
            pairs.append(dict(pair=f"{i}-{j}", raw=float(Cg[i, j]), partial=pr, band=zb / np.sqrt(n), survives=abs(pr) > zb / np.sqrt(n)))
    top = float(eig[-1])
    rho_bar = float((Cg.sum() - M) / (M * (M - 1)))
    fw = float(np.clip(rho_bar, 0.0, 1.0))
    grp_share = float(np.clip((top - 1.0) / (M - 1) - fw, 0.0, 1.0 - fw))
    scope = dict(frame_wide=fw, group=grp_share, idiosyncratic=float(np.clip(1.0 - fw - grp_share, 0, 1)))
    co_moves = top > ref or any(p["survives"] for p in pairs)
    gram = dict(raw=Cg, top_eig=top, reference=float(ref), pairs=pairs,
                verdict="idiosyncratic" if not co_moves else ("frame-wide" if fw > grp_share else "group"))
    profs, leds, member_charters = [], [], []
    pers0 = h0 = v0 = None
    joint_acc = []
    la_frame = dict(value=0.0, band=0.0, scale="residual-variance", grade="est", aim_value=None)
    for i in range(M):
        rows_i = sh.rows[["clock", ecols[i]]].rename(columns={ecols[i]: "e"})
        var_i = float((rows_i["e"] ** 2).mean())
        at_i = dict(atoms)
        at_i["cross_members"] = E[:, [j for j in range(M) if j != i]]
        at_i["cross_labels"] = [j for j in range(M) if j != i]
        pipe = _unit_pipeline(rows_i, var_i, cid, sh.clock_step, at_i, level, lags, periods, holidays, member=f"m{i}")
        la_i = _lam_arch_of(pipe, sh, None, level, var_override=var_i)
        member_charters.append(_file_charter(pipe, rows_i, var_i, la_i, (gram, scope)))
        profs.append(pipe["profiles"])
        leds += pipe["ledger"]
        joint_acc.append(pipe["joint"]["share"])
        if i == 0:
            pers0, h0, v0 = pipe["persistence"], pipe["h_star"], pipe["verdict"]
            la_frame = la_i
    joint = dict(share=float(np.mean(joint_acc)), remainder_share=1.0 - float(np.mean(joint_acc)))
    return Characterization(profiles=pd.concat(profs, ignore_index=True) if profs else pd.DataFrame(),
                            ledger=pd.DataFrame(leds), persistence=pers0, h_star=h0, verdict_persistence=v0,
                            gram=gram, scope_shares=scope, charter=member_charters[0],
                            member_charters=member_charters, joint=joint, lam_arch=la_frame, level=level,
                            caveats=caveats + ["[frame joint share is the member mean; per-member ledgers above]"])


def _lam_arch_of(pipe, sh, po, level, var_override=None):
    var_e = sh.var if var_override is None else var_override
    arch_f = next((f for f in pipe["fams"] if f["name"] == "archive"), None)
    la = dict(value=0.0, band=0.0, scale="residual-variance", grade="est", aim_value=None)
    if arch_f is None:
        return la
    la["value"] = float(np.sqrt(max(arch_f["explained"] / max(var_e, 1e-300), 0.0)))
    la["band"] = float(_hf_radius(arch_f["K"], level, len(pipe["e_r"])))
    if po is not None and not sh.is_frame:
        blk = _po_block_on_slice(sh, po)
        if blk is not None:
            f_arch = arch_f["Q"] @ ((arch_f["Q"].T @ pipe["e_r"]) / len(pipe["e_r"]))
            w = _orth_slice(blk["w"])
            lam2, _, _, _ = _lambda_hat_stat(f_arch, w, sh.m_eff, blk["J"])
            la["aim_value"] = float(np.sqrt(max(lam2, 0.0)))
    return la


# ================================================================= screen

def _lambda_hat_stat(Z, W, m_eff, J):
    """Def 4.3: g_hat = mean(Z w), sigma_hat^2 = mean(Z^2),
    R = m_eff ||g_hat||^2 / sigma_hat^2, lam_hat^2 = (R - J)/(m_eff - 1)."""
    Z = np.asarray(Z, float).ravel()
    W = np.asarray(W, float)
    if W.ndim == 1:
        W = W[:, None]
    g = (W * Z[:, None]).mean(axis=0)
    s2 = float(np.mean(Z * Z))
    if s2 <= 0:
        return 0.0, 0.0, 0.0, 0
    R = m_eff * float(np.sum(g * g)) / s2
    lam2 = (R - J) / max(m_eff - 1.0, 1.0)
    return float(lam2), float(np.sqrt(max(lam2, 0.0))), float(R), int(np.sign(g[0]))


def _po_block_on_slice(sh: Shadow, po) -> Optional[dict]:
    """The state-whitened aim block Psi_hat with J from po, evaluated on
    the Shadow's retained state design. The frozen engine is called, never
    edited: the same deterministic construction the columns leg ran
    (SieveState fit on the Report panel, Localizer at the priced promise)
    is rebuilt from the Report the Shadow consumed, and its projector is
    applied to the slice rows in the Report's own encoding."""
    cols = getattr(po, "columns", None)
    if cols is None or cols.feasible_dark is None or sh._design is None or sh._rep is None:
        return None
    from .purchase._engine.localize import Localizer
    from .purchase._engine.state import SieveState
    rep = sh._rep
    Wp = pd.get_dummies(rep.cd.X_, dtype=float)
    if sh._design_cols != list(Wp.columns):
        return None
    W = Wp.to_numpy()
    yv = np.asarray(rep.cd.y, float).ravel().astype(int)
    st = SieveState(q=float(rep.cd.fdr_q)).fit(yv, W, run_provable=False)
    if st.refusal:
        return None
    pr = cols.promise
    loc = Localizer(st).run(M0=pr.M0, rho=pr.rho)
    if int(loc["J"]) != int(cols.feasible_dark["J"]):
        return None
    mu = W.mean(axis=0)
    R = W - np.where(yv[:, None] == 1, W[yv == 1].mean(axis=0)[None, :], W[yv == 0].mean(axis=0)[None, :])
    sd = np.sqrt((R * R).sum(axis=0) / (st.n - 2))
    sd = np.where(sd < 1e-12, 1.0, sd)
    w = ((sh._design - mu) / sd) @ loc["projector"]
    w = _residualize(w, _span_of(sh.rows))
    return dict(w=w, J=int(loc["J"]))


@dataclass
class Screening:
    """The screen of Section 4 for candidate columns against the aim: one
    row per candidate, the verdict counts, the floor with its source, the
    requirement grid, and the trichotomy line (Cor 4.15, Table 3)."""

    table: pd.DataFrame
    counts: dict
    floor: dict
    aim: dict
    trichotomy: dict
    grid: pd.DataFrame
    split: str
    level: float
    eps_miss: float
    eps_clear: float
    caveats: List[str]

    def render(self) -> str:
        L = [f"-- Screening (Defs 4.3-4.4; band ends one-sided at {self.level:g}, Lem 4.8 / Thm 4.11(i); split {self.split}) --"]
        rows = [[r.name, f"{r.lam_screen:.4f}", f"{r.lam_confirm:.4f}", f"{r.lam_full:.4f}", f"[{r.lo:.4f}, {r.hi:.4f}]",
                 f"{r.p_null:.3g}", f"{r.deff:.2f}", r.sign, r.verdict, f"{r.gap:+.4f}", f"{r.m_req_eff:,.0f} / {r.m_req_rows:,.0f}"]
                for r in self.table.itertuples()]
        L.append(tabulate(rows, headers=["candidate", "lam screen", "lam confirm", "lam full (descriptive)", "verdict band",
                                         "p_null (exact chi2_J, Cor 4.7)", "DEFF", "sign s/c", "verdict", "gap", "m_req eff/rows"], tablefmt="psql"))
        L.append("counts by verdict: " + ", ".join(f"{k} {v}" for k, v in self.counts.items()))
        fl = self.floor
        L.append(f"floor lam_min = {fl['value']:.4f} [{fl['source']}]" + (f"; lam* = {fl['lam_star']:.4f} beside it" if fl.get("lam_star") is not None else "") + (f"; derivation: {fl['derivation']}" if fl.get("derivation") else ""))
        if fl["value"] > 1.0:
            L.append("floor above the geometric ceiling lam <= 1: no admissible column can clear it; the promise is not unlockable by any single column at this slice (the order's feasibility clause)")
        L.append(f"aim block: {self.aim['which']}")
        L.append("m_req is the operating requirement of Def 4.4, achieved by the threshold test at t* = lam_min^2 + dlam^2 (Thm 4.9(a)); the plug-in band rule inherits it to first order (Thm 4.9(f)). Under a split the m_eff figure reads on the verdict slice and the rows figure carries the split factor.")
        grows = [[f"{r.dlam:g}", f"{r.m_req_eff:,.0f}", f"{r.m_req_rows:,.0f}",
                  "corner: the clear side is exact chi2_J null testing (Thm 4.9(e))" if r.corner else ""]
                 for r in self.grid.itertuples()]
        if not any(g[3] for g in grows):
            grows = [g[:3] for g in grows]
            L.append(tabulate(grows, headers=["dlam", "m_req (m_eff)", "m_req (rows)"], tablefmt="psql"))
        else:
            L.append(tabulate(grows, headers=["dlam", "m_req (m_eff)", "m_req (rows)", ""], tablefmt="psql"))
        tri = self.trichotomy
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
        L.append(f"trichotomy (Cor 4.15, Table 3): floor {fmt(tri['floor'])} vs lam_arch {fmt(tri['lam_arch'])} vs lam_cap {fmt(tri['lam_cap'])} -> {tri['action']}")
        if tri.get("archive_built_floor") is not None:
            L.append(f"archive-built floor by exhibition: {tri['archive_built_floor']:.4f} (largest confirmed lam_hat among archive-built candidates; the exhibiting band certifies the build verdict, Cor 4.15)")
        L.append("PASS is one-sided valid for the raw reliability: aim error understates only (Thm 4.11(i)). REJECT is decision-valid for the declared promise only (Thm 4.11(iv)).")
        for c in self.caveats:
            L.append(c)
        return "\n".join(L)

    def __str__(self):
        return self.render()


def _split_masks(sh: Shadow, split):
    ck = pd.to_datetime(sh.rows["clock"])
    if split == "interleaved-day":
        f = ck.dt.floor("D")
        epoch = pd.Timestamp("1970-01-01", tz=getattr(f.dt, "tz", None))
        par = ((f - epoch) // pd.Timedelta("1D")) % 2
        s = (par == 0).to_numpy()
        return s, ~s, float(len(ck) / max((~s).sum(), 1))
    if isinstance(split, tuple) and len(split) == 2 and split[0] == "era":
        b = pd.Timestamp(split[1])
        s = (ck < b).to_numpy()
        return s, ~s, float(len(ck) / max((~s).sum(), 1))
    raise ValueError(f"screen: split '{split}' is not declared; use 'interleaved-day' or ('era', boundary)")


def screen(sh: Shadow, candidates: Mapping[str, Any], floor: Optional[Tuple[float, str]] = None, po=None,
           findings: Optional[Characterization] = None, split="interleaved-day", level: float = 0.05,
           eps_miss: float = 0.10, eps_clear: float = 0.10, dlam: Sequence[float] = (0.02, 0.05, 0.10)) -> Screening:
    """The screen of Section 4 for one or many candidate columns against
    the aim. Admissibility (Ass 4.5): every candidate carries a
    decision-time attestation (its lag class or clock offset relative to
    the shadow's clock); a candidate without one is refused by name.
    Honesty (Thm 5.7): with more than one candidate, ranking uses the
    screen slice and every verdict band the confirm slice, disjoint by the
    declared split; the full-slice value prints as descriptive only, and
    the max-of-K caveat is stated. At K = 1 the full-slice band is honest
    (the selective and naive quotes coincide) and the halves serve only
    the J = 1 sign gate. The floor comes from po's columns leg (lam_A at
    the promise, lam* beside it) or from floor=(value, derivation),
    labelled DECLARED; exactly one of the two."""
    if sh.is_frame:
        raise ValueError("screen: one member at a time; pass a unit Shadow (the ruled frame convention)")
    cols = getattr(po, "columns", None) if po is not None else None
    have_po_floor = cols is not None and cols.feasible_dark is not None and cols.guaranteed_reliability is not None
    if floor is not None and have_po_floor:
        raise ValueError("screen refused: a declared floor and a columns-leg purchase order were both passed; one floor rules, drop one")
    if have_po_floor:
        blk = _po_block_on_slice(sh, po)
        if blk is None:
            raise ValueError("screen refused: the purchase order's aim block cannot be evaluated on this slice; rebuild the Shadow from the Report the order priced")
        aim_raw, J = blk["w"], blk["J"]
        which = f"state-whitened feasible-dark block Psi_hat from the purchase order (J = {J})"
        lam_min = float(np.sqrt(max(cols.guaranteed_reliability["lam2"], 0.0)))
        lam_star = None if cols.reliability_law_lam2 is None else float(np.sqrt(max(cols.reliability_law_lam2, 0.0)))
        fl = dict(value=lam_min, source="purchase order: guaranteed reliability lam_A at the declared promise (unlock at 1-delta)", lam_star=lam_star, derivation=None)
    elif floor is not None:
        lam_min = float(floor[0])
        fl = dict(value=lam_min, source="DECLARED", lam_star=None, derivation=str(floor[1]))
        span = _span_of(sh.rows)
        er = _residualize(sh.rows["e"].to_numpy()[:, None], span)[:, 0]
        aim_raw = (er / max(float(np.sqrt(np.mean(er * er))), 1e-12))[:, None]
        J = 1
        which = "the shadow itself, standardized (J = 1); the Screening says so because no columns leg was passed"
    else:
        raise ValueError("screen refused: no floor; pass floor=(value, derivation) or a purchase order with a priced columns leg (one of the two is mandatory)")
    e = sh.rows["e"].to_numpy()
    n = len(e)
    span = _span_of(sh.rows)
    cid = cluster_ids(sh.rows["clock"], sh.cluster)
    smask, cmask, split_factor = _split_masks(sh, split)
    K = len(candidates)
    zl = norm.ppf(1 - level)
    rows, caveats = [], []
    for name, spec in candidates.items():
        if not isinstance(spec, Mapping) or ("lag_class" not in spec and "clock_offset" not in spec):
            raise ValueError(f"screen refused: candidate '{name}' carries no decision-time attestation (its lag class or clock offset relative to the shadow's clock; Ass 4.5)")
        Z = _complete(spec["values"], "screen", f"candidate '{name}'").ravel()
        if len(Z) != n:
            raise ValueError(f"screen refused: candidate '{name}' does not align row for row with the slice")
        Zr = _residualize(Z[:, None], span)[:, 0]
        Zr = Zr / max(float(np.sqrt(np.mean(Zr * Zr))), 1e-12)
        d = max(deff_cluster(Zr * aim_raw[:, 0], cid), 1.0)

        def stat_on(mask):
            me = mask.sum() / d
            return _lambda_hat_stat(Zr[mask], _orth_slice(aim_raw[mask]), me, J), me

        (l2f, lf, Rf, sgf), mef = stat_on(np.ones(n, bool))
        (l2s, ls, Rs, sgs), mes = stat_on(smask)
        (l2c, lc, Rc, sgc), mec = stat_on(cmask)
        lam_v, R_v, me_v = (lf, Rf, mef) if K == 1 else (lc, Rc, mec)
        sign_ok = (J > 1) or (sgs == sgc) or (sgs == 0) or (sgc == 0)
        lo = float(xi_inv(xi(lam_v) - zl / np.sqrt(me_v)))
        hi = float(xi_inv(xi(lam_v) + zl / np.sqrt(me_v)))
        p_null = float(chi2.sf(R_v, df=J))
        if hi <= lam_min:
            verdict = "REJECT"
        elif J == 1 and not sign_ok:
            verdict = "REJECT (sign flip)"
        elif lo >= lam_min:
            verdict = "PASS"
        else:
            verdict = "CONTINUE"
        gap = lam_v - lam_min
        mg, corner = m_req(lam_min, max(abs(gap), 1e-4), J, eps_miss, eps_clear)
        rows.append(dict(name=str(name), lam_screen=ls, lam_confirm=lc, lam_full=lf, lo=lo, hi=hi, p_null=p_null,
                         deff=d, sign=f"{sgs:+d}/{sgc:+d}", verdict=verdict, gap=gap, m_req_eff=mg,
                         m_req_rows=mg * d * (split_factor if K > 1 else 1.0),
                         archive_built=bool(spec.get("archive_built", False)), corner=corner))
    tab = pd.DataFrame(rows)
    counts = tab["verdict"].value_counts().to_dict() if len(tab) else {}
    if K > 1:
        caveats.append(f"[K = {K}: the screen-slice ranking values are maxima over K selections and print descriptive; the verdict bands sit on the disjoint confirm slice, honest because the selection is independent of the confirm statistic (Thm 5.7); an unsplit winner quote would overstate by up to sqrt(2 log K) noise scale (Thm 5.6(iv))]")
    grid_rows = []
    med_d = float(tab["deff"].median()) if len(tab) else 1.0
    for dl in dlam:
        mg, corner = m_req(lam_min, float(dl), J, eps_miss, eps_clear)
        grid_rows.append(dict(dlam=float(dl), m_req_eff=mg, m_req_rows=mg * med_d * (split_factor if K > 1 else 1.0), corner=corner))
    grid = pd.DataFrame(grid_rows)
    ab = tab[tab["archive_built"]] if len(tab) else tab
    ab_floor = float(ab["lam_confirm"].max()) if len(ab) else None
    la = None if findings is None else float(findings.lam_arch["value"])
    lam_cap = None
    if findings is not None:
        lam_cap = float(np.sqrt(min(max(la, 0.0) ** 2 + max(findings.joint["remainder_share"], 0.0), 1.0)))
    if la is None:
        action = "not available: pass findings= from characterize to place lam_arch and lam_cap on the line"
    elif lam_min <= la:
        action = "build, do not buy (an archive-constructible realizer exists; Table 3)"
    elif lam_cap is not None and lam_min > lam_cap:
        action = "refuse: above the cap; pre-acquisition refusal is an amplitude statement, never a reliability statement (Table 3)"
    else:
        action = "screen, then buy (completion-contingent; the middle verdict is never certifiable, Cor 4.15)"
    tri = dict(floor=lam_min, lam_arch=la, lam_cap=lam_cap, action=action, archive_built_floor=ab_floor)
    if lam_cap is not None:
        caveats.append("[lam_cap is the upper envelope of Def 4.14: lam_arch^2 plus the walled remainder share, the walled aim budget bounded by the aim projection of the identified sum; est]")
    return Screening(table=tab, counts=counts, floor=fl, aim=dict(which=which, J=J), trichotomy=tri, grid=grid,
                     split=str(split), level=level, eps_miss=eps_miss, eps_clear=eps_clear, caveats=caveats)
