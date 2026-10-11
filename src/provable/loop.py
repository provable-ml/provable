"""The loop layer: grow the shadow, then absorb it.

Two verbs on a Shadow, both descriptive: they issue nothing, charge
nothing, and write nothing to any book.

  alternate(sh, classes)          -> Alternation   the build half: two declared
                                                   archive classes fitted in
                                                   alternation on the shadow
                                                   (backfitting, i.e. alternating
                                                   projection), the Friedrichs
                                                   angle between them, the
                                                   half-step ledger and its EXIT
                                                   chart
  pursue(sh, candidates, req)     -> Pursuit       the acquire half: forward
                                                   stagewise priced pursuit of
                                                   candidate columns against a
                                                   declared requirement, with the
                                                   trichotomy verdict at every
                                                   round and the staircase of
                                                   realized drops

What is theorem and what is design. Theorem: an acquisition moves mass
across the wall by exactly the increment of the archive mean, the
increments telescope along any chain of archives, each depends on the
order and their sum does not (Thm 9.4(i) of the companion paper iterated;
the L2 chain rule); the two-class alternation is alternating projection
onto the classes' orthogonal complements, it converges to the joint
projection, and the excess residual mass contracts by at least cos^2 of
the Friedrichs angle at every half-step after the first (von Neumann;
Kayalar and Weinert 1988; the angle of the complements equals the angle
of the classes, Deutsch 1995); the sum of realized increments is bounded
by the external mass of the starting archive, a walled quantity. Design:
the greedy price rule, the honesty split, and the reading of the
staircase as a stopping diagnostic. Every printed number carries a grade
and the statement or the design clause it comes from.
"""
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from tabulate import tabulate

from .inquiry import Shadow, _complete, _residualize, _span_of, _split_masks, cluster_ids, deff_cluster

INK, PAPER, CLARET, TEAL, SLATE, LINE = "#33302E", "#FFF1E5", "#990F3D", "#0D7680", "#66605C", "#CCC1B7"


# ---------------------------------------------------------------- algebra

def _basis(M: np.ndarray, rel_tol: float = 1e-8, scale: Optional[float] = None) -> np.ndarray:
    """Orthonormal basis (standard inner product) of the column span of M,
    numerically null directions dropped: a direction is null when its
    singular value is below rel_tol times the reference scale, the largest
    singular value of the columns before any residualization when given,
    else of M itself."""
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[:, None]
    if M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    ref = float(s.max()) if scale is None else float(scale)
    keep = s > rel_tol * max(ref, 1e-300)
    return U[:, keep]


def _scale(M: np.ndarray) -> float:
    """Reference scale of a column set: its largest singular value after
    centering the columns."""
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[:, None]
    if M.shape[1] == 0:
        return 1.0
    Mc = M - M.mean(axis=0)
    s = np.linalg.svd(Mc, compute_uv=False)
    return float(s.max()) if len(s) and s.max() > 0 else 1.0


def _angles(U: np.ndarray, V: np.ndarray, one_tol: float = 1e-9) -> Tuple[float, int, np.ndarray]:
    """Principal-angle cosines between two subspaces given by orthonormal
    bases. Returns (c, d0, cosines): c the Friedrichs cosine, the largest
    cosine strictly below one (zero when none, which is the case of nested
    or orthogonal classes); d0 the dimension of the intersection, the count
    of cosines at one."""
    if U.shape[1] == 0 or V.shape[1] == 0:
        return 0.0, 0, np.zeros(0)
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    d0 = int(np.sum(s > 1.0 - one_tol))
    below = s[s <= 1.0 - one_tol]
    c = float(below.max()) if len(below) else 0.0
    return c, d0, s


def _declared(spec: Mapping[str, Any], name: str, verb: str) -> np.ndarray:
    if not isinstance(spec, Mapping) or ("lag_class" not in spec and "clock_offset" not in spec):
        raise ValueError(f"{verb} refused: '{name}' carries no decision-time attestation (its lag class or clock offset relative to the shadow's clock)")
    v = _complete(spec["values"], verb, f"'{name}'")
    if v.ndim == 1:
        v = v[:, None]
    return v


def _leak_guard(sh: Shadow, name: str, v: np.ndarray, spec: Mapping[str, Any], verb: str):
    target_ids = {str(sh.manifest.get("y_sha")), "target"}
    if set(map(str, spec.get("lineage", ()))) & target_ids:
        raise ValueError(f"{verb} refused: '{name}' descends from the target by declared lineage; the target's own realized outcome is not an input (leak guard)")
    if "y" in sh.rows.columns:
        yy = sh.rows["y"].to_numpy() - float(sh.rows["y"].mean())
        ny = np.linalg.norm(yy)
        for j in range(v.shape[1]):
            vv = v[:, j] - v[:, j].mean()
            nv = np.linalg.norm(vv)
            if nv > 0 and ny > 0 and abs(float(vv @ yy) / (nv * ny)) > 1 - 1e-9:
                raise ValueError(f"{verb} refused: column {j} of '{name}' is affinely identical to the target's realized outcome on the aligned rows (leak guard)")


# ============================================================== Alternation

@dataclass
class Alternation:
    """The build half on one shadow: two declared archive classes fitted in
    alternation on the residual (each to the other's partial residual),
    the Friedrichs angle between their spans, the half-step ledger of the
    excess residual mass with the contraction bound, and the fixed point,
    which is the joint projection. Descriptive; nothing issued or
    charged."""

    ledger: pd.DataFrame
    classes: Tuple[str, str]
    c: float
    d0: int
    cosines: np.ndarray
    dims: Tuple[int, int, int]
    rounds: int
    converged: bool
    tol: float
    shares: dict
    excess0: float
    var: float
    caveats: List[str]

    def render(self) -> str:
        a, b = self.classes
        L = [f"-- Alternation (build half; free of acquisition, nothing issued or charged) --"]
        L.append(f"classes {a} (dim {self.dims[0]}) and {b} (dim {self.dims[1]}); joint span dim {self.dims[2]}; shared span dim {self.d0} [exact, slice geometry]")
        L.append(f"Friedrichs cosine c = {self.c:.4f} (angle {np.degrees(np.arccos(self.c)):.2f} deg); contraction bound per half-step after the first: c^2 = {self.c ** 2:.4f} [theorem: Kayalar and Weinert 1988; Deutsch 1995 for the complements]")
        rows = [[int(r.half_step), r.cls, f"{r.excess_in:.6g}", f"{r.excess_out:.6g}", f"{r.ratio:.4f}", "" if not np.isfinite(r.bound) else f"{r.bound:.4f}", f"{r.explained_share:.4f}"] for r in self.ledger.itertuples()]
        L.append(tabulate(rows, headers=["half-step", "class", "excess in", "excess out", "ratio", "bound c^2", "explained share so far"], tablefmt="psql"))
        L.append(f"excess = ||r_j - r_inf||^2_n with r_inf the joint projection's residual [exact]; explained share = fitted drop / ||e||^2_n [est of the population share]")
        sh = self.shares
        L.append(f"shares of ||e||^2_n: {a} alone {sh['a_alone']:.4f}, {b} alone {sh['b_alone']:.4f}, joint {sh['joint']:.4f} (the fixed point); remainder {sh['remainder']:.4f}")
        L.append(f"{'converged' if self.converged else 'not converged'} in {self.rounds} rounds ({2 * self.rounds} half-steps) to excess <= {self.tol:g} x initial; the ratio bound held at every half-step after the first [asserted on this run]")
        for c in self.caveats:
            L.append(c)
        return "\n".join(L)

    def __str__(self):
        return self.render()

    # ---------------------------------------------------------- EXIT chart
    def exit_points(self) -> pd.DataFrame:
        """The chart's data: normalized excess in and out per half-step."""
        e0 = max(self.excess0, 1e-300)
        d = self.ledger[["half_step", "cls", "excess_in", "excess_out", "ratio", "bound"]].copy()
        d["x"] = d["excess_in"] / e0
        d["y"] = d["excess_out"] / e0
        return d

    def exit_svg(self, width: int = 640, height: int = 460) -> str:
        """The build-half EXIT chart as a self-contained SVG: excess residual
        mass leaving a half-step against the excess entering it, both as
        fractions of the initial excess, on log axes; the diagonal; the
        theorem's bound line y = c^2 x; the two classes' half-steps; and
        the trajectory drawn as the usual staircase between the diagonal and
        the points."""
        pts = self.exit_points()
        vals = np.concatenate([pts["x"].to_numpy(), pts["y"].to_numpy()])
        vals = vals[vals > 0]
        lo = 10.0 ** np.floor(np.log10(max(vals.min() if len(vals) else 1e-6, 1e-12)))
        lo = min(lo, 1e-2)
        ml, mr, mt, mb = 64, 24, 54, 58
        W, H = width - ml - mr, height - mt - mb

        def sx(v):
            v = max(float(v), lo)
            return ml + W * (np.log10(v) - np.log10(lo)) / (0.0 - np.log10(lo))

        def sy(v):
            v = max(float(v), lo)
            return mt + H * (1.0 - (np.log10(v) - np.log10(lo)) / (0.0 - np.log10(lo)))

        L = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Georgia, \'Times New Roman\', serif">',
             f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
             f'<text x="{ml}" y="24" font-size="15" font-weight="700" fill="{INK}">Build half: alternation between {self.classes[0]} and {self.classes[1]}</text>',
             f'<text x="{ml}" y="42" font-size="11" fill="{SLATE}">Friedrichs cosine c = {self.c:.4f}, bound c&#178; = {self.c ** 2:.4f}; shared span {self.d0}; {self.rounds} rounds to excess &#8804; {self.tol:g} of initial</text>']
        # grid and ticks
        k = int(round(np.log10(lo)))
        for p in range(k, 1):
            v = 10.0 ** p
            L.append(f'<line x1="{sx(v):.1f}" y1="{mt}" x2="{sx(v):.1f}" y2="{mt + H}" stroke="{LINE}" stroke-width="0.6"/>')
            L.append(f'<line x1="{ml}" y1="{sy(v):.1f}" x2="{ml + W}" y2="{sy(v):.1f}" stroke="{LINE}" stroke-width="0.6"/>')
            L.append(f'<text x="{sx(v):.1f}" y="{mt + H + 16}" font-size="10" text-anchor="middle" fill="{SLATE}">1e{p}</text>')
            L.append(f'<text x="{ml - 6}" y="{sy(v) + 3.5:.1f}" font-size="10" text-anchor="end" fill="{SLATE}">1e{p}</text>')
        L.append(f'<rect x="{ml}" y="{mt}" width="{W}" height="{H}" fill="none" stroke="{INK}" stroke-width="1"/>')
        L.append(f'<text x="{ml + W / 2:.1f}" y="{height - 22}" font-size="11" text-anchor="middle" fill="{INK}">excess residual mass entering the half-step (fraction of initial)</text>')
        L.append(f'<text transform="translate(16 {mt + H / 2:.1f}) rotate(-90)" font-size="11" text-anchor="middle" fill="{INK}">excess residual mass leaving the half-step</text>')
        # diagonal and bound line
        L.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="{INK}" stroke-width="1"/>')
        c2 = self.c ** 2
        if c2 > 0:
            xa, xb = max(lo / c2, lo), 1.0
            if xa < xb:
                L.append(f'<line x1="{sx(xa):.1f}" y1="{sy(xa * c2):.1f}" x2="{sx(xb):.1f}" y2="{sy(xb * c2):.1f}" stroke="{CLARET}" stroke-width="1.2" stroke-dasharray="6 4"/>')
        # staircase: (x_j, y_j) -> (y_j, y_j) -> (y_j, y_{j+1})
        path = []
        xs, ys = pts["x"].to_numpy(), pts["y"].to_numpy()
        for j in range(len(xs)):
            if j == 0:
                path.append(f"M {sx(xs[0]):.1f} {sy(xs[0]):.1f}")
            path.append(f"L {sx(xs[j]):.1f} {sy(ys[j]):.1f}")
            if j + 1 < len(xs):
                path.append(f"L {sx(ys[j]):.1f} {sy(ys[j]):.1f}")
        L.append(f'<path d="{" ".join(path)}" fill="none" stroke="{SLATE}" stroke-width="1"/>')
        for r in pts.itertuples():
            if r.cls == self.classes[0]:
                L.append(f'<circle cx="{sx(r.x):.1f}" cy="{sy(r.y):.1f}" r="3.2" fill="{TEAL}"/>')
            else:
                L.append(f'<rect x="{sx(r.x) - 3:.1f}" y="{sy(r.y) - 3:.1f}" width="6" height="6" fill="{CLARET}"/>')
        # legend
        lx, ly = ml + 12, mt + 14
        L.append(f'<circle cx="{lx}" cy="{ly}" r="3.2" fill="{TEAL}"/><text x="{lx + 9}" y="{ly + 3.5}" font-size="10.5" fill="{INK}">half-step on {self.classes[0]}</text>')
        L.append(f'<rect x="{lx - 3}" y="{ly + 13}" width="6" height="6" fill="{CLARET}"/><text x="{lx + 9}" y="{ly + 19.5}" font-size="10.5" fill="{INK}">half-step on {self.classes[1]}</text>')
        L.append(f'<line x1="{lx - 4}" y1="{ly + 32}" x2="{lx + 6}" y2="{ly + 32}" stroke="{CLARET}" stroke-width="1.2" stroke-dasharray="6 4"/><text x="{lx + 9}" y="{ly + 35.5}" font-size="10.5" fill="{INK}">bound y = c&#178; x (theorem, after the first half-step)</text>')
        L.append(f'<line x1="{lx - 4}" y1="{ly + 48}" x2="{lx + 6}" y2="{ly + 48}" stroke="{INK}" stroke-width="1"/><text x="{lx + 9}" y="{ly + 51.5}" font-size="10.5" fill="{INK}">diagonal y = x (no progress)</text>')
        L.append("</svg>")
        return "\n".join(L)

    def exit_html(self, path: str) -> str:
        """Write the EXIT chart as one self-contained HTML page (no external
        fonts, scripts, or images) in the package's report theme; returns
        the path."""
        a, b = self.classes
        rows = "".join(f"<tr><td>{int(r.half_step)}</td><td>{r.cls}</td><td class='num'>{r.excess_in:.6g}</td><td class='num'>{r.excess_out:.6g}</td><td class='num'>{r.ratio:.4f}</td><td class='num'>{'' if not np.isfinite(r.bound) else f'{r.bound:.4f}'}</td><td class='num'>{r.explained_share:.4f}</td></tr>" for r in self.ledger.itertuples())
        doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Build half: alternation</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{background:{PAPER};color:{INK};font-family:Georgia,'Times New Roman',serif;line-height:1.45}}
.page{{max-width:780px;margin:0 auto;padding:34px 40px 26px}}.rule2{{border-top:2px solid {INK};border-bottom:1px solid {INK};height:5px;margin-bottom:14px}}
.kicker{{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:{SLATE}}}
h1{{font-size:26px;font-weight:700;margin:6px 0 10px}}p{{font-size:13.5px;max-width:64ch;margin:8px 0}}
table{{border-collapse:collapse;width:100%;margin-top:14px;font-size:12.5px}}th{{text-align:left;font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:{SLATE};border-bottom:1px solid {INK};padding:6px 6px}}
td{{padding:5px 6px;border-bottom:1px solid {LINE}}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}.foot{{font-size:11px;color:{SLATE};margin-top:14px}}</style></head>
<body><div class="page"><div class="rule2"></div><div class="kicker">provable &middot; loop layer &middot; build half</div>
<h1>Alternation between {a} and {b}</h1>
<p>Two declared archive classes fitted in alternation on the shadow, each to the other's partial residual. The fixed point is the joint projection; the excess residual mass contracts by at least c&#178; at every half-step after the first, with c the Friedrichs cosine between the two spans (shared span excluded).</p>
{self.exit_svg()}
<table><thead><tr><th>half-step</th><th>class</th><th>excess in</th><th>excess out</th><th>ratio</th><th>bound c&#178;</th><th>explained share</th></tr></thead><tbody>{rows}</tbody></table>
<p class="foot">c = {self.c:.4f}; dims {self.dims[0]} / {self.dims[1]} / joint {self.dims[2]}; shared span {self.d0}; shares of ||e||&#178;<sub>n</sub>: {a} alone {self.shares['a_alone']:.4f}, {b} alone {self.shares['b_alone']:.4f}, joint {self.shares['joint']:.4f}. Excess and ratios are exact slice geometry; shares are estimates of population shares. Free of acquisition: nothing issued, nothing charged.</p>
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        return path


def alternate(sh: Shadow, classes: Mapping[str, Any], tol: float = 1e-6, max_rounds: int = 500) -> Alternation:
    """The build half: fit two declared archive classes to the shadow in
    alternation, each to the other's partial residual (backfitting). In
    residual form this is alternating projection onto the two classes'
    orthogonal complements, so it converges to the residual of the joint
    projection, and the excess residual mass contracts by at least cos^2
    of the Friedrichs angle between the two spans at every half-step after
    the first; both are asserted on the run. Exactly two classes; each
    carries a decision-time attestation; the leak guard of characterize
    applies. The licensed span (intercept and the shadow's own p_hat) is
    removed from the residual and from both classes first."""
    if sh.is_frame:
        raise ValueError("alternate: one member at a time; pass a unit Shadow (the ruled frame convention)")
    if len(classes) != 2:
        raise ValueError(f"alternate refused: the build alternation is between two declared classes; {len(classes)} passed")
    names = list(classes.keys())
    n = sh.m
    span = _span_of(sh.rows)
    e_r = _residualize(sh.rows["e"].to_numpy()[:, None], span)[:, 0]
    var_e = max(sh.var, 1e-300)
    bases = []
    for nm in names:
        v = _declared(classes[nm], nm, "alternate")
        if v.shape[0] != n:
            raise ValueError(f"alternate refused: class '{nm}' does not align row for row with the slice")
        _leak_guard(sh, nm, v, classes[nm], "alternate")
        bases.append(_basis(_residualize(v, span), scale=_scale(v)))
    U, V = bases
    c, d0, cosines = _angles(U, V)
    Wj = _basis(np.column_stack([U, V])) if (U.shape[1] + V.shape[1]) else np.zeros((n, 0))
    r_inf = e_r - Wj @ (Wj.T @ e_r)
    fA = np.zeros(n)
    fB = np.zeros(n)
    r = e_r.copy()
    ex = lambda v: float(np.mean((v - r_inf) ** 2))
    excess0 = ex(r)
    rows = []
    hs = 0
    converged = False
    rounds = 0
    for k in range(max_rounds):
        for cls, B, which in ((names[0], U, 0), (names[1], V, 1)):
            hs += 1
            e_in = ex(r)
            g = B @ (B.T @ r)
            if which == 0:
                fA = fA + g
            else:
                fB = fB + g
            r = r - g
            e_out = ex(r)
            ratio = e_out / e_in if e_in > 0 else 0.0
            bound = c * c if hs >= 2 else np.nan
            if hs >= 2 and e_in > 1e-14 * max(excess0, 1e-300):
                assert ratio <= c * c + 1e-9, f"contraction bound failed at half-step {hs}: ratio {ratio:.6g} > c^2 {c * c:.6g}"
            drop = float(np.mean(e_r ** 2) - np.mean(r ** 2))
            rows.append(dict(half_step=hs, cls=cls, excess_in=e_in, excess_out=e_out, ratio=ratio, bound=bound, explained_share=drop / var_e))
        rounds = k + 1
        if ex(r) <= tol * max(excess0, 1e-300):
            converged = True
            break
    drop_joint = float(np.mean(e_r ** 2) - np.mean(r_inf ** 2))
    shares = dict(a_alone=float(np.mean((U @ (U.T @ e_r)) ** 2)) / var_e, b_alone=float(np.mean((V @ (V.T @ e_r)) ** 2)) / var_e,
                  joint=drop_joint / var_e, remainder=1.0 - drop_joint / var_e)
    caveats = ["[the split of the fitted value between the two classes is not identified on their shared span (concurvity); the sum is; a nonzero shared span is reported above]" if d0 > 0 else "[no shared span: the split of the fitted value between the two classes is identified]",
               "[alternation is a build algorithm inside the archive: its fixed point is the joint projection, at most the archive ceiling; nothing here moves the wall]"]
    return Alternation(ledger=pd.DataFrame(rows), classes=(names[0], names[1]), c=c, d0=d0, cosines=cosines,
                       dims=(U.shape[1], V.shape[1], Wj.shape[1]), rounds=rounds, converged=converged, tol=tol,
                       shares=shares, excess0=excess0, var=sh.var, caveats=caveats)


# ================================================================== Pursuit

@dataclass
class Pursuit:
    """The acquire half on one shadow: forward stagewise priced pursuit of
    declared candidate columns against a declared requirement. One row
    per round: the trichotomy verdict on the remaining requirement, the
    candidate acquired, its price, its confirmed drop with band, the
    residual mass before and after. Descriptive; the acquisitions are
    declared by the caller, not made by this verb, and nothing is issued
    or charged."""

    staircase: pd.DataFrame
    candidates: pd.DataFrame
    requirement: dict
    outcome: str
    total_drop: float
    total_drop_full: float
    total_price: float
    residual_share: float
    split: str
    level: float
    var: float
    caveats: List[str]

    def render(self) -> str:
        L = [f"-- Pursuit (acquire half; forward stagewise, priced; split {self.split}; bands one-sided at {self.level:g}) --"]
        rq = self.requirement
        L.append(f"requirement gamma = {rq['value']:.4f} of ||e||^2_n [{rq['source']}]" + (f"; derivation: {rq['derivation']}" if rq.get("derivation") else "") + (f"; archive-built share lam_arch^2 = {rq['lam_arch2']:.4f} from findings" if rq.get("lam_arch2") is not None else "; no findings passed, so the build verdict is not available"))
        rows = [[int(r.round), r.verdict, r.candidate, "" if not np.isfinite(r.price) else f"{r.price:.4f}", "" if not np.isfinite(r.delta) else f"{r.delta:.4f}", "" if not np.isfinite(r.delta_lo) else f"{r.delta_lo:.4f}", "" if not np.isfinite(r.delta_full) else f"{r.delta_full:.4f}", f"{r.residual_before:.4f}", "" if not np.isfinite(r.residual_after) else f"{r.residual_after:.4f}", f"{r.remaining:.4f}"] for r in self.staircase.itertuples()]
        L.append(tabulate(rows, headers=["round", "verdict", "acquired", "price", "drop (confirm)", "drop lower end", "drop (full, descr.)", "residual before", "residual after", "remaining req."], tablefmt="psql"))
        if len(self.candidates):
            crow = [[r.round, r.candidate, f"{r.delta:.4f}", f"{r.delta_lo:.4f}", f"{r.price:.4f}", f"{r.per_price:.3g}" if np.isfinite(r.per_price) else "free", r.status] for r in self.candidates.itertuples()]
            L.append(tabulate(crow, headers=["round", "candidate", "drop (confirm)", "lower end", "price", "drop per price", "status"], tablefmt="psql"))
        L.append(f"outcome: {self.outcome}")
        L.append(f"sum of confirmed drops = {self.total_drop:.4f} of ||e||^2_n at total price {self.total_price:.4f}; full-slice descriptive total = {self.total_drop_full:.4f}, which telescopes exactly and does not depend on the order [theorem]; remaining residual share {self.residual_share:.4f}")
        L.append("the theorem bound on the sum of increments, M_ext of the starting archive, is walled (Thm 2.11) and does not print; the printable envelope is the residual share above")
        for c in self.caveats:
            L.append(c)
        return "\n".join(L)

    def __str__(self):
        return self.render()


def pursue(sh: Shadow, candidates: Mapping[str, Any], requirement: Tuple[float, str], findings=None,
           split="interleaved-day", level: float = 0.05, max_rounds: Optional[int] = None) -> Pursuit:
    """The acquire half: forward stagewise priced pursuit. Each candidate
    carries a decision-time attestation and a declared price on the same
    scale as the drops (a share of ||e||^2_n; the exchange rate is the
    caller's). Each round: the remaining requirement is placed by the
    trichotomy (build when the archive-built share from findings covers
    it; impossible when it exceeds the residual mass; acquire between);
    every remaining candidate is residualized against the licensed span
    and the columns already acquired (only what the archive cannot compute
    is credited), its coefficients are fitted on the screen slice and its
    drop confirmed on the disjoint confirm slice with a clustered plug-in
    band; the candidate with the largest confirmed drop per unit price
    among those whose lower band end clears their price is acquired and
    absorbed, and the staircase records the step. Stops on: requirement
    met, build, impossible, stall (no candidate clears its price, a state
    the wall makes indistinguishable from noise reached), exhaustion.
    Greedy selection and the price rule are design; the increment
    accounting is theorem."""
    if sh.is_frame:
        raise ValueError("pursue: one member at a time; pass a unit Shadow (the ruled frame convention)")
    if not (isinstance(requirement, tuple) and len(requirement) == 2):
        raise ValueError("pursue refused: no requirement; pass requirement=(value, derivation), the required gain as a share of ||e||^2_n with its derivation (mandatory)")
    gamma = float(requirement[0])
    if not (0 < gamma):
        raise ValueError("pursue refused: the requirement must be a positive share of ||e||^2_n")
    n = sh.m
    span = _span_of(sh.rows)
    var_e = max(sh.var, 1e-300)
    e_r = _residualize(sh.rows["e"].to_numpy()[:, None], span)[:, 0]
    cid = cluster_ids(sh.rows["clock"], sh.cluster)
    smask, cmask, _ = _split_masks(sh, split)
    zl = norm.ppf(1 - level)
    cand = {}
    for name, spec in candidates.items():
        v = _declared(spec, name, "pursue")
        if v.shape[0] != n:
            raise ValueError(f"pursue refused: candidate '{name}' does not align row for row with the slice")
        if "price" not in spec:
            raise ValueError(f"pursue refused: candidate '{name}' carries no declared price; a pursuit is priced or it is not a pursuit")
        price = float(spec["price"])
        if price < 0:
            raise ValueError(f"pursue refused: candidate '{name}' carries a negative price")
        _leak_guard(sh, name, v, spec, "pursue")
        cand[str(name)] = (v, price)
    lam2 = None if findings is None else float(findings.lam_arch["value"]) ** 2
    rq = dict(value=gamma, source="DECLARED", derivation=str(requirement[1]), lam_arch2=lam2)
    B = np.zeros((n, 0))
    r = e_r.copy()
    r_full = e_r.copy()
    remaining = gamma
    acquired = []
    stairs, crows = [], []
    total_price = 0.0
    outcome = None
    K = len(cand)
    rounds_cap = K if max_rounds is None else min(K, int(max_rounds))
    for k in range(rounds_cap + 1):
        res_before = float(np.mean(r * r)) / var_e
        if remaining <= 1e-12:
            outcome = "met: the requirement is covered by the confirmed drops of the acquisitions"
            stairs.append(dict(round=k, verdict="met", candidate="", price=np.nan, delta=np.nan, delta_lo=np.nan, delta_full=np.nan, residual_before=res_before, residual_after=np.nan, remaining=0.0))
            break
        if lam2 is not None and lam2 >= remaining:
            outcome = "build: the exhibited archive-built share covers the remaining requirement (Cor 4.15, build verdict); no purchase"
            stairs.append(dict(round=k, verdict="build", candidate="", price=np.nan, delta=np.nan, delta_lo=np.nan, delta_full=np.nan, residual_before=res_before, residual_after=np.nan, remaining=remaining))
            break
        if remaining > res_before + 1e-12:
            outcome = "impossible: the remaining requirement exceeds the residual mass; no column of any law reaches it (Prop 2 impossible region)"
            stairs.append(dict(round=k, verdict="impossible", candidate="", price=np.nan, delta=np.nan, delta_lo=np.nan, delta_full=np.nan, residual_before=res_before, residual_after=np.nan, remaining=remaining))
            break
        if k == rounds_cap or not cand:
            outcome = "exhausted: no candidate left and the requirement stands; the remainder sits in the walled band"
            stairs.append(dict(round=k, verdict="exhausted", candidate="", price=np.nan, delta=np.nan, delta_lo=np.nan, delta_full=np.nan, residual_before=res_before, residual_after=np.nan, remaining=remaining))
            break
        best = None
        evals = {}
        for name, (v, price) in cand.items():
            Q = _basis(_residualize(_residualize(v, span), B) if B.shape[1] else _residualize(v, span), scale=_scale(v))
            if Q.shape[1] == 0:
                crows.append(dict(round=k, candidate=name, delta=0.0, delta_lo=0.0, price=price, per_price=np.nan, status="skipped: no content beyond the archive already held"))
                continue
            Qs, rs = Q[smask], r[smask]
            b = np.linalg.lstsq(Qs, rs, rcond=None)[0]
            rc, Qc = r[cmask], Q[cmask]
            psi = rc * rc - (rc - Qc @ b) ** 2
            d = max(deff_cluster(psi, np.asarray(cid)[cmask]), 1.0)
            delta = float(np.mean(psi)) / var_e
            band = zl * float(np.std(psi)) * np.sqrt(d) / np.sqrt(max(cmask.sum(), 1)) / var_e
            d_lo = delta - band
            d_full = float(np.sum((Q.T @ r_full) ** 2) / n) / var_e
            clears = d_lo >= price
            per = (delta / price) if price > 0 else (np.inf if delta > 0 else np.nan)
            evals[name] = (Q, b, delta, d_lo, d_full, price, per)
            crows.append(dict(round=k, candidate=name, delta=delta, delta_lo=d_lo, price=price, per_price=per, status="clears its price" if clears else "does not clear its price at the band's lower end"))
            if clears and (best is None or per > evals[best][6] or (per == evals[best][6] and delta > evals[best][2])):
                best = name
        if best is None:
            outcome = "stall: no remaining candidate clears its price at the lower band end; whether the residual is noise or an unbought feature is walled (Thm 2.11)"
            stairs.append(dict(round=k, verdict="stall", candidate="", price=np.nan, delta=np.nan, delta_lo=np.nan, delta_full=np.nan, residual_before=res_before, residual_after=np.nan, remaining=remaining))
            break
        Q, b, delta, d_lo, d_full, price, per = evals[best]
        r = r - Q @ b
        r_full = r_full - Q @ (Q.T @ r_full)
        B = np.column_stack([B, Q]) if B.shape[1] else Q
        del cand[best]
        acquired.append(best)
        total_price += price
        remaining = remaining - delta
        stairs.append(dict(round=k, verdict="acquire (walled band: above the archive-built share, below the residual mass; Prop 2)", candidate=best, price=price, delta=delta, delta_lo=d_lo, delta_full=d_full, residual_before=res_before, residual_after=float(np.mean(r * r)) / var_e, remaining=max(remaining, 0.0)))
    if outcome is None:
        outcome = "round cap reached"
    st = pd.DataFrame(stairs)
    total_drop = float(st["delta"].dropna().sum()) if len(st) else 0.0
    total_full = float(np.mean(e_r ** 2) - np.mean(r_full ** 2)) / var_e
    if len(st) and st["delta_full"].notna().any():
        tele = float(st["delta_full"].dropna().sum())
        assert abs(tele - total_full) <= 1e-9 * max(1.0, total_full), f"telescoping identity failed: {tele:.6g} vs {total_full:.6g}"
    caveats = ["[design: greedy selection by confirmed drop per unit price; each increment depends on the order, the full-slice total does not (the L2 chain rule); a different order can reach the requirement at a different price]",
               "[honesty: coefficients fitted on the screen slice, drops confirmed on the disjoint confirm slice with a clustered plug-in band; the full-slice drop prints descriptive only]",
               "[the acquisitions are declared inputs; this verb prices and orders them, it buys nothing and issues nothing]"]
    return Pursuit(staircase=st, candidates=pd.DataFrame(crows), requirement=rq, outcome=outcome, total_drop=total_drop,
                   total_drop_full=total_full, total_price=total_price, residual_share=float(np.mean(r * r)) / var_e,
                   split=str(split), level=level, var=sh.var, caveats=caveats)


# ================================================================ Selection

def _logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _enrich(Xa: np.ndarray, level: int) -> np.ndarray:
    """The class-enrichment menu on standardized admitted columns: level 0
    linear; level 1 adds squares; level 2 adds pairwise products."""
    if Xa.shape[1] == 0:
        return Xa
    cols = [Xa]
    if level >= 1:
        cols.append(Xa ** 2)
    if level >= 2 and Xa.shape[1] >= 2:
        p = Xa.shape[1]
        cols.append(np.column_stack([Xa[:, i] * Xa[:, j] for i in range(p) for j in range(i + 1, p)]))
    return np.column_stack(cols)


def _n_params(p: int, level: int) -> int:
    return p + (p if level >= 1 else 0) + (p * (p - 1) // 2 if level >= 2 else 0)


class _Logit:
    """Nearly unpenalized logistic fit on standardized columns (C = 1e3),
    so that separation on an enriched design does not abort a round."""

    def __init__(self, level: int):
        self.level = level
        self.mu = None
        self.sd = None
        self.model = None
        self.base = None

    def fit(self, Xa, y):
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(y, float)
        if Xa.shape[1] == 0:
            self.base = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            return self
        self.mu, self.sd = Xa.mean(axis=0), Xa.std(axis=0)
        self.sd = np.where(self.sd < 1e-12, 1.0, self.sd)
        D = _enrich((Xa - self.mu) / self.sd, self.level)
        self.model = LogisticRegression(C=1e3, max_iter=5000).fit(D, y)
        return self

    def predict_proba(self, Xa):
        if self.base is not None:
            return np.full(len(Xa), self.base)
        D = _enrich((Xa - self.mu) / self.sd, self.level)
        return self.model.predict_proba(D)[:, 1]


class _Learner:
    """The flexible learner that exhibits the archive ceiling on a column
    set: gradient-boosted trees at a fixed modest setting."""

    def __init__(self, params=None, seed: int = 0):
        self.params = dict(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=5, reg_lambda=1.0, tree_method="hist", n_jobs=2, random_state=seed, verbosity=0)
        if params:
            self.params.update(params)
        self.model = None
        self.base = None

    def fit(self, Xa, y):
        y = np.asarray(y, float)
        if Xa.shape[1] == 0:
            self.base = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            return self
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**self.params).fit(Xa, y)
        return self

    def predict_proba(self, Xa):
        if self.base is not None:
            return np.full(len(Xa), self.base)
        return self.model.predict_proba(Xa)[:, 1]


def _band(d, cid, level):
    """One-sided plug-in band half-width for the mean of per-row loss
    differences, clustered."""
    d = np.asarray(d, float)
    if len(d) < 3:
        return float("inf")
    dd = max(deff_cluster(d, cid), 1.0)
    return float(norm.ppf(1 - level) * np.std(d) * np.sqrt(dd) / np.sqrt(len(d)))


@dataclass
class Selection:
    """The loop run over owned columns as a feature-selection procedure
    for a logistic class: per round the trichotomy verdict, the candidate
    admitted, its model-free value (the flexible learner's confirmed
    log-loss drop, an exhibition lower bound on the conditional mutual
    information) and its class gain (the logistic fit's confirmed drop),
    the enrichment level the absorb step reached, and the class shadow
    left on the admitted set. Descriptive; nothing issued or charged."""

    rounds: pd.DataFrame
    candidates: pd.DataFrame
    admitted: List[str]
    level_enrich: int
    outcome: str
    loss_base: float
    loss_final: float
    ceiling_all: float
    requirement: Optional[dict]
    split: str
    level: float
    price_rule: str
    caveats: List[str]
    _fit: Any = None
    _cols: Optional[List[str]] = None
    budget: Optional[dict] = None

    def predict_proba(self, X) -> np.ndarray:
        """The final logistic fit with its enrichment, on fresh rows."""
        Xf = pd.DataFrame(X)
        if not self.admitted:
            return self._fit.predict_proba(np.zeros((len(Xf), 0)))
        return self._fit.predict_proba(Xf[self.admitted].to_numpy(float))

    def render(self) -> str:
        L = [f"-- Selection (the loop on owned columns; logistic class; split {self.split}; bands one-sided at {self.level:g}; price rule {self.price_rule}) --"]
        L.append(f"base-rate log-loss on the confirm slice {self.loss_base:.5f}; exhibited ceiling on all candidates {self.ceiling_all:.5f} (flexible learner, confirm slice) [est]" + ("" if self.requirement is None else f"; requirement {self.requirement['value']:.5f} of log-loss reduction [{self.requirement['derivation']}]"))
        rows = [[int(r.round), r.verdict, r.admitted, "" if not np.isfinite(r.value_free) else f"{r.value_free:+.5f} (lo {r.value_free_lo:+.5f})", "" if not np.isfinite(r.gain_class) else f"{r.gain_class:+.5f} (lo {r.gain_class_lo:+.5f})", "" if not np.isfinite(r.price) else f"{r.price:.5f}", f"{r.loss_before:.5f}", "" if not np.isfinite(r.loss_after) else f"{r.loss_after:.5f}", "" if not np.isfinite(r.shadow_after) else f"{r.shadow_after:+.5f}", "" if not np.isfinite(r.enrich) else str(int(r.enrich))] for r in self.rounds.itertuples()]
        L.append(tabulate(rows, headers=["round", "verdict", "admitted", "model-free value", "class gain", "price", "loss before", "loss after", "class shadow after", "enrich"], tablefmt="psql"))
        if len(self.candidates):
            last = self.candidates[self.candidates["round"] == self.candidates["round"].max()]
            crow = [[r.candidate, f"{r.rank_stat:+.5f}", f"{r.value_free:+.5f}", f"{r.value_free_lo:+.5f}", f"{r.gain_class:+.5f}", f"{r.gain_class_lo:+.5f}", f"{r.price:.5f}", r.status] for r in last.itertuples()]
            L.append(f"candidates at the last round ({int(self.candidates['round'].max())}); verdict bands at level {self.level:g} over the candidates on the confirm slice; ranking by the cross-fitted screen-slice value:")
            L.append(tabulate(crow, headers=["candidate", "rank stat", "model-free value", "lower end", "class gain", "lower end", "price", "status"], tablefmt="psql"))
        L.append(f"admitted {self.admitted} at enrichment level {self.level_enrich}; final confirm log-loss {self.loss_final:.5f} (reduction {self.loss_base - self.loss_final:+.5f} from the base rate)")
        if self.budget:
            b = self.budget
            L.append(f"class-fit parameter budget {b['params_allowed']:.0f} ({b['events']} events on the screen slice at {b['epv']:g} events per parameter); parameters in the final class {b['params_used']}" + (f"; enrichment stopped by the budget at level {b['stopped_at_level']} with the class shadow's lower end still {b['shadow_lower_end']:+.5f} above zero (unabsorbed by budget, not closed)" if b.get("stopped_at_level") is not None else ""))
        L.append(f"outcome: {self.outcome}")
        for c in self.caveats:
            L.append(c)
        return "\n".join(L)

    def __str__(self):
        return self.render()


def select(X, y, clock, candidates: Optional[Sequence[str]] = None, start: Sequence[str] = (), requirement: Optional[Tuple[float, str]] = None,
           split="interleaved-day", level: float = 0.05, price="none", enrich_max: int = 2, learner_params: Optional[dict] = None,
           max_rounds: Optional[int] = None, cluster: str = "day", seed: int = 0, epv: float = 10.0) -> Selection:
    """The loop on owned columns: forward stagewise selection for a logistic
    class, with the acquire step run by the model-free value of a candidate
    and the absorb step run by class enrichment. The admitted columns are
    the archive; every other declared candidate is a column at a price.

    Per round: the class (a logistic fit on the admitted columns at the
    current enrichment level) and the flexible learner (gradient-boosted
    trees on the admitted columns) are fitted on the screen slice and
    scored on the confirm slice; their gap is the class shadow, and the
    absorb step raises the enrichment level (squares, then pairwise
    products) while the shadow's lower band end stays above zero and the
    enriched class stays inside its parameter budget: at most one
    parameter per `epv` events on the screen slice (events = the smaller
    class count; the ten-events-per-parameter rule at the default), since
    pairwise products grow as the square of the admitted count and a
    near-unpenalized logistic fit beyond that budget is separation, not
    absorption. A stop by budget is named in the record and the shadow
    left is reported as unabsorbed by budget, not closed. Each
    remaining candidate is scored twice: its model-free value, the
    learner's confirmed log-loss drop when the column joins the admitted
    set, an exhibition lower bound on I(Y; X_j | admitted); and its class
    gain, the logistic fit's confirmed drop. Ranking uses a cross-fitted
    value inside the screen slice, so the max over candidates never
    touches the confirm slice; the winner is admitted when the larger of
    its two confirmed values clears its price at the lower band end, and
    when it does not, the highest-ranked candidate whose value is real at
    the union-bound level (level/K over the round's candidates) is admitted
    in its place, so that a stall means no candidate is real rather than
    that the screen-slice ranking picked the wrong near-duplicate.
    Prices: 'none' (the band alone), 'aic' (1/n per parameter on the mean
    log-loss scale), 'bic' (log n / 2n per parameter), or a mapping.
    Stops on met, impossible (requirement above the remaining log-loss),
    stall, exhaustion. Theorem: the accounting, the class shadow as the
    gap to the ceiling. Design: the ranking, the price rule, the
    enrichment menu, the learner."""
    Xf = pd.DataFrame(X).reset_index(drop=True)
    yv = np.asarray(y, float).ravel()
    if not set(np.unique(yv)) <= {0.0, 1.0}:
        raise ValueError("select: the target must be binary 0/1; binarizing it is a registration decision, not one this verb makes")
    ck = pd.to_datetime(pd.Series(clock)).reset_index(drop=True)
    if len(Xf) != len(yv) or len(ck) != len(yv):
        raise ValueError("select: X, y and clock must align row for row; a clock key is mandatory")
    nonnum = [c for c in Xf.columns if not pd.api.types.is_numeric_dtype(Xf[c])]
    if nonnum:
        raise ValueError(f"select refused: non-numeric columns {nonnum[:4]}; encode them before the loop")
    cands = [str(c) for c in (Xf.columns if candidates is None else candidates)]
    start = [str(c) for c in start]
    missing = [c for c in cands + start if c not in Xf.columns]
    if missing:
        raise ValueError(f"select refused: columns {missing[:4]} are not in X")
    cands = [c for c in cands if c not in start]
    for c in cands + start:
        v = _complete(Xf[c].to_numpy(float), "select", f"column '{c}'")
        vv, yy = v - v.mean(), yv - yv.mean()
        nv, ny = np.linalg.norm(vv), np.linalg.norm(yy)
        if nv > 0 and ny > 0 and abs(float(vv @ yy) / (nv * ny)) > 1 - 1e-9:
            raise ValueError(f"select refused: column '{c}' is affinely identical to the target (leak guard)")
    n = len(yv)
    if isinstance(price, str) and price not in ("none", "aic", "bic"):
        raise ValueError(f"select refused: price rule '{price}' is not declared; use 'none', 'aic', 'bic' or a mapping")
    # honesty split on the clock: screen (fit) and confirm (score); a cross-fit split inside the screen slice for ranking
    f = ck.dt.floor("D")
    epoch = pd.Timestamp("1970-01-01", tz=getattr(f.dt, "tz", None))
    day = ((f - epoch) // pd.Timedelta("1D")).to_numpy()
    if split == "interleaved-day":
        smask = (day % 2 == 0)
    elif isinstance(split, tuple) and split[0] == "era":
        smask = (ck < pd.Timestamp(split[1])).to_numpy()
    else:
        raise ValueError(f"select: split '{split}' is not declared; use 'interleaved-day' or ('era', boundary)")
    cmask = ~smask
    if smask.sum() < 50 or cmask.sum() < 50:
        raise ValueError("select refused: a slice has fewer than 50 rows")
    inner = (day % 4 == 0) if split == "interleaved-day" else (np.arange(n) % 2 == 0)
    sA, sB = smask & inner, smask & ~inner
    cid = cluster_ids(ck, cluster)
    cid_c = np.asarray(cid)[cmask]
    yS, yC = yv[smask], yv[cmask]
    M = lambda cols: Xf[cols].to_numpy(float) if cols else np.zeros((n, 0))

    def score(cols, level_e, mask_fit, mask_eval, use_learner):
        Xa = M(cols)
        mdl = (_Learner(learner_params, seed) if use_learner else _Logit(level_e)).fit(Xa[mask_fit], yv[mask_fit])
        return _logloss(mdl.predict_proba(Xa[mask_eval]), yv[mask_eval]), mdl

    def xfit(cols, level_e, use_learner):
        """Cross-fitted mean log-loss inside the screen slice (fit on one half, score the other, both ways)."""
        a, _ = score(cols, level_e, sA, sB, use_learner)
        b, _ = score(cols, level_e, sB, sA, use_learner)
        return float((a.sum() + b.sum()) / (sA.sum() + sB.sum()))

    zl = norm.ppf(1 - level)
    price_rule = price if isinstance(price, str) else "declared"

    def price_of(c, k_params):
        if isinstance(price, str):
            return 0.0 if price == "none" else (k_params / n if price == "aic" else k_params * np.log(n) / (2 * n))
        return float(price.get(c, 0.0))

    admitted = list(start)
    lvl = 0
    events = int(min(yS.sum(), len(yS) - yS.sum()))
    if epv <= 0:
        raise ValueError("select refused: epv (events per parameter) must be positive")
    params_allowed = events / float(epv)
    budget = dict(events=events, epv=float(epv), params_allowed=params_allowed, params_used=_n_params(len(admitted), 0), stopped_at_level=None, shadow_lower_end=None)
    if _n_params(len(admitted), 0) > params_allowed:
        budget["start_over_budget"] = True
    l_base = float(_logloss(np.full(cmask.sum(), yS.mean()), yC).mean())
    lc_all, _ = score(start + cands, 0, smask, cmask, True)
    ceiling_all = float(lc_all.mean())
    rq = None if requirement is None else dict(value=float(requirement[0]), derivation=str(requirement[1]))
    rounds, crows = [], []
    outcome = None
    remaining = list(cands)
    cap = len(cands) if max_rounds is None else min(len(cands), int(max_rounds))

    def absorb(cols, lvl0):
        """Raise the enrichment level while the class shadow's lower end stays above zero."""
        lvl_ = lvl0
        lc_class, fit_ = score(cols, lvl_, smask, cmask, False)
        lc_learn, _ = score(cols, 0, smask, cmask, True)
        while True:
            gap = lc_class - lc_learn
            sh, hw = float(gap.mean()), _band(gap, cid_c, level)
            if sh - hw <= 0 or lvl_ >= enrich_max or len(cols) == 0:
                budget["stopped_at_level"], budget["shadow_lower_end"] = None, None
                return lvl_, fit_, float(lc_class.mean()), float(lc_learn.mean()), sh
            if _n_params(len(cols), lvl_ + 1) > params_allowed:
                budget["stopped_at_level"], budget["shadow_lower_end"] = lvl_, sh - hw
                return lvl_, fit_, float(lc_class.mean()), float(lc_learn.mean()), sh
            lvl_ += 1
            lc_class, fit_ = score(cols, lvl_, smask, cmask, False)

    lvl, fit, l_cur, h_cur, sh_cur = absorb(admitted, 0)
    for k in range(cap + 1):
        red = l_base - l_cur
        rem = None if rq is None else rq["value"] - red
        base_row = dict(round=k, admitted="", value_free=np.nan, value_free_lo=np.nan, gain_class=np.nan, gain_class_lo=np.nan, price=np.nan, loss_before=l_cur, loss_after=np.nan, shadow_before=sh_cur, shadow_after=np.nan, enrich=np.nan)
        if rem is not None and rem <= 1e-12:
            outcome = "met: the requirement is covered by the confirmed reduction"
            rounds.append(dict(base_row, verdict="met")); break
        if rem is not None and rem > l_cur + 1e-12:
            outcome = "impossible: the remaining requirement exceeds the log-loss itself; no column of any law reaches it"
            rounds.append(dict(base_row, verdict="impossible")); break
        if k == cap or not remaining:
            outcome = "exhausted: no candidate left"
            rounds.append(dict(base_row, verdict="exhausted")); break
        warn = rem is not None and rem > l_cur - ceiling_all + 1e-12
        best, evals = None, {}
        K = len(remaining)
        xf_learn_a, xf_class_a = xfit(admitted, 0, True), xfit(admitted, lvl, False)
        lc_learn_a, _ = score(admitted, 0, smask, cmask, True)
        lc_class_a, _ = score(admitted, lvl, smask, cmask, False)
        for c in remaining:
            cols = admitted + [c]
            kp = _n_params(len(cols), lvl) - _n_params(len(admitted), lvl)
            pr = price_of(c, kp)
            rank_stat = max(xf_learn_a - xfit(cols, 0, True), xf_class_a - xfit(cols, lvl, False))
            lc_learn_j, _ = score(cols, 0, smask, cmask, True)
            lc_class_j, _ = score(cols, lvl, smask, cmask, False)
            dfree, dclass = lc_learn_a - lc_learn_j, lc_class_a - lc_class_j
            vf, vc = float(dfree.mean()), float(dclass.mean())
            hf, hc = _band(dfree, cid_c, level), _band(dclass, cid_c, level)
            hfK, hcK = _band(dfree, cid_c, level / K), _band(dclass, cid_c, level / K)
            real_free, real_class = vf - hfK > pr, vc - hcK > pr
            status = ("real, absorbed by the class" if real_class else ("real, unabsorbed by the class (enrich after admission)" if real_free else "no exhibited value beyond the admitted set"))
            evals[c] = dict(cols=cols, price=pr, vf=vf, vc=vc, lo_f=vf - hf, lo_c=vc - hc, lo_fK=vf - hfK, lo_cK=vc - hcK, rank=rank_stat)
            crows.append(dict(round=k, candidate=c, rank_stat=rank_stat, value_free=vf, value_free_lo=vf - hfK, gain_class=vc, gain_class_lo=vc - hcK, price=pr, status=status))
            if best is None or rank_stat > evals[best]["rank"]:
                best = c
        e = evals[best]
        clears = max(e["lo_f"], e["lo_c"]) > e["price"]
        fallback = False
        if not clears:
            # the ranked winner failed at its own band; the per-candidate statuses hold simultaneously at level/K, so the
            # highest-ranked candidate that is real at the union-bound level is admitted instead (near-duplicate candidates
            # make the screen-slice ranking noisy, and a stall must mean that no candidate is real, not that the ranking guessed wrong)
            real = [c for c in sorted(remaining, key=lambda c: -evals[c]["rank"]) if max(evals[c]["lo_fK"], evals[c]["lo_cK"]) > evals[c]["price"]]
            if real:
                best, fallback = real[0], True
                e, clears = evals[best], True
        if not clears:
            outcome = "stall: the top-ranked candidate does not clear its price at the lower band end on the confirm slice and no candidate is real at the union-bound level; whether the residual is noise or an unadmitted column is walled" + ("; the remaining requirement also lies beyond the exhibited ceiling" if warn else "")
            rounds.append(dict(base_row, verdict="stall")); break
        admitted = e["cols"]
        remaining.remove(best)
        lvl, fit, l_new, h_new, sh_new = absorb(admitted, lvl)
        verdict = "acquire" + (" (union-bound fallback: the ranked winner did not clear at its own band)" if fallback else "") + (" (remaining requirement beyond the exhibited ceiling)" if warn else "")
        rounds.append(dict(round=k, verdict=verdict, admitted=best, value_free=e["vf"], value_free_lo=e["lo_f"], gain_class=e["vc"], gain_class_lo=e["lo_c"], price=e["price"], loss_before=l_cur, loss_after=l_new, shadow_before=sh_cur, shadow_after=sh_new, enrich=lvl))
        assert abs((l_cur - l_new) - (e["vf"] - (sh_new - sh_cur))) <= 1e-9, "loop accounting failed: class reduction != model-free value minus the change in the class shadow"
        l_cur, h_cur, sh_cur = l_new, h_new, sh_new
    if outcome is None:
        outcome = "round cap reached"
    caveats = ["[design: ranking by the cross-fitted value inside the screen slice, admission by the winner's confirmed value on the confirm slice at the lower band end; per-candidate statuses use a union-bound band over the candidates of the round; when the winner fails at its own band the highest-ranked candidate real at the union-bound level is admitted instead, so a stall means no candidate is real at level/K]",
               "[theorem: the confirmed reduction of the class equals the model-free value admitted minus the change in the class shadow; the model-free value is an exhibition lower bound on the conditional mutual information, never its value]",
               "[design: the enrichment menu is squares then pairwise products; the learner is gradient-boosted trees at a fixed modest setting; both are declared, not derived]",
               f"[design: the enriched class is held to one parameter per {epv:g} events on the screen slice ({events} events, {params_allowed:.0f} parameters); a level the budget cannot carry is not fitted, and the shadow it would have addressed is reported as unabsorbed by budget]"]
    budget["params_used"] = _n_params(len(admitted), lvl)
    if budget.get("start_over_budget"):
        caveats.append(f"[the admitted start set alone ({len(start)} columns) exceeds the parameter budget of {params_allowed:.0f}; the class was fitted at level 0 as declared and never enriched]")
    if budget["stopped_at_level"] is not None:
        caveats.append(f"[enrichment stopped at level {budget['stopped_at_level']} by the parameter budget: level {budget['stopped_at_level'] + 1} on {len(admitted)} admitted columns needs {_n_params(len(admitted), budget['stopped_at_level'] + 1)} parameters against {params_allowed:.0f}; the class shadow's lower end was still {budget['shadow_lower_end']:+.5f} above zero, so the shadow left is unabsorbed by budget, not closed]")
    return Selection(rounds=pd.DataFrame(rounds), candidates=pd.DataFrame(crows), admitted=admitted, level_enrich=lvl, outcome=outcome,
                     loss_base=l_base, loss_final=l_cur, ceiling_all=ceiling_all, requirement=rq, split=str(split), level=level,
                     price_rule=price_rule, caveats=caveats, _fit=fit, _cols=list(Xf.columns), budget=budget)
