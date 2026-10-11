"""provable.transport -- the screen that routes a choice before it is tuned.

A choice between two alternatives (a feature in or out, a hyperparameter, a
regime filter) can be tuned through a holdout safely when its effect is the
same in every time block. When the effect moves from block to block, an
analyst who remembers what the holdout period was like can steer the choice
with that memory, outside the holdout's log, and the certificate the holdout
issues stops describing the future. route() reads the choice's effect on the
training rows only, block by block, and decides where the choice belongs:
through the holdout (VAULT), or into a forward registration with predictions
issued before their outcomes (FORWARD).

  route(d, blocks, h=..., r=2)  -> Routing   the verdict, the effect per
                                             block, the bound on its spread
                                             and the fluctuation statistic

  d       the per-row paired loss differences of the choice on the training
          rows (alternative a's loss minus b's), cross-fitted so that no row
          scores a model fitted on it, in time order
  blocks  the time block of each row, at the resolution at which history is
          remembered; each block one contiguous run
  h       the holdout rows that would answer the choice (holdout= reads the
          tuning slice of a sealed holdout)
  r       the regimes per block that memory can tell apart

What is theorem and what is design:
  theorem  Memory acts only through the spread of the effect. Let the
           holdout's estimate of the effect be sum_m w_m delta_m plus noise
           independent of the regime path, the block effects delta_m being
           independent across blocks and subgaussian about their mean with
           proxy tau_c. Any selection that uses information M about the
           regime path biases the certificate for the mean effect by at most
           tau_c |w|_2 sqrt(2 I(choice; M)); with equal weights over the
           holdout blocks and memory that tells r regimes apart in each, at
           most tau_c sqrt(2 ln r), whatever the length of the holdout. With
           tau_c = 0 the estimate is independent of M. (Donsker-Varadhan, as
           for the information bound on a reported score.)
  cited    The upper confidence bound on tau_c^2 is the Q-profile interval
           (Viechtbauer, Statistics in Medicine 2007) of the random-effects
           model delta_hat_m = delta_m + e_m with e_m of variance s_m^2, the
           block's standard error; Cochran's Q and the DerSimonian-Laird
           estimate are reported beside it. The fluctuation statistic is the
           one of Giacomini and Rossi (Journal of Applied Econometrics 2010),
           the largest moving-window sum of the differences scaled by a
           long-run standard deviation, taken here on the demeaned
           differences, so that its null is a constant effect rather than
           equal performance at every point; its critical value is the 1 -
           delta quantile of the limiting Brownian-bridge statistic, by a
           seeded simulation.
  design   The rule: VAULT when the fluctuation statistic stays below its
           critical value and tau_c^UCB sqrt(2 ln r) <= z_0 se_H, so that the
           worst memory bias stays below one clean bar of holdout noise;
           FORWARD otherwise. The rule uses the upper bound rather than a
           failure to reject, so a short history, whose bound is wide, routes
           forward: the conservative failure. se_H is the standard deviation
           of d over sqrt(h); the window is a share of the rows (0.3 by
           default); the long-run standard deviation is Newey-West with
           bandwidth floor(4 (n/100)^(2/9)).
  meaning  The verdict concerns the choice's effect on the training rows,
           read without the holdout; it spends no bits and charges no budget.
           With ledger= it is recorded in the book as the choice's transport
           status.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, norm
from tabulate import tabulate

from . import holdout as _ho
from .certificates.lineage import Lineage, identity_overlap
from .certificates.store import Ledger
from .holdout import _canon, _npy_bytes, _sha

VERDICTS = ("VAULT", "FORWARD")
_CRIT = {}


def fluctuation_critical(window: float, delta: float, paths: int = 20000, steps: int = 2000, seed: int = 20100701) -> float:
    """The 1 - delta quantile of sup over s in [window, 1] of |B(s) - B(s -
    window)| / sqrt(window), B a Brownian bridge on [0, 1]: the limit of the
    fluctuation statistic on demeaned differences under a constant effect.
    Seeded simulation on a grid, cached per (window, delta)."""
    key = (round(float(window), 6), round(float(delta), 6), paths, steps, seed)
    if key not in _CRIT:
        rng = np.random.default_rng(seed)
        k = max(1, int(round(window * steps)))
        sups = np.empty(paths)
        for i in range(0, paths, 2000):
            z = rng.standard_normal((min(2000, paths - i), steps)) / math.sqrt(steps)
            w = np.concatenate([np.zeros((len(z), 1)), np.cumsum(z, axis=1)], axis=1)
            b = w - np.linspace(0.0, 1.0, steps + 1)[None, :] * w[:, -1:]
            sups[i:i + len(z)] = np.abs(b[:, k:] - b[:, :-k]).max(axis=1) / math.sqrt(k / steps)
        _CRIT[key] = float(np.quantile(sups, 1.0 - delta))
    return _CRIT[key]


def _newey_west_sd(x: np.ndarray) -> float:
    n = len(x)
    e = x - x.mean()
    lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    v = float(e @ e) / n
    for j in range(1, min(lag, n - 1) + 1):
        v += 2.0 * (1.0 - j / (lag + 1.0)) * float(e[j:] @ e[:-j]) / n
    return math.sqrt(max(v, 0.0))


def _q_gen(t2: float, eff: np.ndarray, s2: np.ndarray) -> float:
    w = 1.0 / (s2 + t2)
    mu = float(np.sum(w * eff) / np.sum(w))
    return float(np.sum(w * (eff - mu) ** 2))


def tau_upper(eff: np.ndarray, se: np.ndarray, delta: float) -> float:
    """The Q-profile upper confidence bound at 1 - delta on the spread tau of
    the block effects: the tau^2 at which the generalized Q statistic falls to
    the delta quantile of chi^2 with M - 1 degrees of freedom (zero when it is
    already below at tau^2 = 0)."""
    eff, s2 = np.asarray(eff, float), np.asarray(se, float) ** 2
    q = float(chi2.ppf(delta, len(eff) - 1))
    if _q_gen(0.0, eff, s2) <= q:
        return 0.0
    hi = max(float(np.var(eff)), float(s2.max()), 1e-300)
    while _q_gen(hi, eff, s2) > q:
        hi *= 4.0
    return math.sqrt(brentq(lambda t2: _q_gen(t2, eff, s2) - q, 0.0, hi, xtol=1e-14, rtol=1e-12, maxiter=500))


@dataclass
class Routing:
    """The verdict of the transport screen and the numbers behind it."""
    verdict: str
    reason: str
    blocks: pd.DataFrame
    rows: int
    effect: float
    q: float
    q_p: float
    tau_dl: float
    tau_ucb: float
    r: int
    memory_bound: float
    h: int
    se_h: float
    z0: float
    bar: float
    window: float
    fluctuation: float
    critical: float
    delta: float
    label: str = ""
    record: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def render(self) -> str:
        head = [["verdict", f"{self.verdict}: {self.reason}"], ["rows, blocks", f"{self.rows}, {len(self.blocks)}"], ["mean effect (a minus b)", f"{self.effect:.6g}"],
                ["Cochran's Q (p)", f"{self.q:.4g} ({self.q_p:.3g}) on {len(self.blocks) - 1} df"], ["spread of the effect tau_c", f"{self.tau_dl:.4g} (DerSimonian-Laird), at most {self.tau_ucb:.4g} at {1 - self.delta:g} (Q-profile)"],
                ["worst memory bias", f"tau_c^UCB sqrt(2 ln {self.r}) = {self.memory_bound:.4g}"], ["one clean bar of holdout noise", f"z_0 se_H = {self.z0:.4f} x {self.se_h:.4g} = {self.bar:.4g} (h = {self.h})"],
                ["fluctuation statistic", f"{self.fluctuation:.4f} against {self.critical:.4f} (window {self.window:g} of the rows)"]]
        if self.record:
            head.append(["recorded", self.record])
        out = [f"-- route{': ' + self.label if self.label else ''} --", tabulate(head, tablefmt="psql")]
        return "\n".join(out)

    def __str__(self):
        return self.render()


def route(d, blocks, *, h: Optional[int] = None, holdout=None, r: int = 2, delta: float = 0.05, window: float = 0.3, lineage: Optional[Lineage] = None, ledger=None, label: str = "") -> Routing:
    """Screen a choice on the training rows and route it: VAULT (tune it
    through the holdout) or FORWARD (register it forward). d holds the
    cross-fitted per-row paired loss differences of the choice in time order
    and blocks the time block of each row, each block one contiguous run of
    at least two rows, at least three blocks. h is the number of holdout rows
    that would answer the choice, or holdout= a sealed holdout whose tuning
    slice answers it (with lineage=, the rows d was computed on, refused when
    they meet rows in its custody). r is the number of regimes per block that
    memory can tell apart. With ledger= the verdict is recorded in the book;
    it charges nothing."""
    dv = np.asarray(d, dtype=float).ravel()
    bl = np.asarray(blocks).ravel()
    n = len(dv)
    if len(bl) != n:
        raise ValueError(f"route refused: {len(bl)} block labels for {n} differences")
    if n == 0 or not np.all(np.isfinite(dv)):
        raise ValueError("route refused: the differences must be finite, one per row")
    if isinstance(r, bool) or int(r) != r or int(r) < 1:
        raise ValueError("route refused: r is the number of regimes memory tells apart per block, a whole number at least 1")
    r = int(r)
    delta = float(delta)
    if not (0.0 < delta < 0.5):
        raise ValueError("route refused: delta must lie in (0, 0.5)")
    if not (0.0 < float(window) < 1.0):
        raise ValueError("route refused: window is a share of the rows in (0, 1)")
    change = np.r_[True, bl[1:] != bl[:-1]]
    labels = bl[change]
    if len(set(labels.tolist())) != len(labels):
        raise ValueError("route refused: a block appears in more than one run; give the rows in time order with each block contiguous")
    if len(labels) < 3:
        raise ValueError(f"route refused: {len(labels)} blocks; the spread of an effect across blocks needs at least three")
    if holdout is not None:
        st = holdout.commit
        h = int(st["n_tune"])
        if lineage is not None:
            custody = holdout._sealed_lineage(holdout.state)
            if identity_overlap(lineage, custody):
                raise ValueError(f"route refused: the rows of d meet rows in the custody of {holdout.id}; the screen reads the training rows only")
    if h is None or int(h) < 2:
        raise ValueError("route refused: pass h, the holdout rows that would answer the choice (at least 2), or holdout= a sealed holdout")
    h = int(h)
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    eff, se, cnt = [], [], []
    for a, b in zip(starts, ends):
        x = dv[a:b]
        if len(x) < 2:
            raise ValueError(f"route refused: block {bl[a]} has {len(x)} row; a block's standard error needs at least two")
        s = float(x.std(ddof=1)) / math.sqrt(len(x))
        if not s > 0:
            raise ValueError(f"route refused: the differences in block {bl[a]} are constant, so its standard error is zero")
        eff.append(float(x.mean()))
        se.append(s)
        cnt.append(len(x))
    eff, se = np.asarray(eff), np.asarray(se)
    M = len(eff)
    w = 1.0 / se ** 2
    q = _q_gen(0.0, eff, se ** 2)
    q_p = float(chi2.sf(q, M - 1))
    tau_dl = math.sqrt(max(0.0, (q - (M - 1)) / (w.sum() - (w ** 2).sum() / w.sum())))
    tau_ucb = tau_upper(eff, se, delta)
    mem = tau_ucb * math.sqrt(2.0 * math.log(r))
    z0 = float(norm.isf(delta))
    se_h = float(dv.std(ddof=1)) / math.sqrt(h)
    bar = z0 * se_h
    k = max(2, int(round(float(window) * n)))
    sd = _newey_west_sd(dv)
    c = np.concatenate([[0.0], np.cumsum(dv - dv.mean())])
    fl = float(np.abs(c[k:] - c[:-k]).max() / (sd * math.sqrt(k))) if sd > 0 else math.inf
    crit = fluctuation_critical(float(window), delta)
    if fl > crit:
        verdict, reason = "FORWARD", f"the effect moves over time (fluctuation {fl:.3f} > {crit:.3f})"
    elif mem > bar:
        verdict, reason = "FORWARD", f"the worst memory bias {mem:.4g} exceeds one clean bar of holdout noise {bar:.4g}"
    else:
        verdict, reason = "VAULT", f"the worst memory bias {mem:.4g} is within one clean bar of holdout noise {bar:.4g}"
    tab = pd.DataFrame({"block": [bl[a] for a in starts], "rows": cnt, "effect": eff, "se": se})
    out = Routing(verdict=verdict, reason=reason, blocks=tab, rows=n, effect=float(dv.mean()), q=float(q), q_p=q_p, tau_dl=tau_dl, tau_ucb=tau_ucb, r=r, memory_bound=mem, h=h, se_h=se_h, z0=z0, bar=bar, window=float(window), fluctuation=fl, critical=crit, delta=delta, label=str(label))
    if ledger is not None:
        led = ledger if isinstance(ledger, Ledger) else Ledger(str(ledger))
        op = {"op": "route", "at": _ho._now(), "label": str(label), "verdict": verdict, "reason": reason, "rows": n, "blocks": M, "effect": out.effect, "q": out.q, "tau_dl": tau_dl, "tau_ucb": tau_ucb, "r": r, "memory_bound": mem, "h": h, "se_h": se_h, "bar": bar, "window": float(window), "fluctuation": fl, "critical": crit, "delta": delta, "holdout": None if holdout is None else holdout.id, "d_sha": _sha(_npy_bytes(dv)), "blocks_sha": _sha(_canon([str(x) for x in bl.tolist()]))}
        rid = "RT-" + _sha(_canon(op))[:16].upper()
        op["id"] = rid
        led._append(op)
        out.record = f"{rid} in {led.path}"
    return out
