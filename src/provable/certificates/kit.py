"""Theorem-fixed kit functions. Pure functions, register-cited.

khat is the step-up count, khat_i its leave-one-in variant, tau_i the
transfer level in (0, q]. The valid same-rows confirmation level is
alpha * tau_i. Screen-then-gate: screening at |z| > c passes nulls at
rate s = 2 Phibar(c); a gate at nominal level q on the screened rows
realizes min(1, q/s); the exact repair gates at level q*s, equivalently
divides screened p-values by s. The selective lower confidence limit on
a certified coordinate is |z_i| - Phibar_inv(q khat / (2m)), nonnegative
on the certified set by construction. Budgets add across certificate
sets.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def khat(p, q: float) -> int:
    """max{ k in 0..m : #{j : p_j <= q k / m} >= k }."""
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return 0
    best = 0
    for k in range(1, m + 1):
        if int((p <= q * k / m).sum()) >= k:
            best = k
    return best


def khat_leave_one_in(p, q: float, i: int) -> int:
    """khat with p_i set to 0."""
    p2 = np.array(p, dtype=float)
    p2[i] = 0.0
    return khat(p2, q)


def rejected(p, q: float) -> np.ndarray:
    """Indices with p_j <= q khat / m."""
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=int)
    kh = khat(p, q)
    return np.flatnonzero(p <= q * kh / m)


def tau(p, q: float, i: int) -> float:
    """tau_i = q khat_i / m, in (0, q]. i is rejected iff p_i <= tau_i,
    and khat == khat_i on that event."""
    p = np.asarray(p, dtype=float)
    m = p.size
    return q * khat_leave_one_in(p, q, i) / m


def confirmation_level(alpha: float, tau_i: float) -> float:
    """Valid same-rows confirmation level: alpha * tau_i."""
    return float(alpha) * float(tau_i)


def screen_pass_rate(c: float) -> float:
    """Null pass rate of the screen |z| > c: s = 2 Phibar(c)."""
    return float(2.0 * norm.sf(c))


def realized_level(q: float, s: float) -> float:
    """Realized level of a nominal-q gate on rows already screened at
    null pass rate s: min(1, q/s)."""
    return float(min(1.0, q / s))


def repaired_level(q: float, s: float) -> float:
    """Exact repair: gate at level q*s."""
    return float(q * s)


def repair_pvalues(p, s: float) -> np.ndarray:
    """Equivalent repair: divide screened p-values by s."""
    return np.asarray(p, dtype=float) / float(s)


def selective_lcl(z, p, q: float) -> np.ndarray:
    """LCL_i = |z_i| - Phibar_inv(q khat / (2m)). Nonnegative on the
    certified set by construction."""
    z = np.asarray(z, dtype=float)
    p = np.asarray(p, dtype=float)
    m = p.size
    kh = khat(p, q)
    if kh == 0:
        return np.full(m, -np.inf)
    thr = float(norm.isf(q * kh / (2.0 * m)))
    return np.abs(z) - thr


def additive_budget(levels) -> float:
    """Additive budget across certificate sets: sum of levels."""
    return float(np.sum(np.asarray(list(levels), dtype=float)))
