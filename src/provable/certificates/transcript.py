"""Transcript accounting for a mediated holdout.

A holdout reached only through a custodian that answers from an alphabet
fixed when the holdout is sealed can price its certificate by the
codelength of the custodian's own log. The pieces, all pure functions:

  codes        per-answer codelengths that satisfy Kraft's inequality at
               every step: the Krichevsky-Trofimov code for yes/no answers
               (no declared number of queries) and the fixed-rate code
               (log2 of the alphabet size per answer)
  bars         the level delta 2^-L at which a certificate priced by L bits
               is taken, its one-sided Gaussian bar z_L = Phibar^-1(delta
               2^-L) computed in log space, and for a declared range two
               exact one-sided bounds: the Hoeffding width and the betting
               bound (a mixture of bets on the values, Markov's inequality
               at fixed n; Waudby-Smith and Ramdas, JRSS-B 2024)
  rows         the clean share s(L) = (z_0 / z_L)^2: a holdout that has
               spent L bits certifies with the width of a clean holdout
               holding that fraction of its rows; an exam slice of share s
               beats the transcript certificate exactly when s > s(L)
  checks       the transcript weights of Guess and Check (Rogers, Roth,
               Smith, Srebro, Thakkar, Woodworth, AISTATS 2020, Algorithm
               1): the per-query level beta c_{i-1} c_f / nu with
               c_j = 6 / (pi^2 (j+1)^2), and its Hoeffding tolerance

The certificate these price (the Kraft union bound over transcripts): if
every history fixes codelengths with sum_a 2^-l(a | history) <= 1, every
choice of the analyst is a function of their own rows, independent
randomness and past answers, and a single model fixed in advance has
P(D(f) > c_gamma(f)) <= gamma, then P(D(f_hat) > c_{delta 2^-L}(f_hat))
<= delta for every strategy and every stopping rule, L the realized
codelength. The leaves of the analyst's strategy tree carry Kraft weights
summing to at most one (kraft_leaves enumerates them), and the union bound
over leaves gives the statement. Cited in substance: the transcript
union bound of Guess and Check (Rogers et al. 2020, Theorem 3.2) and the
description-length argument of the Ladder (Blum and Hardt, ICML 2015).
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

KT = "kt"
FIXED = "fixed"
CODES = (KT, FIXED)
LN2 = math.log(2.0)


# ------------------------------------------------------------------ codes

def kt_bits(k: int, r: int) -> float:
    """Closed-form Krichevsky-Trofimov codelength of a yes/no sequence of
    length k with r yeses, in bits:
    L = -log2[ Gamma(r + 1/2) Gamma(k - r + 1/2) / (pi Gamma(k + 1)) ].
    Equal to the sum of the sequential answer codelengths in any order."""
    k, r = int(k), int(r)
    if k < 0 or r < 0 or r > k:
        raise ValueError(f"kt_bits needs 0 <= r <= k, got k={k}, r={r}")
    if k == 0:
        return 0.0
    return float(-(gammaln(r + 0.5) + gammaln(k - r + 0.5) - math.log(math.pi) - gammaln(k + 1)) / LN2)


def kt_answer_bits(t: int, r: int, yes: bool) -> float:
    """Sequential KT codelength of the next answer after t answers of
    which r were yes: P(yes) = (r + 1/2) / (t + 1)."""
    t, r = int(t), int(r)
    if t < 0 or r < 0 or r > t:
        raise ValueError(f"kt_answer_bits needs 0 <= r <= t, got t={t}, r={r}")
    p = (r + 0.5) / (t + 1.0) if yes else (t - r + 0.5) / (t + 1.0)
    return float(-math.log2(p))


def fixed_answer_bits(alphabet: int = 2) -> float:
    """Fixed-rate codelength: log2 of the alphabet size per answer."""
    if int(alphabet) < 1:
        raise ValueError("alphabet size must be at least 1")
    return float(math.log2(int(alphabet)))


def answer_bits(code: str, t: int, r: int, yes: bool) -> float:
    """Codelength of the next yes/no answer under a code fixed at sealing."""
    if code == KT:
        return kt_answer_bits(t, r, yes)
    if code == FIXED:
        return fixed_answer_bits(2)
    raise ValueError(f"unknown code {code!r}; codes are {CODES}")


def sequence_bits(answers: Iterable[bool], code: str = KT) -> float:
    """Total codelength of a yes/no answer sequence, summed step by step."""
    t = r = 0
    total = 0.0
    for a in answers:
        total += answer_bits(code, t, r, bool(a))
        t += 1
        r += int(bool(a))
    return float(total)


def kraft_leaves(code: str, stop: Callable[[Tuple[bool, ...]], bool], depth: int) -> float:
    """Sum of 2^-L over the leaves of a yes/no strategy tree: the analyst
    stops at a history h when stop(h) is true, the tree is truncated at
    the given depth, and L is the codelength of the path to the leaf. At
    most one for every stopping rule (Kraft at each step, induction on
    depth); exactly one for a finite tree."""
    total = 0.0
    stack = [((), 0.0)]
    while stack:
        h, L = stack.pop()
        if len(h) >= depth or stop(h):
            total += 2.0 ** (-L)
            continue
        t, r = len(h), sum(h)
        for a in (True, False):
            stack.append((h + (a,), L + answer_bits(code, t, r, a)))
    return float(total)


# ------------------------------------------------------------------ bars

def level(delta: float, bits: float) -> float:
    """The level at which a certificate priced by L bits is taken:
    delta 2^-L (underflows to 0.0 for very large L; the bars below work
    in log space)."""
    return float(delta) * 2.0 ** (-float(bits))


def inflation(bits: float) -> float:
    """Error-rate multiplier of an unpriced test on rows seen through an
    L-bit transcript: 2^L (infinite past the float range). Bits are log2
    of the inflation."""
    b = float(bits)
    return math.inf if b >= 1024.0 else float(2.0 ** b)


def _isf_log(log_p: float) -> float:
    """Phibar^-1(exp(log_p)) for log_p <= log(1/2), stable far in the tail:
    scipy's ndtri_exp, or Newton steps on log Phibar with the hazard
    computed in log space."""
    try:
        from scipy.special import ndtri_exp
        return float(-ndtri_exp(log_p))
    except ImportError:
        return _isf_log_newton(log_p)


def _isf_log_newton(log_p: float) -> float:
    p = math.exp(log_p)
    if p > 1e-300:
        return float(norm.isf(p))
    y = -2.0 * log_p
    z = math.sqrt(y - math.log(y) - math.log(2 * math.pi))
    for _ in range(8):
        step = (norm.logsf(z) - log_p) / math.exp(norm.logpdf(z) - norm.logsf(z))
        z += step
        if abs(step) < 1e-13 * z:
            break
    return float(z)


def z_bar(delta: float, bits: float = 0.0) -> float:
    """One-sided Gaussian bar z_L = Phibar^-1(delta 2^-L), in log space."""
    delta = float(delta)
    if not (0.0 < delta < 0.5):
        raise ValueError("delta must lie in (0, 0.5)")
    if float(bits) < 0:
        raise ValueError("bits must be nonnegative")
    return _isf_log(math.log(delta) - float(bits) * LN2)


def hoeffding_width(delta: float, bits: float, n: int, span: float) -> float:
    """One-sided Hoeffding width for the mean of n values in a range of
    the given span, at level delta 2^-L: span sqrt((ln(1/delta) + L ln 2)
    / (2 n)). Exact for a declared range."""
    return float(span) * math.sqrt((-math.log(float(delta)) + float(bits) * LN2) / (2.0 * int(n)))


def betting_grid(n: int) -> np.ndarray:
    """The bets of the betting bound, fixed by n alone: lambda = 2^(-j/2)
    for j = 0 .. J-1, J = max(4, ceil(log2 n) - 2), half-octave steps from
    1 down to about 3/sqrt(n), the scale at which a mean of n values in
    [0, 1] is resolved."""
    J = max(4, int(math.ceil(math.log2(max(int(n), 2)))) - 2)
    return 2.0 ** (-0.5 * np.arange(J))


def betting_upper(u: np.ndarray, delta: float, bits: float = 0.0, iters: int = 48) -> float:
    """Exact upper confidence bound on the mean of n values in [0, 1] at
    level delta 2^-L: the largest m whose capital
    K(m) = (1/J) sum_j prod_i (1 + lambda_j (m - u_i))
    stays below 2^L / delta, the bets lambda_j from betting_grid(n). For
    independent values with mean mu, K(mu) is an average of products of
    factors with expectation one, so E K(mu) = 1 and Markov's inequality
    gives P(K(mu) >= 2^L / delta) <= delta 2^-L; K is increasing in m, so
    the bound falls below mu with at most that probability. The search
    returns the upper end of its bracket, so rounding errs high."""
    u = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
    n = u.size
    if n == 0:
        raise ValueError("betting_upper needs at least one value")
    lam = betting_grid(n)
    thr = -math.log(float(delta)) + float(bits) * LN2 + math.log(lam.size)
    lo, hi = float(u.mean()), 1.0

    def log_capital(m: float) -> float:
        terms = np.log1p(lam[:, None] * (m - u[None, :])).sum(axis=1)
        top = terms.max()
        return float(top + math.log(np.exp(terms - top).sum())) if np.isfinite(top) else -math.inf

    if log_capital(hi) < thr:
        return 1.0
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if log_capital(mid) >= thr:
            hi = mid
        else:
            lo = mid
    return float(hi)


def betting_bound(values, lo: float, hi: float, delta: float, bits: float = 0.0, side: str = "upper") -> float:
    """The betting bound on the mean of values in [lo, hi] at level
    delta 2^-L, in the values' units: side="upper" bounds the mean from
    above, side="lower" from below (the upper bound of hi + lo - values).
    Exact for any independent values in the range (see betting_upper)."""
    v = np.asarray(values, dtype=float).ravel()
    span = float(hi) - float(lo)
    if not span > 0:
        raise ValueError("betting_bound needs hi > lo")
    if side == "upper":
        return float(lo) + span * betting_upper((v - lo) / span, delta, bits)
    if side == "lower":
        return float(hi) - span * betting_upper((hi - v) / span, delta, bits)
    raise ValueError("side is 'upper' or 'lower'")


def clean_share(delta: float, bits: float) -> float:
    """s(L) = (z_0 / z_L)^2: the fraction of its rows a clean holdout needs
    to certify with the width of a holdout that has spent L bits."""
    return float((z_bar(delta, 0.0) / z_bar(delta, bits)) ** 2)


def bits_for_share(delta: float, share: float) -> float:
    """The codelength at which the clean share falls to the given value:
    the inverse of clean_share, L = log2(delta / Phibar(z_0 / sqrt(s)))."""
    share = float(share)
    if not (0.0 < share <= 1.0):
        raise ValueError("share must lie in (0, 1]")
    z = z_bar(delta, 0.0) / math.sqrt(share)
    return float((math.log(float(delta)) - norm.logsf(z)) / LN2)


def exam_wins(delta: float, bits: float, exam_share: float) -> bool:
    """An exam slice holding a share s of the rows, opened once at the
    clean bar, certifies more narrowly than the transcript certificate at
    L bits exactly when s > s(L) (tuning-row change ignored)."""
    return bool(float(exam_share) > clean_share(delta, bits))


# ------------------------------------------------------------------ Guess and Check

def gnc_weight(j: int) -> float:
    """c_j = 6 / (pi^2 (j + 1)^2), j >= 0; sums to one over j."""
    return float(6.0 / (math.pi ** 2 * (int(j) + 1) ** 2))


def gnc_grid_size(gamma: float) -> int:
    """Number of values a failure can release on the grid of step gamma
    over [0, 1]: multiples of gamma from 0 to ceil(1/gamma) gamma."""
    return int(math.ceil(1.0 / float(gamma) - 1e-12)) + 1


def gnc_transcripts(i: int, gammas: Sequence[float]) -> int:
    """nu = C(i-1, f) prod_j N_j, an exact integer: the transcripts that
    reach query i with f = len(gammas) failures, N_j the grid size of
    failure j. The count uses the exact grid size, at least the paper's
    1/gamma_j."""
    nu = math.comb(int(i) - 1, len(gammas))
    for g in gammas:
        nu *= gnc_grid_size(g)
    return nu


def gnc_bits(i: int, gammas: Sequence[float]) -> float:
    """The codelength behind beta_i, in log space: beta_i = beta 2^-L_i
    with L_i = log2(nu / (c_{i-1} c_f))."""
    return float(math.log2(gnc_transcripts(i, gammas)) - math.log2(gnc_weight(int(i) - 1)) - math.log2(gnc_weight(len(gammas))))


def gnc_level(beta: float, i: int, gammas: Sequence[float]) -> float:
    """beta_i = beta c_{i-1} c_f / nu (Algorithm 1), as beta 2^-L_i
    (underflows to 0.0 for very long transcripts; the tolerance below is
    computed from the bits)."""
    return float(beta) * 2.0 ** (-gnc_bits(i, gammas))


def hoeffding_tol(level_i: float, n: int) -> float:
    """Two-sided Hoeffding tolerance at level beta_i for the mean of n
    values in [0, 1]: sqrt(ln(2 / beta_i) / (2 n)) (the paper's simple
    HoldoutTol)."""
    if float(level_i) <= 0.0:
        return math.inf
    return float(math.sqrt(math.log(2.0 / float(level_i)) / (2.0 * int(n))))


def hoeffding_tol_bits(beta: float, bits: float, n: int) -> float:
    """The same tolerance at beta_i = beta 2^-bits, computed from the bits:
    sqrt((ln 2 - ln beta + bits ln 2) / (2 n)); finite where beta_i
    underflows."""
    return float(math.sqrt((math.log(2.0) - math.log(float(beta)) + float(bits) * LN2) / (2.0 * int(n))))


def gnc_round(value: float, gamma: float) -> float:
    """[a]_gamma: the nearest multiple of gamma, kept on the grid over [0, 1]."""
    k = int(round(float(value) / float(gamma)))
    k = min(max(k, 0), gnc_grid_size(gamma) - 1)
    return float(k * float(gamma))
