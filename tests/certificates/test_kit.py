import numpy as np
import pytest
from scipy.stats import norm

from provable.certificates import kit


def brute_force_step_up(p, q):
    p = np.sort(np.asarray(p, dtype=float))
    m = p.size
    kh = 0
    for k in range(1, m + 1):
        if p[k - 1] <= q * k / m:
            kh = k
    return kh


def random_pvector(rng, m):
    kind = rng.integers(0, 4)
    if kind == 0:
        p = rng.uniform(0, 1, m)
    elif kind == 1:
        p = np.concatenate([rng.uniform(0, 0.02, max(1, m // 4)), rng.uniform(0, 1, m - max(1, m // 4))])[:m]
    elif kind == 2:
        p = np.round(rng.uniform(0, 1, m), 2)
    else:
        z = rng.normal(rng.uniform(0, 3), 1, m)
        p = 2 * norm.sf(np.abs(z))
    rng.shuffle(p)
    return p


def test_khat_matches_brute_force_10k():
    rng = np.random.default_rng(20260909)
    for _ in range(10000):
        m = int(rng.integers(1, 61))
        q = float(rng.choice([0.01, 0.05, 0.1, 0.2]))
        p = random_pvector(rng, m)
        assert kit.khat(p, q) == brute_force_step_up(p, q)


def test_rejected_set_size_equals_khat():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        m = int(rng.integers(1, 41))
        q = float(rng.choice([0.05, 0.1, 0.2]))
        p = random_pvector(rng, m)
        kh = kit.khat(p, q)
        assert len(kit.rejected(p, q)) == kh


def test_leave_one_in_property():
    rng = np.random.default_rng(99)
    checked_rejections = 0
    for _ in range(1500):
        m = int(rng.integers(2, 31))
        q = float(rng.choice([0.05, 0.1, 0.2]))
        p = random_pvector(rng, m)
        kh = kit.khat(p, q)
        rej = set(kit.rejected(p, q).tolist())
        for i in range(m):
            khi = kit.khat_leave_one_in(p, q, i)
            ti = kit.tau(p, q, i)
            assert 0.0 < ti <= q + 1e-15
            assert abs(ti - q * khi / m) < 1e-15
            in_rej = i in rej
            assert in_rej == (p[i] <= ti + 1e-15)
            if in_rej:
                assert kh == khi
                checked_rejections += 1
    assert checked_rejections > 1000


def test_selective_lcl_nonnegative_on_certified():
    rng = np.random.default_rng(123)
    hits = 0
    for _ in range(500):
        m = int(rng.integers(5, 80))
        q = 0.1
        mu = np.zeros(m)
        k1 = int(rng.integers(0, m // 2 + 1))
        mu[:k1] = rng.uniform(2, 5, k1)
        z = rng.normal(mu, 1.0)
        p = 2 * norm.sf(np.abs(z))
        lcl = kit.selective_lcl(z, p, q)
        rej = kit.rejected(p, q)
        if len(rej):
            assert np.all(lcl[rej] >= -1e-12)
            hits += 1
    assert hits > 200


def test_screen_repair_equivalence_and_level():
    rng = np.random.default_rng(2026)
    c, q = 2.0, 0.1
    s = kit.screen_pass_rate(c)
    assert abs(s - 2 * norm.sf(2.0)) < 1e-15
    assert abs(kit.realized_level(q, s) - min(1.0, q / s)) < 1e-15
    assert abs(kit.repaired_level(q, s) - q * s) < 1e-15
    for _ in range(200):
        m = int(rng.integers(3, 40))
        p = rng.uniform(0, s, m)
        r1 = set(kit.rejected(kit.repair_pvalues(p, s), q).tolist())
        r2 = set(kit.rejected(p, kit.repaired_level(q, s)).tolist())
        assert r1 == r2


def test_screen_repair_reproduces_nominal_level():
    rng = np.random.default_rng(31415)
    m, m0, c, q = 60, 50, 1.5, 0.1
    s = kit.screen_pass_rate(c)
    reps = 3000
    fdp_naive, fdp_rep = [], []
    for _ in range(reps):
        z = np.concatenate([rng.normal(0, 1, m0), rng.normal(3.5, 1, m - m0)])
        keep = np.abs(z) > c
        p = 2 * norm.sf(np.abs(z[keep]))
        isnull = np.arange(m)[keep] < m0
        rn = kit.rejected(p, q)
        rr = kit.rejected(kit.repair_pvalues(p, s), q)
        fdp_naive.append(isnull[rn].sum() / max(1, len(rn)))
        fdp_rep.append(isnull[rr].sum() / max(1, len(rr)))
    fdr_naive, fdr_rep = float(np.mean(fdp_naive)), float(np.mean(fdp_rep))
    assert fdr_rep <= q + 0.02
    assert fdr_naive > q + 0.05


def test_additive_budget():
    assert kit.additive_budget([0.1, 0.05, 0.025]) == pytest.approx(0.175)


def test_khat_empty_and_confirmation():
    assert kit.khat([], 0.1) == 0
    assert kit.confirmation_level(0.05, 0.08) == pytest.approx(0.004)
