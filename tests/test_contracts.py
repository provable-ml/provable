"""Contract tests: the identities the certificates stand on. These are theorems of the
design, not implementation details -- they must hold for every model shape."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from provable import load, XGBAdapter
from provable.calculus import Spectrum


def _train(n=800, nf=5, depth=3, trees=40, base_score=None, y=None, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, nf)); X[:, -1] = (rng.random(n) < 0.4).astype(float)
    if y is None: y = X[:, 0] + (0.8 * (X[:, 1] >= 0.3) * X[:, -1] if nf >= 3 else 0.0) + 0.1 * rng.normal(size=n)
    p = {'max_depth': depth, 'eta': 0.2, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'seed': 1}
    if base_score is not None: p['base_score'] = base_score
    bst = xgb.train(p, xgb.DMatrix(X.astype(np.float32), label=y), num_boost_round=trees)
    return bst, X


@pytest.mark.parametrize('depth,trees,nf', [(1, 1, 1), (1, 5, 3), (2, 30, 4), (4, 60, 8), (6, 25, 5)])
def test_transport_identity_across_shapes(depth, trees, nf):
    bst, X = _train(nf=nf, depth=depth, trees=trees)
    ad, spec = load(booster=bst)
    d = np.abs(spec.evaluate(X[:200]) - ad.predict_margin(X[:200]))
    assert d.max() <= 1e-3, f'transport broke at depth={depth} trees={trees}: {d.max():.2e}'


def test_constant_model_is_empty_atom_only():
    bst, X = _train(trees=3, y=np.full(800, 2.5))
    ad, spec = load(booster=bst)
    assert all(not A for A in spec.atoms), 'constant model must transport to the empty atom alone'
    assert np.abs(spec.evaluate(X[:50]) - ad.predict_margin(X[:50])).max() <= 1e-5


@pytest.mark.parametrize('bs', [0.0, 0.25, 1.5, -0.7])
def test_base_score_parsed_and_in_identity(bs):
    bst, X = _train(base_score=bs, trees=20)
    ad, spec = load(booster=bst)
    assert abs(ad.base_score - bs) <= 1e-6
    assert np.abs(spec.evaluate(X[:100]) - ad.predict_margin(X[:100])).max() <= 1e-3


def test_float32_boundary_rows_route_identically():
    bst, X = _train(trees=40, depth=3)
    ad, spec = load(booster=bst)
    th = ad.thresholds()
    f = next(k for k in th if len(th[k]) >= 3)
    rows = np.repeat(X[:1], 3 * len(th[f][:8]), axis=0)
    r = 0
    for t in th[f][:8]:
        for v in (np.float32(t), np.nextafter(np.float32(t), np.float32(-np.inf)), np.nextafter(np.float32(t), np.float32(np.inf))):
            rows[r, f] = v; r += 1
    d = np.abs(spec.evaluate(rows) - ad.predict_margin(rows))
    assert d.max() <= 1e-3, f'boundary routing mismatch: {d.max():.2e}'


def test_family_selection_and_profile_contract():
    bst, X = _train(trees=50, depth=4)
    _, spec = load(booster=bst)
    fam = spec.family(feats=[1, 4])
    assert fam and all({1, 4} <= spec.support(A) for A in fam)
    inv = spec.family(involving=1, min_order=2)
    assert all(1 in spec.support(A) and len(spec.support(A)) >= 2 for A in inv)
    prof = spec.profile(fam, 1)
    thr1 = set(map(float, load(booster=bst)[0].thresholds()[1]))
    assert set(prof) <= thr1 and all(v > 0 for v in prof.values())


def test_transport_guard_refuses_above_cap():
    bst, _ = _train(trees=60, depth=6, nf=8)
    with pytest.raises(RuntimeError, match='max_atoms'):
        load(booster=bst, max_atoms=10)


def test_probe_values_edges():
    assert list(Spectrum._probe_values([])) == [0.0]
    assert list(Spectrum._probe_values([2.0])) == [1.0, 3.0]
    v = Spectrum._probe_values([1.0, 3.0, 7.0])
    assert len(v) == 4 and all(np.diff(v) > 0)


def test_append_nothing_is_identity():
    bst, X = _train(trees=25)
    ad, _ = load(booster=bst)
    XGBAdapter.save_raw_json(ad.append_rules({}), 'm_id.json')
    ad2 = XGBAdapter.reload('m_id.json')
    assert np.array_equal(ad.predict_margin(X[:200]), ad2.predict_margin(X[:200]))


def test_append_empty_support_shifts_base_score():
    bst, X = _train(trees=25)
    ad, _ = load(booster=bst)
    XGBAdapter.save_raw_json(ad.append_rules({frozenset(): 0.37}), 'm_shift.json')
    ad2 = XGBAdapter.reload('m_shift.json')
    d = ad.predict_margin(X[:200]) - ad2.predict_margin(X[:200])
    assert np.abs(d - 0.37).max() <= 1e-6


def test_append_single_rule_on_off_semantics():
    bst, X = _train(trees=25)
    ad, _ = load(booster=bst)
    A, c = frozenset({(0, 0.0), (4, 1.0)}), 0.5
    XGBAdapter.save_raw_json(ad.append_rules({A: c}), 'm_one.json')
    ad2 = XGBAdapter.reload('m_one.json')
    Xf = X[:400].astype(np.float32)
    on = (Xf[:, 0] >= np.float32(0.0)) & (Xf[:, 4] >= np.float32(1.0))
    F, G = ad.predict_margin(X[:400]), ad2.predict_margin(X[:400])
    assert np.array_equal(F[~on], G[~on]), 'off-rule rows must be bit-identical'
    assert np.abs((F[on] - G[on]) - c).max() <= 1e-6
