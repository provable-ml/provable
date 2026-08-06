"""Protocol tests: policy resolution and refusal codes, canonical hashing and tamper
evidence, verdict grammar, the coded R-07 refusal, sieve determinism."""
import json
import copy
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from provable import Policy, between, involving, audit, load, CertifiedDiscovery
from provable.evidence import make_bundle, verify_id, bundle_hash, canonical, LIMITS, render_text, render_markdown


def _model(tmp='m_pol.json'):
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (900, 4)); X[:, 3] = (rng.random(900) < 0.5).astype(float)
    y = X[:, 0] + 0.6 * (X[:, 1] >= 0) * X[:, 3] + 0.1 * rng.normal(size=900)
    bst = xgb.train({'max_depth': 3, 'eta': 0.2, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'seed': 1},
                    xgb.DMatrix(X.astype(np.float32), label=y, feature_names=['a', 'b', 'c', 'g']), num_boost_round=30)
    bst.save_model(tmp)
    return tmp, X


def test_policy_unknown_feature_is_refusal_not_exception():
    path, X = _model()
    b = audit(path, Policy([between('a', 'NOPE')]), X[:200])
    assert b['refusals'] and b['refusals'][0]['code'] == 'R-05'
    assert b['verdict']['state'] in ('PASS', 'DECLINED')


def test_policy_canonicalization_stable():
    p1 = Policy([between('b', 'a')], name='p'); p2 = Policy([between('a', 'b')], name='p')
    assert p1.sha256() == p2.sha256(), 'between() must sort features'
    assert Policy.from_dict(json.loads(json.dumps(p1.to_dict()))).sha256() == p1.sha256()


def test_bundle_id_deterministic_and_order_free():
    m = {'path': 'x', 'sha256': 'h', 'n_trees': 1, 'n_atoms': 1}
    b1 = make_bundle('audit', m, {'sha256': 'p'}, [], None, {'state': 'PASS', 'statement': 's'}, [], [], {'e': 1})
    b2 = make_bundle('audit', dict(reversed(list(m.items()))), {'sha256': 'p'}, [], None, {'statement': 's', 'state': 'PASS'}, [], [], {'e': 1})
    body = lambda b: {k: v for k, v in b.items() if k not in ('certificate_id', 'created_utc')}
    assert bundle_hash(body(b1)) == bundle_hash(body(b2))
    assert verify_id(b1) and verify_id(b2)


def test_any_single_field_mutation_breaks_id():
    m = {'path': 'x', 'sha256': 'h', 'n_trees': 1, 'n_atoms': 1}
    b = make_bundle('enforce', m, {'sha256': 'p'}, [{'clause': 'c', 'n_atoms': 2, 'data_mass': 0.1, 'share_of_var': 0.05, 'profile_top': []}],
                    {'mode': 'excise', 'atoms_removed': 2, 'trees_appended': 2, 'structural_residual': 1e-9, 'grid_probe_max': 1e-6,
                     'stock_diff_on': 1e-6, 'stock_diff_off': 0.0, 'holdout_mse_before': 1.0, 'holdout_mse_after': 1.1,
                     'holdout_delta': 0.1, 'holdout_delta_ci': 0.01, 'model_out': 'm', 'model_out_sha256': 'z'},
                    {'state': 'PASS', 'statement': 's'}, [], [{'quantity': 'q', 'predicted': 1.0, 'observed': 1.05, 'status': 'ok'}], {'e': 1})
    assert verify_id(b)
    muts = [('verdict', lambda x: x['verdict'].__setitem__('state', 'FINDINGS')),
            ('action', lambda x: x['action'].__setitem__('structural_residual', 0.5)),
            ('findings', lambda x: x['findings'][0].__setitem__('data_mass', 0.0)),
            ('limits', lambda x: x.__setitem__('limits', [])),
            ('pvo', lambda x: x['predicted_vs_observed'][0].__setitem__('observed', 0.0))]
    for name, mut in muts:
        bb = copy.deepcopy(b); mut(bb)
        assert not verify_id(bb), f'mutation of {name} not caught by certificate id'


def test_limits_always_present_and_rendered():
    m = {'path': 'x', 'sha256': 'h', 'n_trees': 1, 'n_atoms': 1}
    b = make_bundle('audit', m, {'sha256': 'p'}, [], None, {'state': 'PASS', 'statement': 's'}, [], [], {})
    assert b['limits'] == LIMITS and len(LIMITS) >= 4
    txt = render_text(b)
    for section in ('VERDICT', 'DOES NOT ESTABLISH', 'WHERE THE INSTRUMENT DECLINED', 'VERIFY WITHOUT TRUSTING'):
        assert section in txt


def test_nan_input_is_coded_refusal():
    path, X = _model()
    Xn = X[:100].copy(); Xn[3, 1] = np.nan
    b = audit(path, Policy([between('a', 'g')]), Xn)
    assert b['verdict']['state'] == 'DECLINED'
    assert any(r['code'] == 'R-07' for r in b['refusals'])


def test_verdict_grammar_closed_set():
    path, X = _model()
    b = audit(path, Policy([between('b', 'g')]), X[:300])
    assert b['verdict']['state'] in ('PASS', 'FINDINGS', 'DECLINED')


def test_sieve_deterministic_given_seed():
    rng = np.random.default_rng(5)
    X = pd.DataFrame({'u': rng.uniform(0, 1, 2500), 'v': rng.normal(0, 1, 2500), 'g': (rng.random(2500) < 0.5).astype(float)})
    y = 1.2 * (X['u'] >= 0.6).values + 0.8 * X['g'].values * (X['v'] >= 0).values + 0.3 * rng.normal(size=2500)
    r1 = CertifiedDiscovery(seed=3, verbose=False, xgb_params={'n_estimators': 120}).run(X, y)
    r2 = CertifiedDiscovery(seed=3, verbose=False, xgb_params={'n_estimators': 120}).run(X, y)
    assert [c['atom'] for c in r1['claims']] == [c['atom'] for c in r2['claims']]
    assert r1['certification']['k'] == r2['certification']['k'] > 0


def test_markdown_certificate_sections_and_escaping():
    m = {'path': 'x', 'sha256': 'h', 'n_trees': 1, 'n_atoms': 1}
    b = make_bundle('audit', m, {'sha256': 'p'}, [{'clause': 'between(a_b|c, g*)', 'n_atoms': 2, 'data_mass': 0.1, 'share_of_var': 0.05, 'profile_top': ['a_b|c>=1 (|c| 0.2)']}],
                    None, {'state': 'FINDINGS', 'statement': 's'}, [], [], {})
    md = render_markdown(b)
    for sec in ('## Verdict', '## Findings', '## Where the instrument declined', '## What this does not establish', '## Verify without trusting'):
        assert sec in md
    assert 'a\\_b\\|c' in md, 'feature-name markdown specials must be escaped'
    assert b['certificate_id'] in md


def test_html_certificate_sections_seal_escaping():
    from provable.report_html import render_html
    m = {'path': 'x', 'sha256': 'h' * 64, 'n_trees': 1, 'n_atoms': 1}
    b = make_bundle('enforce', m, {'sha256': 'p' * 64, 'name': 'pol'},
                    [{'clause': 'between(a<script>, g)', 'n_atoms': 2, 'data_mass': 0.1, 'share_of_var': 0.05, 'profile_top': []}],
                    {'mode': 'excise', 'atoms_removed': 2, 'trees_appended': 2, 'structural_residual': 1e-9, 'grid_probe_max': 1e-6,
                     'stock_diff_on': 1e-6, 'stock_diff_off': 0.0, 'holdout_mse_before': 1.0, 'holdout_mse_after': 1.1,
                     'holdout_delta': 0.1, 'holdout_delta_ci': 0.01, 'model_out': 'm2.json', 'model_out_sha256': 'z' * 64},
                    {'state': 'PASS', 'statement': 's', 'establishes': 'e', 'certified_absent': ['age × group_A']}, [],
                    [{'quantity': 'price', 'predicted': 1.0, 'observed': 1.05, 'status': 'within 25%'}], {'provable': '0.1.0', 'xgboost': 'x'})
    h = render_html(b)
    assert 'no rules combining' in h and 'age × group_A' in h, 'hero must name the certified-absent families'
    assert 'Tolerance' in h and 'Evidence panel' in h and ("class='notes'" in h or 'class="notes"' in h)
    assert '<script>' not in h and '&lt;script&gt;' in h, 'dynamic strings must be html-escaped'
    for frag in ('Verdict', 'Does not establish', 'Standing limits', 'Verify without trusting', 'grades itself', 'sealring', b['certificate_id']):
        assert frag in h
    assert ('class="qr"' in h) or ('QR unavailable' in h)
    assert 'bit-identical' in h


def test_sieve_v2_flags_off_schema_and_claims_stable():
    rng = np.random.default_rng(5)
    X = pd.DataFrame({'u': rng.uniform(0, 1, 2500), 'v': rng.normal(0, 1, 2500), 'g': (rng.random(2500) < 0.5).astype(float)})
    y = 1.2 * (X['u'] >= 0.6).values + 0.8 * X['g'].values * (X['v'] >= 0).values + 0.3 * rng.normal(size=2500)
    r = CertifiedDiscovery(seed=3, verbose=False, xgb_params={'n_estimators': 120}).run(X, y)
    assert set(r['setup']['dict_counts']) == {'guideline', 'cut', 'cart', 'onehot', 'harvest', 'conj'}
    assert r['certification']['k'] > 0


def test_sieve_v2_vocabulary_deterministic_and_low_side_claimable():
    rng = np.random.default_rng(9)
    n = 2500
    X = pd.DataFrame({'a': rng.normal(0, 1, n), 'b': rng.normal(0, 1, n), 'c': rng.uniform(0, 1, n)})
    y = 0.9 * (X['a'] <= X['a'].quantile(0.2)).values - 0.8 * ((X['b'] > X['b'].quantile(0.4)) & (X['b'] <= X['b'].quantile(0.6))).values + 0.3 * rng.normal(size=n)
    kw = dict(seed=3, verbose=False, xgb_params={'n_estimators': 120}, le_cuts=True, band_conj=True, n_path_harvest=20, path_depth=3, core_dedup_r=0.95)
    r1 = CertifiedDiscovery(**kw).run(X, y)
    r2 = CertifiedDiscovery(**kw).run(X, y)
    assert [c['atom'] for c in r1['claims']] == [c['atom'] for c in r2['claims']]
    assert 'path' in r1['setup']['dict_counts']
    low = [c for c in r1['claims'] if '<=' in c['atom'] and c['effect'] > 0]
    assert low, [c['atom'] for c in r1['claims']]
