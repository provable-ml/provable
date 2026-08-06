"""Verifier mutation battery: every check must be individually killable. A verifier
whose checks cannot fail is decoration; each tamper below targets exactly one check
(re-hashing the bundle where needed so earlier checks stay green)."""
import os
import sys
import json
import copy
import hashlib
import subprocess
import numpy as np
import pytest
import xgboost as xgb
from provable import load, enforce, Policy, between, XGBAdapter
from provable.evidence import canonical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _reid(b):
    body = {k: v for k, v in b.items() if k != 'certificate_id'}
    b['certificate_id'] = 'PV-' + hashlib.sha256(canonical(body).encode()).hexdigest()[:16].upper()
    return b


def _run(bundle_path, model_path):
    r = subprocess.run([sys.executable, '-m', 'provable_verify', bundle_path, model_path],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'verify', 'src')))
    return r.returncode, r.stdout


@pytest.fixture(scope='module')
def dossier(tmp_path_factory):
    d = tmp_path_factory.mktemp('doss')
    rng = np.random.default_rng(9)
    X = rng.normal(0, 1, (1200, 4)); X[:, 3] = (rng.random(1200) < 0.4).astype(float)
    y = X[:, 0] + 0.9 * (X[:, 1] >= 0.2) * X[:, 3] + 0.15 * rng.normal(size=1200)
    bst = xgb.train({'max_depth': 3, 'eta': 0.15, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'seed': 1},
                    xgb.DMatrix(X.astype(np.float32), label=y, feature_names=['p', 'q', 'r', 'g']), num_boost_round=80)
    mp = str(d / 'm.json'); bst.save_model(mp)
    out = str(d / 'm2.json')
    enforce(mp, Policy([between('q', 'g')]), X[:900], X[900:], out, y_holdout=y[900:], out_dir=str(d))
    return str(d / 'evidence.json'), out, str(d)


def test_clean_dossier_passes(dossier):
    ev, m2, _ = dossier
    code, out = _run(ev, m2)
    assert code == 0 and 'VERIFICATION: PASS' in out


def test_check1_bundle_hash(dossier):
    ev, m2, d = dossier
    b = json.load(open(ev)); b['action']['holdout_delta'] = 0.0
    p = os.path.join(d, 'ev_c1.json'); json.dump(b, open(p, 'w'))
    code, out = _run(p, m2)
    assert code != 0 and '[FAIL] 1' in out


def test_check2_model_hash(dossier):
    ev, m2, d = dossier
    raw = json.load(open(m2)); p = os.path.join(d, 'm_c2.json')
    raw['learner']['attributes']['tampered'] = '1'; json.dump(raw, open(p, 'w'))
    code, out = _run(ev, p)
    assert code != 0 and '[FAIL] 2' in out


def test_check3_policy_hash(dossier):
    ev, m2, d = dossier
    b = json.load(open(ev)); b['policy']['tolerance'] = 999.0
    _reid(b)
    p = os.path.join(d, 'ev_c3.json'); json.dump(b, open(p, 'w'))
    code, out = _run(p, m2)
    assert code != 0 and '[FAIL] 3' in out and '[PASS] 1' in out


def test_check45_resurrected_interaction(dossier):
    ev, m2, d = dossier
    ad = XGBAdapter(path=m2)
    th = ad.thresholds()
    A = frozenset({(1, float(th[1][0])), (3, float(th[3][0]))})
    p = os.path.join(d, 'm_c45.json')
    XGBAdapter.save_raw_json(ad.append_rules({A: -0.6}), p)
    b = json.load(open(ev)); b['action']['model_out_sha256'] = XGBAdapter.sha256(p)
    _reid(b)
    pb = os.path.join(d, 'ev_c45.json'); json.dump(b, open(pb, 'w'))
    code, out = _run(pb, p)
    assert code != 0 and '[FAIL] 4' in out and '[FAIL] 5' in out and '[FAIL] 6' in out and '[PASS] 2' in out


def test_check7_limits_stripped(dossier):
    ev, m2, d = dossier
    b = json.load(open(ev)); b['limits'] = []
    _reid(b)
    p = os.path.join(d, 'ev_c7.json'); json.dump(b, open(p, 'w'))
    code, out = _run(p, m2)
    assert code != 0 and '[FAIL] 7' in out
