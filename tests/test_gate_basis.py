"""Gate basis: the forward core is reduced to a basis on S2 before the single OLS pass.

The partial-variance guard conditions the forward core on S1 only. A member admitted on
S1 can therefore be constant on S2, or an exact combination of the intercept and other
members there. Three designs that reach this:

  twins      a threshold atom and its <= twin. Rows tied at the threshold break the
             collinearity on S1, so both twins are admitted. With no tie on S2 the pair
             is exactly collinear with the intercept.
  partition  one-hot atoms of a categorical feature. A level present only on S1 rows
             means the remaining one-hots do not partition S1, so all are admitted. On S2
             they sum to the intercept.
  absent     a one-hot atom whose level has no S2 row. It is a zero column on S2.

Without the reduction the OLS returns a minimum-norm solution, the partial effects and
p-values of such members are not identified, and both members of a pair can pass the
gate. With it (cert_basis_tol, on by default) the member is dropped before the pass, is
not tested, and is reported. A core of full column rank on S2 is untouched.
"""
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import SingularMatrixWarning
import provable as pv
from provable.sieve import CertifiedDiscovery

N = 1600
FAST = dict(n_estimators=150, max_depth=3)
PLAIN = dict(interactions=False, n_harvest=0, cart_cuts=False)
Q = 0.10


def _split(seed, n=N, test_frac=0.25, s2_frac=0.5):
    rng = np.random.default_rng(4000 + seed); perm = rng.permutation(n)
    n_test = int(round(test_frac * n)); tr = perm[n_test:]
    n2 = int(round(s2_frac * len(tr)))
    return tr[:n2], tr[n2:]


def _run(cd, X, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        cd.run(X, y)
    return any(issubclass(x.category, SingularMatrixWarning) for x in w)


def _twins_world(seed=0):
    rng = np.random.default_rng(100 + seed)
    _, i1 = _split(seed)
    x1 = np.round(np.exp(rng.normal(1.4, 0.6, N)), 1)
    x1[x1 == 4.0] = 4.1
    tie_rows = rng.choice(i1, 10, replace=False); x1[tie_rows] = 4.0
    x2 = rng.normal(size=N)
    y = 1.0 * (x1 >= 4.0) + 0.3 * x2 + rng.normal(size=N); y[tie_rows] += 1.5
    return pd.DataFrame(dict(x1=x1, x2=x2)), y


def _levels_world(seed=0, only_s1='D', n_only=5):
    rng = np.random.default_rng(200 + seed)
    _, i1 = _split(seed)
    g = rng.choice(['A', 'B', 'C'], N)
    rows = rng.choice(i1, n_only, replace=False); g[rows] = only_s1
    x = rng.normal(size=N)
    eff = {'A': 0.8, 'B': 0.5, 'C': -0.6, only_s1: 3.0}
    y = np.array([eff[v] for v in g]) + 0.2 * x + rng.normal(size=N)
    return pd.DataFrame(dict(group=g, x=x)), y


def _additive_world(seed, n_noise=6):
    beta = np.array([0.9, 0.6, 0.4, 0.3, 0.25, 0.2])
    rng = np.random.default_rng(300 + seed)
    Xm = rng.normal(size=(N, 6 + n_noise)); X = pd.DataFrame(Xm, columns=[f'x{j+1}' for j in range(6 + n_noise)])
    y = (Xm[:, :6] >= 0).astype(float) @ beta + rng.normal(size=N)
    return X, y


def _names(cd, ids):
    return [cd.atom_names_[j] for j in ids]


def test_twins_one_dropped():
    X, y = _twins_world()
    cd = CertifiedDiscovery(seed=0, verbose=False, xgb_params=FAST, le_cuts=True, forced_cuts={'x1': [4.0]}, **PLAIN)
    fired = _run(cd, X, y)
    assert not fired
    assert {'x1>=4', 'x1<=4'} <= set(_names(cd, cd.core_full_))
    assert len({'x1>=4', 'x1<=4'} & set(_names(cd, cd.core_))) == 1
    assert len(cd.cert_dropped_) == 1
    d = cd.cert_dropped_[0]
    assert d['why'] == 'in span on S2' and d['r'] == pytest.approx(-1.0)
    assert {cd.atom_names_[d['atom']], cd.atom_names_[d['partner']]} == {'x1>=4', 'x1<=4'}
    gb = cd.report_['certification']['gate_basis']
    assert gb['core_in'] == len(cd.core_full_) and gb['tested'] == len(cd.core_) == cd.report_['certification']['core']
    assert len(np.asarray(cd.core_res_.pvalues)[1:]) == len(cd.core_) == len(cd.cert_sel_) == len(cd.core_t_)


def test_partition_one_dropped():
    X, y = _levels_world()
    cd = CertifiedDiscovery(seed=0, verbose=False, xgb_params=FAST, **PLAIN)
    fired = _run(cd, X, y)
    assert not fired
    admitted = [a for a in _names(cd, cd.core_full_) if a.startswith('group==')]
    tested = [a for a in _names(cd, cd.core_) if a.startswith('group==')]
    assert len(admitted) == 3 and len(tested) == 2
    assert [d['why'] for d in cd.cert_dropped_] == ['in span on S2']
    assert cd.atom_names_[cd.cert_dropped_[0]['atom']].startswith('group==')


def test_level_absent_from_s2_dropped():
    X, y = _levels_world(only_s1='E', n_only=24)
    cd = CertifiedDiscovery(seed=0, verbose=False, xgb_params=FAST, **PLAIN)
    fired = _run(cd, X, y)
    assert not fired
    assert 'group==E' in _names(cd, cd.core_full_) and 'group==E' not in _names(cd, cd.core_)
    why = {cd.atom_names_[d['atom']]: d['why'] for d in cd.cert_dropped_}
    assert why['group==E'] == 'constant on S2'
    assert [d['r'] for d in cd.cert_dropped_ if d['why'] == 'constant on S2'] == [None]
    assert 'group==E' not in _names(cd, cd.certified_)


def _rank_gap_on_s2(cd):
    Z2 = cd.Z[np.ix_(cd.idx_s2, np.array(cd.core_, int))].astype(np.float64)
    return Z2.shape[1] + 1 - int(np.linalg.matrix_rank(np.column_stack([np.ones(len(Z2)), Z2])))


def test_reduction_can_be_switched_off():
    X, y = _twins_world()
    cd = CertifiedDiscovery(seed=0, verbose=False, xgb_params=FAST, le_cuts=True, forced_cuts={'x1': [4.0]}, cert_basis_tol=None, **PLAIN)
    _run(cd, X, y)
    assert {'x1>=4', 'x1<=4'} <= set(_names(cd, cd.core_))
    assert cd.cert_dropped_ == [] and cd.core_ == cd.core_full_
    assert 'gate_basis' not in cd.report_['certification']
    assert _rank_gap_on_s2(cd) == 1


def test_unreduced_pass_gave_an_absent_level_the_intercepts_t():
    X, y = _levels_world(only_s1='E', n_only=24)
    kw = dict(seed=0, verbose=False, xgb_params=FAST, **PLAIN)
    off = CertifiedDiscovery(cert_basis_tol=None, **kw); _run(off, X, y)
    pos = _names(off, off.core_).index('group==E')
    t = np.asarray(off.core_res_.tvalues)
    assert float(np.ptp(off.Z[off.idx_s2, off.core_[pos]])) == 0.0
    assert abs(t[1 + pos]) == pytest.approx(abs(t[0]), rel=1e-6) and abs(t[0]) > 4
    assert 'group==E' in _names(off, off.certified_)
    on = CertifiedDiscovery(**kw); _run(on, X, y)
    assert 'group==E' not in _names(on, on.core_) and _rank_gap_on_s2(on) == 0


def test_new_parameter_is_keyword_only():
    import inspect
    p = inspect.signature(CertifiedDiscovery.__init__).parameters['cert_basis_tol']
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default == 1e-6


@pytest.mark.parametrize('seed', range(4))
def test_full_rank_core_untouched(seed):
    X, y = _additive_world(seed)
    kw = dict(seed=seed, verbose=False, xgb_params=FAST, forced_cuts={f'x{j+1}': [0.0] for j in range(6)}, n_cuts=4, **PLAIN)
    on = CertifiedDiscovery(**kw); off = CertifiedDiscovery(cert_basis_tol=None, **kw)
    assert not _run(on, X, y) and not _run(off, X, y)
    assert on.cert_dropped_ == [] and on.core_ == on.core_full_ == off.core_
    assert 'gate_basis' not in on.report_['certification']
    assert on.certified_ == off.certified_
    assert np.array_equal(np.asarray(on.core_res_.pvalues), np.asarray(off.core_res_.pvalues))
    pd.testing.assert_frame_equal(on.claims_, off.claims_)


def test_front_door_counts_the_tested_family(tmp_path):
    X, y = _levels_world()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rep = pv.certify(X, y, q=Q, seed=0, ledger=str(tmp_path / 'book'), xgb_params=FAST, stability_reps=0, **PLAIN)
    cd = rep.cd
    assert len(cd.cert_dropped_) == 1 and cd.khat_ >= 1
    m = len(cd.core_)
    assert rep.state['core'] == m == len(cd.core_full_) - 1
    assert rep.alpha_sel == pytest.approx(cd.khat_ * Q / m)
    fam = rep.certificate.guarantee.family
    assert fam['m'] == m == len(fam['p']) == len(fam['core_atoms'])
    assert cd.atom_names_[cd.cert_dropped_[0]['atom']] not in fam['core_atoms']
    assert rep.engine['certification'] == cd.report_['certification']
