"""selftest.py -- the provable v0.2 acceptance ledger (T1-T7), house psql format.

Synthetic lending-shaped world with a PLANTED forbidden interaction 1[age>=46]*group_A.
External-artifact discipline throughout: the model is trained by stock xgboost, saved,
and every provable operation goes through the saved file, never the in-memory object.
"""
import sys, os, json, time, subprocess
import numpy as np
import pandas as pd
import xgboost as xgb
from tabulate import tabulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'verify', 'src'))
os.chdir(os.path.join(ROOT, 'tests'))
VENV = dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'verify', 'src'))
from provable import load, audit, enforce, Policy, between, XGBAdapter, CertifiedDiscovery

T0 = time.time(); ROWS = []
def add(name, metric, val, tol, cmp='le'): ROWS.append([len(ROWS) + 1, name, metric, f'{val:.3e}' if isinstance(val, float) else str(val), f'{tol:.0e}' if isinstance(tol, float) else str(tol), 'PASS' if ((val <= tol) if cmp == 'le' else (val == tol)) else 'FAIL'])

# ---------------------------------------------------------------- world + external model
rng = np.random.default_rng(7)
N, NF = 6000, 8
NAMES = ['age', 'income', 'hours', 'tenure', 'group_A', 'group_B', 'x6', 'x7']
X = np.zeros((N, NF))
X[:, 0] = rng.uniform(18, 80, N); X[:, 1] = rng.lognormal(10.5, 0.6, N) / 1e3; X[:, 2] = rng.uniform(10, 70, N)
X[:, 3] = rng.exponential(6, N); X[:, 4] = (rng.random(N) < 0.35).astype(float); X[:, 5] = (rng.random(N) < 0.5).astype(float)
X[:, 6] = rng.normal(0, 1, N); X[:, 7] = rng.normal(0, 1, N)
y = 0.02 * (X[:, 0] - 40) + 0.008 * X[:, 1] + 0.01 * X[:, 2] + 0.15 * X[:, 5] + 0.8 * (X[:, 0] >= 46) * X[:, 4] + 0.25 * rng.normal(0, 1, N)
tr, ho = np.arange(0, 4500), np.arange(4500, N)
D = xgb.DMatrix(X[tr].astype(np.float32), label=y[tr], feature_names=NAMES)
bst = xgb.train({'max_depth': 4, 'eta': 0.1, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'seed': 1}, D, num_boost_round=200)
bst.save_model('model.json')
t_setup = time.time() - T0

# ---------------------------------------------------------------- T1 parse exactness
t1 = time.time()
ad, spec = load('model.json')
stock = ad.predict_margin(X[:300])
recon = np.full(300, ad.base_score)
Xf32 = X[:300].astype(np.float32)
for box, w in ad.leaf_boxes():
    m = np.ones(300, dtype=bool)
    for f, (lo, hi) in box.items(): m &= (Xf32[:, f] >= np.float32(lo)) & (Xf32[:, f] < np.float32(hi))
    recon += w * m
add('T1 parse: leaf boxes reconstruct stock margin', 'max abs diff (300 rows)', float(np.max(np.abs(recon - stock))), 1e-3)

# ---------------------------------------------------------------- T2 transport theorem
ev = spec.evaluate(X[:300])
add('T2 transport: atom spectrum == stock margin', f'max abs diff ({len(spec.atoms)} atoms)', float(np.max(np.abs(ev - stock))), 1e-3)
t_parse = time.time() - t1

# ---------------------------------------------------------------- T3 reference cross-check
t3 = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_mobius_calculus import MobiusCalculus
Xb = (np.random.default_rng(3).random((3000, 8)) < 0.5).astype(np.float64)
yb = 1.0 * Xb[:, 0] - 0.7 * Xb[:, 1] + 1.3 * Xb[:, 2] * Xb[:, 3] + 0.2 * np.random.default_rng(4).normal(size=3000)
mc = MobiusCalculus(seed=0, depth=4, trees=120).fit(pd.DataFrame(Xb, columns=[f'b{i}' for i in range(8)]), yb)
ref = mc.full_spectrum()
_, spec_b = load(booster=mc.bst_)
mine = {}
for A, c in spec_b.atoms.items():
    assert all(0.0 < t <= 1.0 for _, t in A), 'non-binary threshold on binary model'
    key = frozenset(f for f, _ in A); mine[key] = mine.get(key, 0.0) + c
keys = set(ref) | set(mine)
add('T3 transport == reference Mobius (binary model)', f'max |I diff| over {len(keys)} supports', float(max(abs(ref.get(k, 0.0) - mine.get(k, 0.0)) for k in keys)), 1e-6)
t_ref = time.time() - t3

# ---------------------------------------------------------------- T4 audit
t4 = time.time()
pol = Policy([between('age', 'group_A'), between('hours', 'group_B')], name='fair_lending_demo')
b_aud = audit('model.json', pol, X[tr], out_dir='dossier_audit')
f_planted = next(f for f in b_aud['findings'] if 'group_A' in f['clause'])
f_null = next(f for f in b_aud['findings'] if 'group_B' in f['clause'])
add('T4a audit verdict on unedited model', 'state==FINDINGS', 0.0 if b_aud['verdict']['state'] == 'FINDINGS' else 1.0, 0.0)
add('T4b planted family found', 'atoms > 0', 0.0 if f_planted['n_atoms'] > 0 else 1.0, 0.0)
add('T4c planted mass dominates null pair', 'null/planted mass ratio', f_null['data_mass'] / f_planted['data_mass'], 0.2)
t_aud = time.time() - t4

# ---------------------------------------------------------------- T5 enforce write-back
t5 = time.time()
pol1 = Policy([between('age', 'group_A')], name='fair_lending_demo')
b_enf = enforce('model.json', pol1, X[tr], X[ho], 'model_v2.json', y_holdout=y[ho], out_dir='dossier_enforce')
a = b_enf['action']
add('T5a off-rule rows bit-identical after reload', 'max abs diff', a['stock_diff_off'], 0.0, 'le')
add('T5b stock reload == functional edit on-rule', 'max abs diff', a['stock_diff_on'], 5e-3)
add('T5c structural residual (re-transported)', 'max |forbidden coef|', a['structural_residual'], 1e-4)
add('T5d behavioral grid probe (full pair grid)', 'max |second diff|', a['grid_probe_max'], 1e-3)
add('T5e enforce verdict', 'state==PASS', 0.0 if b_enf['verdict']['state'] == 'PASS' else 1.0, 0.0)
pvo = b_enf['predicted_vs_observed'][0]
add('T5f self-grading: price predicted vs realized', '|obs-pred|/pred', abs(pvo['observed'] - pvo['predicted']) / pvo['predicted'], 0.25)
t_enf = time.time() - t5

# ---------------------------------------------------------------- T6 independent verify + mutation
t6 = time.time()
r = subprocess.run([sys.executable, '-m', 'provable_verify', 'dossier_enforce/evidence.json', 'model_v2.json'], capture_output=True, text=True, env=VENV)
add('T6a independent verifier passes', 'exit code', float(r.returncode), 0.0)
VERIFY_OUT = r.stdout.strip()
raw2 = json.load(open('model_v2.json'))
trees = raw2['learner']['gradient_booster']['model']['trees']
leaf = trees[-1]; kids = leaf['left_children']
li = max(i for i in range(len(kids)) if kids[i] == -1)
leaf['split_conditions'][li] = float(leaf['split_conditions'][li]) + 1e-3
json.dump(raw2, open('model_v2_tampered.json', 'w'))
bun = json.load(open('dossier_enforce/evidence.json'))
bun['action']['model_out_sha256'] = XGBAdapter.sha256('model_v2_tampered.json')
body = {k: v for k, v in bun.items() if k != 'certificate_id'}
import hashlib as _h; from provable.evidence import canonical as _canon
bun['certificate_id'] = 'PV-' + _h.sha256(_canon(body).encode()).hexdigest()[:16].upper()
json.dump(bun, open('evidence_tampered.json', 'w'))
r2 = subprocess.run([sys.executable, '-m', 'provable_verify', 'evidence_tampered.json', 'model_v2_tampered.json'], capture_output=True, text=True, env=VENV)
add('T6b mutation caught (attacker re-hashed bundle)', 'exit code != 0', 1.0 if r2.returncode != 0 else 0.0, 1.0, 'eq')
t_ver = time.time() - t6

# ---------------------------------------------------------------- T7 discover + refusal
t7 = time.time()
Xdf = pd.DataFrame(X, columns=NAMES)
cd = CertifiedDiscovery(seed=0, verbose=False, stability_reps=0, xgb_params={'n_estimators': 300})
rep = cd.run(Xdf, y)
add('T7a discover certifies planted structure', 'k certified >= 3', float(rep['certification']['k']), 3.0, cmp='ge') if False else add('T7a discover certifies planted structure', 'k certified (>=3)', 0.0 if rep['certification']['k'] >= 3 else 1.0, 0.0)
names_cert = ' | '.join(c['atom'] for c in rep['claims'][:6])
cd0 = CertifiedDiscovery(seed=0, verbose=False, stability_reps=0, xgb_params={'n_estimators': 200})
rep0 = cd0.run(Xdf, np.random.default_rng(11).permutation(y))
add('T7b refusal on shuffled target', 'k == 0', float(rep0['certification']['k']), 0.0, 'le')
add('T7c refusal phase named', "phase startswith 'I '", 0.0 if rep0['diagnostics']['phase'].startswith('I ') else 1.0, 0.0)
t_dis = time.time() - t7

# ---------------------------------------------------------------- report
print(f"provable v0.2 selftest | xgb {xgb.__version__} np {np.__version__} | N={N} model: 200 trees depth 4 | atoms={len(spec.atoms)}")
print(tabulate(ROWS, headers=['#', 'test', 'metric', 'value', 'tol', 'status'], tablefmt='psql'))
print(f"planted clause: atoms={f_planted['n_atoms']} mass={f_planted['data_mass']:.4f} share={f_planted['share_of_var']:.3f} where: {', '.join(f_planted['profile_top'])}")
print(f"null clause:    atoms={f_null['n_atoms']} mass={f_null['data_mass']:.6f} share={f_null['share_of_var']:.5f}")
print(f"enforce: removed={a['atoms_removed']} atoms; price predicted={pvo['predicted']:.4f} realized={pvo['observed']:.4f}; holdout MSE {a['holdout_mse_before']:.4f} -> {a['holdout_mse_after']:.4f} (delta {a['holdout_delta']:+.4f} +/- {a['holdout_delta_ci']:.4f})")
print(f"discover: k={rep['certification']['k']} of core {rep['certification']['core']} | top claims: {names_cert}")
print(f"refusal:  k={rep0['certification']['k']} | {rep0['diagnostics']['phase']}")
print(f"timing: setup {t_setup:.1f}s parse+transport {t_parse:.1f}s ref-check {t_ref:.1f}s audit {t_aud:.1f}s enforce {t_enf:.1f}s verify {t_ver:.1f}s discover {t_dis:.1f}s | total {time.time()-T0:.1f}s")
print('\n--- independent verifier attestation ---'); print(VERIFY_OUT)
npass = sum(1 for r_ in ROWS if r_[-1] == 'PASS'); print(f"\n{npass}/{len(ROWS)} PASS")
