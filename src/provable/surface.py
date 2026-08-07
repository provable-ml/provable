"""surface.py -- the verbs. load -> Spectrum; audit(model, policy, X) -> bundle;
enforce(model, policy, X_train, X_holdout, out_path) -> edited native file + bundle.
Every bundle carries verdict, limits, refusals, and the self-grading panel.
"""
import os
import json
import platform
import numpy as np
import xgboost

from .adapters_xgb import XGBAdapter
from .calculus import Spectrum
from .evidence import make_bundle, render_text, render_markdown, REFUSAL_CODES
from .report_html import render_html


def load(path=None, booster=None, max_atoms=2_000_000):
    ad = XGBAdapter(path=path, booster=booster)
    return ad, Spectrum(ad, max_atoms=max_atoms)


def _env():
    from . import __version__
    return {'xgboost': xgboost.__version__, 'numpy': np.__version__, 'python': platform.python_version(), 'provable': __version__}


def _families(spec, policy):
    resolved, refusals = policy.resolve(spec.ad.feature_names)
    fams = []
    for cl in resolved:
        if cl['kind'] == 'between': fam = spec.family(feats=[cl['i'], cl['j']]); label = f"between({cl['features'][0]}, {cl['features'][1]})"
        else: fam = spec.family(involving=cl['i'], min_order=cl['min_order']); label = f"involving({cl['feature']}, order>={cl['min_order']})"
        fams.append((cl, label, fam))
    return fams, refusals


def _findings(spec, fams, X):
    rows, var = [], float(np.var(spec.ad.predict_margin(X)))
    for cl, label, fam in fams:
        mass, _ = spec.data_mass(fam, X)
        prof = []
        if cl['kind'] == 'between':
            for fname, fidx in ((cl['features'][0], cl['i']), (cl['features'][1], cl['j'])):
                p = spec.profile(fam, fidx)
                if p: t = max(p, key=p.get); prof.append(f"{fname}>={t:g} (|c| {p[t]:.3g})")
        rows.append({'clause': label, 'n_atoms': int(sum(1 for A in fam if A)), 'data_mass': mass, 'share_of_var': mass / var if var > 0 else 0.0, 'profile_top': prof})
    return rows


def audit(model_path, policy, X, out_dir=None):
    ad, spec = load(model_path)
    fams, refusals = _families(spec, policy)
    X = np.asarray(X, dtype=np.float64)
    if np.isnan(X).any():
        refusals = refusals + [{'code': 'R-07', 'reason': 'missing values present; transported audit not defined'}]
        bundle = make_bundle('audit', {'path': str(model_path), 'sha256': XGBAdapter.sha256(model_path), 'n_trees': len(ad.raw['learner']['gradient_booster']['model']['trees']), 'n_atoms': len(spec.atoms)},
                             policy.to_dict() | {'sha256': policy.sha256()}, [], None,
                             {'state': 'DECLINED', 'statement': 'Input contains missing values; the transported audit is not defined for NaN routing (R-07).', 'establishes': 'nothing'},
                             refusals, [], _env())
        return _emit(bundle, out_dir)
    findings = _findings(spec, fams, X)
    n_viol = sum(1 for f in findings if f['n_atoms'] > 0 and f['data_mass'] > policy.tolerance)
    if n_viol == 0 and not refusals: state, stmt = 'PASS', 'No rules matching any forbidden clause are present in the artifact (transported spectrum, exact).'
    elif n_viol > 0: state, stmt = 'FINDINGS', f'{n_viol} clause(s) matched by rules present in the artifact; masses and locations quantified below. Removal is priced and available via enforce().'
    else: state, stmt = 'DECLINED', 'Audit incomplete: see refusals.'
    bundle = make_bundle('audit', {'path': str(model_path), 'sha256': XGBAdapter.sha256(model_path), 'n_trees': len(ad.raw['learner']['gradient_booster']['model']['trees']), 'n_atoms': len(spec.atoms)},
                         policy.to_dict() | {'sha256': policy.sha256()}, findings, None,
                         {'state': state, 'statement': stmt, 'establishes': 'presence/absence and exact mass of the named rule families in this artifact on this sample'},
                         refusals, [], _env())
    return _emit(bundle, out_dir)


def enforce(model_path, policy, X_train, X_holdout, out_path, y_holdout=None, out_dir=None, probe_contexts=200, seed=0):
    ad, spec = load(model_path)
    fams, refusals = _families(spec, policy)
    X_train = np.asarray(X_train, dtype=np.float64); X_holdout = np.asarray(X_holdout, dtype=np.float64)
    if np.isnan(X_train).any() or np.isnan(X_holdout).any(): raise ValueError('R-07: missing values present; transported enforce is not defined for NaN routing')
    findings = _findings(spec, fams, X_train)
    removal = {}
    for _, _, fam in fams:
        for A, c in fam.items():
            if A and A not in removal: removal[A] = c
    predicted_price, _ = spec.data_mass(removal, X_train)
    raw2 = ad.append_rules(removal)
    XGBAdapter.save_raw_json(raw2, out_path)
    ad2 = XGBAdapter.reload(out_path)
    F_tr, G_tr = ad.predict_margin(X_train), ad2.predict_margin(X_train)
    F_ho, G_ho = ad.predict_margin(X_holdout), ad2.predict_margin(X_holdout)
    func_tr = F_tr - spec.evaluate(X_train, atoms=removal)
    Xf = np.asarray(X_train, dtype=np.float32)
    on = np.zeros(len(Xf), dtype=bool)
    for A in removal:
        m = np.ones(len(Xf), dtype=bool)
        for f, t in A: m &= Xf[:, f] >= np.float32(t)
        on |= m
    stock_on = float(np.max(np.abs(G_tr[on] - func_tr[on]))) if on.any() else 0.0
    stock_off = float(np.max(np.abs(G_tr[~on] - F_tr[~on]))) if (~on).any() else 0.0
    spec2 = Spectrum(ad2)
    struct = 0.0
    for cl, _, _ in fams:
        fam2 = spec2.family(feats=[cl['i'], cl['j']]) if cl['kind'] == 'between' else spec2.family(involving=cl['i'], min_order=cl['min_order'])
        struct = max(struct, max((abs(c) for A, c in fam2.items() if A), default=0.0))
    th2 = ad2.thresholds()
    ctx = Spectrum.grid_contexts(th2, ad2.num_feature, probe_contexts, seed)
    probe = 0.0
    for cl, _, _ in fams:
        if cl['kind'] == 'between': probe = max(probe, Spectrum.grid_probe(ad2.predict_margin, th2.get(cl['i'], []), th2.get(cl['j'], []), cl['i'], cl['j'], ctx))
    realized = float(np.mean((F_ho - G_ho) ** 2))
    dsq = (F_ho - G_ho) ** 2
    if y_holdout is not None:
        y_ho = np.asarray(y_holdout, dtype=np.float64)
        mse_b, mse_a = float(np.mean((y_ho - F_ho) ** 2)), float(np.mean((y_ho - G_ho) ** 2))
        dd = (y_ho - G_ho) ** 2 - (y_ho - F_ho) ** 2
        mse_ci = float(1.96 * np.std(dd) / np.sqrt(len(dd)))
    else: mse_b, mse_a, mse_ci = float('nan'), float('nan'), float('nan')
    GATES = {'structural_residual': 1e-4, 'grid_probe_max': 1e-3, 'stock_diff_off': 0.0}
    ok = struct <= GATES['structural_residual'] and probe <= GATES['grid_probe_max'] and stock_off == GATES['stock_diff_off']
    state = 'PASS' if ok else 'FINDINGS'
    labels = [(f"{cl['features'][0]} × {cl['features'][1]}" if cl['kind'] == 'between' else f"any rule of order ≥ {cl['min_order']} involving {cl['feature']}") for cl, _, _ in fams]
    claim = 'This artifact contains no rules combining ' + '; and none for '.join(labels) + ', at any threshold the artifact can express.'
    stmt = (claim + ' Verified structurally and behaviorally; independently re-checkable.' if ok
            else f'Edit applied but certification gates not met (structural {struct:.2e}, probe {probe:.2e}); see evidence panel and refusals.')
    action = {'mode': 'excise', 'atoms_removed': int(sum(1 for A in removal if A)), 'trees_appended': int(sum(1 for A in removal if A)),
              'structural_residual': struct, 'grid_probe_max': probe, 'stock_diff_on': stock_on, 'stock_diff_off': stock_off,
              'holdout_mse_before': mse_b, 'holdout_mse_after': mse_a, 'holdout_delta': mse_a - mse_b,
              'holdout_delta_ci': mse_ci, 'holdout_pred_change_mass': realized, 'holdout_pred_change_ci': float(1.96 * np.std(dsq) / np.sqrt(len(dsq))), 'model_out': os.path.basename(out_path), 'model_out_sha256': XGBAdapter.sha256(out_path), 'gates': GATES}
    pvo = [{'quantity': 'excision price: train-sample mass vs realized holdout mean squared change', 'predicted': predicted_price, 'observed': realized,
            'status': 'within 25%' if predicted_price > 0 and abs(realized - predicted_price) <= 0.25 * predicted_price else 'outside 25%'}]
    bundle = make_bundle('enforce', {'path': str(model_path), 'sha256': XGBAdapter.sha256(model_path), 'n_trees': len(ad.raw['learner']['gradient_booster']['model']['trees']), 'n_atoms': len(spec.atoms)},
                         policy.to_dict() | {'sha256': policy.sha256()}, findings, action,
                         {'state': state, 'statement': stmt, 'certified_absent': labels if ok else [], 'establishes': 'exact removal of the named rule families from the artifact, priced, with a still-native output file'},
                         refusals, pvo, _env())
    return _emit(bundle, out_dir)


def _emit(bundle, out_dir):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'evidence.json'), 'w') as fh: json.dump(bundle, fh, indent=1, sort_keys=True)
        with open(os.path.join(out_dir, 'certificate.md'), 'w') as fh: fh.write(render_markdown(bundle))
        with open(os.path.join(out_dir, 'certificate.html'), 'w') as fh: fh.write(render_html(bundle))
    return bundle
