"""evidence.py -- verdicts, limits, canonical evidence bundles, text certificate.

Verdict grammar (system-wide): PASS = certified absence within stated tolerance;
FINDINGS = violations exist, quantified; DECLINED = the instrument refuses, with a coded
reason. The limits panel is furniture: every bundle carries it, the renderer cannot drop
it, and the verifier checks it is present. The certificate id IS the truncated sha256 of
the canonical bundle -- rendering and hashing share one source of truth.

Marks: values tagged '#exact' are arithmetic facts of the artifact (the report's solid
dot); '#est' values are statistical estimates and carry intervals (the open dot).
"""
import json
import hashlib
import datetime

LIMITS = [
    'Scope is the model artifact, not the world: certified absence of named rule families in the file at the stated hash.',
    'No proxy claim: other features may jointly reconstruct a protected attribute; this instrument does not test that here.',
    'No causal claim: prices are model-behavior quantities, not effects of interventions on the world.',
    'No off-distribution claim: behavior is characterized at the artifact\'s own thresholds and probed contexts.',
    'Missing-value routing of ORIGINAL trees is not modeled by the transported spectrum; appended trees route NaN to the fail leaf. Audit assumes NaN-free inputs (R-07 declared).',
]

REFUSAL_CODES = {'R-01': 'nothing certifiable at the agreed FDR budget (k = 0)', 'R-02': 'reserved certification split too small', 'R-03': 'expansion above exact-enumeration cap', 'R-04': 'pattern off the representable lattice', 'R-05': 'feature not present in the model', 'R-06': 'probe budget exceeded', 'R-07': 'missing values present; transported audit not defined', 'R-08': 'unsupported clause kind'}


def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=_num)


def _num(x):
    try: return float(x)
    except Exception: return str(x)


def bundle_hash(bundle): return hashlib.sha256(canonical(bundle).encode()).hexdigest()


def make_bundle(verb, model, policy, findings, action, verdict, refusals, pvo, env):
    b = {'schema': 'provable/evidence/v0.1', 'verb': verb, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
         'model': model, 'policy': policy, 'findings': findings, 'action': action,
         'verdict': verdict, 'refusals': refusals, 'limits': LIMITS, 'predicted_vs_observed': pvo, 'environment': env}
    b['certificate_id'] = 'PV-' + bundle_hash({k: v for k, v in b.items() if k != 'certificate_id'})[:16].upper()
    return b


def verify_id(bundle):
    body = {k: v for k, v in bundle.items() if k != 'certificate_id'}
    return bundle.get('certificate_id') == 'PV-' + bundle_hash(body)[:16].upper()


def render_text(bundle):
    """Page-one content as filing-grade text (HTML/PDF renderer binds to the same bundle)."""
    v = bundle['verdict']; L = [f"PROVABLE - {bundle['verb'].upper()}  |  CERTIFICATE {bundle['certificate_id']}", '=' * 78,
         f"VERDICT: {v['state']}", v['statement'], '',
         'ESTABLISHES: ' + v.get('establishes', 'see findings'), 'DOES NOT ESTABLISH: proxies; fairness of outcomes; causal effects; off-distribution behavior.', '']
    if bundle['findings']:
        L.append('FINDINGS')
        for f in bundle['findings']:
            L.append(f"  clause {f['clause']}: atoms={f['n_atoms']}  data_mass#exact={f['data_mass']:.6g}  share_of_var={f['share_of_var']:.4f}")
            if f.get('profile_top'): L.append('    where: ' + ', '.join(f"{t}" for t in f['profile_top']))
    a = bundle.get('action') or {}
    if a:
        L += ['', 'ACTION', f"  mode={a['mode']}  atoms_removed={a['atoms_removed']}  trees_appended={a['trees_appended']}",
              f"  structural_residual#exact={a['structural_residual']:.3e}  grid_probe_max#exact={a['grid_probe_max']:.3e}",
              f"  stock_reload_diff: on_support#exact={a['stock_diff_on']:.3e}  off_support#exact={a['stock_diff_off']:.3e}",
              f"  holdout mse before/after#est: {a['holdout_mse_before']:.6g} -> {a['holdout_mse_after']:.6g}  (delta {a['holdout_delta']:+.3g} +/- {a['holdout_delta_ci']:.3g})"]
    if bundle['predicted_vs_observed']:
        L += ['', 'THE INSTRUMENT GRADES ITSELF (predicted vs observed)']
        for r in bundle['predicted_vs_observed']: L.append(f"  {r['quantity']}: predicted {r['predicted']:.6g}  observed {r['observed']:.6g}  [{r['status']}]")
    L += ['', 'WHERE THE INSTRUMENT DECLINED: ' + ('none in this run' if not bundle['refusals'] else '; '.join(f"{r['code']} {r['reason']}" for r in bundle['refusals']))]
    L += ['', 'WHAT THIS DOES NOT ESTABLISH (standing limits)'] + ['  - ' + s for s in bundle['limits']]
    L += ['', f"VERIFY WITHOUT TRUSTING THIS DOCUMENT:", f"  $ python -m provable_verify evidence.json {a.get('model_out','<model>')}",
         '', f"id = sha256(canonical bundle)[:16]  |  #exact = arithmetic fact of the artifact  |  #est = estimate"]
    return '\n'.join(L)


def _esc(x):
    t = str(x)
    for a, b in (('\\', '\\\\'), ('|', '\\|'), ('`', '\\`'), ('*', '\\*'), ('_', '\\_')): t = t.replace(a, b)
    return t


def render_markdown(bundle):
    """Human rendering of the canonical bundle as GitHub-flavored markdown. The bundle
    is the artifact of record; this file is derived from it and carries its id."""
    v, a = bundle['verdict'], bundle.get('action') or {}
    ca = v.get('certified_absent') or []
    L = [f"# provable · {bundle['verb']}  —  `{bundle['certificate_id']}`", '',
         f"## Verdict: **{v['state']}**", ''] + ([f"**Certified absent:** {' ; '.join(_esc(x) for x in ca)}", ''] if ca else []) + [v['statement'], '',
         '| Establishes | Does not establish |', '|---|---|',
         f"| {_esc(v.get('establishes', 'see findings'))} | proxies; fairness of outcomes; causal effects; off-distribution behavior |", '']
    if bundle['findings']:
        L += ['## Findings', '', '| clause | atoms | data mass `#exact` | share of model var | where |', '|---|---|---|---|---|']
        for f in bundle['findings']:
            L.append(f"| {_esc(f['clause'])} | {f['n_atoms']} | {f['data_mass']:.6g} | {f['share_of_var']:.4f} | {_esc(', '.join(f.get('profile_top', [])) or '—')} |")
        L.append('')
    if a:
        L += ['## Action', '', '| quantity | value |', '|---|---|',
              f"| mode | {a['mode']} |", f"| atoms removed / trees appended | {a['atoms_removed']} / {a['trees_appended']} |",
              f"| structural residual `#exact` | {a['structural_residual']:.3e} |", f"| behavioral grid probe `#exact` | {a['grid_probe_max']:.3e} |",
              f"| stock reload diff on/off rule `#exact` | {a['stock_diff_on']:.3e} / {a['stock_diff_off']:.3e} |",
              f"| holdout MSE before → after `#est` | {a['holdout_mse_before']:.6g} → {a['holdout_mse_after']:.6g} (Δ {a['holdout_delta']:+.4g} ± {a['holdout_delta_ci']:.3g}) |",
              f"| edited artifact | `{_esc(a.get('model_out', ''))}` sha256 `{a.get('model_out_sha256', '')[:16]}…` |", '']
    if bundle['predicted_vs_observed']:
        L += ['## The instrument grades itself', '', '| quantity | predicted | observed | status |', '|---|---|---|---|']
        for r in bundle['predicted_vs_observed']: L.append(f"| {_esc(r['quantity'])} | {r['predicted']:.6g} | {r['observed']:.6g} | {r['status']} |")
        L.append('')
    L += ['## Where the instrument declined', '', ('none in this run' if not bundle['refusals'] else '\n'.join(f"- `{r['code']}` {_esc(r['reason'])}" for r in bundle['refusals'])), '',
          '## What this does not establish (standing limits)', ''] + [f"- {_esc(x)}" for x in bundle['limits']]
    L += ['', '## Verify without trusting this document', '', '```', f"python -m provable_verify evidence.json {a.get('model_out', '<model>')}", '```', '',
          f"`{bundle['certificate_id']}` = `PV-` + sha256(canonical bundle)[:16] · `#exact` = arithmetic fact of the artifact · `#est` = estimate with interval · created {bundle['created_utc']}"]
    return '\n'.join(L)
