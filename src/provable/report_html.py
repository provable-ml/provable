"""report_html.py -- the certificate as a product surface. Renders the canonical
evidence bundle into ONE self-contained HTML file: no external fonts, scripts, or
images; prints on A4; survives grayscale (verdict is ring-SHAPE redundant: PASS solid,
FINDINGS double, DECLINED dashed). Theme: FT paper (#FFF1E5), claret, teal, Georgia.

The seal's ring text carries the SCOPE ("MODEL ARTIFACT - NAMED FAMILIES - date"), so a
cropped screenshot of the seal still carries its own qualification. The QR encodes the
certificate id + full sha256 of the canonical bundle (today: offline payload; when the
anchor registry exists, the same payload is its lookup key). QR needs the pure-python
`qrcode` package; without it the panel degrades to the printed hash.
"""
import html as _html
import hashlib
from .evidence import canonical

INK, PAPER, CLARET, TEAL, SLATE, LINE, CARD = '#33302E', '#FFF1E5', '#990F3D', '#0D7680', '#66605C', '#CCC1B7', '#FFFFFF'
VCOL = {'PASS': '#00693F', 'FINDINGS': CLARET, 'DECLINED': SLATE}
HEADLINE = {'PASS': 'Certified absent', 'FINDINGS': 'Findings quantified', 'DECLINED': 'Declined to certify'}

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#FFF1E5;color:#33302E;font-family:Georgia,'Times New Roman',serif;line-height:1.45;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.page{max-width:780px;margin:0 auto;padding:34px 40px 26px}
.rule2{border-top:2px solid #33302E;border-bottom:1px solid #33302E;height:5px;margin-bottom:14px}
.rule1{border-top:1px solid #CCC1B7;margin:22px 0 16px}
.mast{display:flex;justify-content:space-between;align-items:baseline}
.word{font-size:26px;letter-spacing:.14em;font-weight:700}
.word .dot{color:#990F3D}
.kicker,.lab{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:#66605C}
.certid{font-family:Menlo,Consolas,monospace;font-size:12.5px;color:#33302E}
.meta{display:flex;gap:26px;flex-wrap:wrap;margin-top:8px}
.meta div{font-size:12.5px;color:#66605C}
.meta b{color:#33302E;font-weight:400;font-family:Menlo,Consolas,monospace;font-size:11.5px}
.hero{display:flex;gap:30px;margin:26px 0 6px;align-items:flex-start}
.head{font-size:34px;line-height:1.12;font-weight:700;margin:2px 0 10px}
.stmt{font-size:15.5px;max-width:56ch}
.claim{font-size:17.5px;max-width:58ch;margin-top:6px}
.claim .fam{color:#990F3D;font-weight:700;white-space:nowrap}
.notes{font-size:11px;color:#66605C;margin-top:10px;line-height:1.5}
.notes b{font-family:Menlo,Consolas,monospace}
sup{font-family:Menlo,Consolas,monospace;font-size:9px;color:#66605C}
.scope{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:22px 0}
.card{background:#FFFFFF;border:1px solid #E5D9CC;padding:14px 16px}
.card h4{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px}
.card.yes h4{color:#00693F}.card.no h4{color:#990F3D}
.card li{font-size:13.5px;margin:0 0 6px 16px}
h3.sec{font-size:19px;margin:26px 0 10px;border-bottom:1px solid #CCC1B7;padding-bottom:6px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:#66605C;text-align:left;padding:6px 10px;border-bottom:1px solid #33302E}
td{padding:7px 10px;border-bottom:1px solid #E5D9CC;font-variant-numeric:tabular-nums}
td.num,th.num{text-align:right;font-family:Menlo,Consolas,monospace;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:6px 0}
.stat{background:#FFFFFF;border:1px solid #E5D9CC;padding:12px 14px}
.stat .v{font-family:Menlo,Consolas,monospace;font-size:16.5px;margin-top:4px}
.stat .n{font-size:11px;color:#66605C;margin-top:3px}
.chip{display:inline-block;font-family:'Helvetica Neue',Arial,sans-serif;font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:2px 8px;border-radius:9px;color:#fff}
.limits{background:#FFFFFF;border-left:4px solid #990F3D;border-top:1px solid #E5D9CC;border-right:1px solid #E5D9CC;border-bottom:1px solid #E5D9CC;padding:14px 18px;margin-top:8px}
.limits li{font-size:13px;margin:0 0 7px 16px}
.verify{background:#262A33;color:#FFF1E5;display:flex;gap:22px;align-items:center;padding:18px 20px;margin-top:26px}
.verify code{font-family:Menlo,Consolas,monospace;font-size:12.5px;color:#9CE2D8;display:block;margin-top:8px}
.verify .qr{background:#fff;padding:6px;flex:0 0 auto}
.foot{margin-top:18px;font-size:11.5px;color:#66605C}
.foot code{font-family:Menlo,Consolas,monospace;font-size:10.5px}
@media print{.page{padding:18px 24px}body{background:#FFF1E5}}
"""


def _e(x): return _html.escape(str(x), quote=True)


def _seal(state, date, color):
    ring = {'PASS': f'<circle cx="90" cy="90" r="62" fill="none" stroke="{color}" stroke-width="5"/>',
            'FINDINGS': f'<circle cx="90" cy="90" r="65" fill="none" stroke="{color}" stroke-width="2.5"/><circle cx="90" cy="90" r="58" fill="none" stroke="{color}" stroke-width="2.5"/>',
            'DECLINED': f'<circle cx="90" cy="90" r="62" fill="none" stroke="{color}" stroke-width="4" stroke-dasharray="7 6"/>'}[state]
    ringtext = f'MODEL ARTIFACT · NAMED FAMILIES · {date} · '
    return f'''<svg width="180" height="180" viewBox="0 0 180 180" role="img" aria-label="verdict seal {state}">
<defs><path id="sealring" d="M 90,90 m -76,0 a 76,76 0 1,1 152,0 a 76,76 0 1,1 -152,0"/></defs>
{ring}
<text font-family="Helvetica Neue,Arial,sans-serif" font-size="9.2" letter-spacing="2.1" fill="{SLATE}"><textPath href="#sealring">{_e(ringtext * 2)}</textPath></text>
<text x="90" y="86" text-anchor="middle" font-family="Helvetica Neue,Arial,sans-serif" font-size="19" font-weight="700" letter-spacing="3" fill="{color}">{state}</text>
<text x="90" y="104" text-anchor="middle" font-family="Georgia,serif" font-size="10.5" fill="{SLATE}">provable</text>
</svg>'''


def _qr_svg(payload):
    try:
        import qrcode
        q = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=3)
        q.add_data(payload); q.make(fit=True)
        m = q.get_matrix(); n = len(m); s = 3
        rects = ''.join(f'<rect x="{x*s}" y="{y*s}" width="{s}" height="{s}"/>' for y in range(n) for x in range(n) if m[y][x])
        return f'<svg class="qr" width="{n*s}" height="{n*s}" viewBox="0 0 {n*s} {n*s}" fill="#111"><rect width="100%" height="100%" fill="#fff"/>{rects}</svg>'
    except Exception:
        return ''


def render_html(bundle):
    v = bundle['verdict']; a = bundle.get('action') or {}; state = v['state']; color = VCOL.get(state, SLATE)
    date = bundle['created_utc'][:10]
    full_hash = hashlib.sha256(canonical({k: x for k, x in bundle.items() if k != 'certificate_id'}).encode()).hexdigest()
    model8 = bundle['model']['sha256'][:8]; pol = bundle['policy']
    prov = f"model <b>{model8}…</b>" + (f" → edited <b>{a['model_out_sha256'][:8]}…</b>" if a else '')
    ca = v.get('certified_absent') or []
    if ca:
        fams_html = '</span>; and none for <span class="fam">'.join(_e(x) for x in ca)
        claim_html = f'<div class="claim">This artifact contains <b>no rules combining</b> <span class="fam">{fams_html}</span> — at any threshold the artifact can express.</div><div class="stmt" style="margin-top:8px;color:{SLATE}">Verified structurally and behaviorally (evidence panel below); independently re-checkable without trusting this document.</div>'
    else:
        claim_html = f'<div class="stmt">{_e(v["statement"])}</div>'
    findings = ''
    if bundle['findings']:
        rows = ''.join(f"<tr><td>{_e(f['clause'])}</td><td class='num'>{f['n_atoms']}</td><td class='num'>{f['data_mass']:.6g} ●</td><td class='num'>{f['share_of_var']:.2%}</td><td>{_e(', '.join(f.get('profile_top', [])) or '—')}</td></tr>" for f in bundle['findings'])
        findings = f"<h3 class='sec'>Findings</h3><table><tr><th>Clause</th><th class='num'>Rules</th><th class='num'>Data mass</th><th class='num'>Share of model</th><th>Where</th></tr>{rows}</table>"
    action = ''
    if a:
        g = a.get('gates', {})
        def flag(ok_, info=False): return '<span class="chip" style="background:#66605C">INFO</span>' if info else ('<span class="chip" style="background:#00693F">PASS</span>' if ok_ else f'<span class="chip" style="background:{CLARET}">FAIL</span>')
        offv = 'bit-identical (0)' if a['stock_diff_off'] == 0.0 else f"{a['stock_diff_off']:.2e}"
        rows = [
          f"<tr><td>Structural residual ●<sup>1</sup></td><td class='num'>{a['structural_residual']:.2e}</td><td class='num'>≤ {g.get('structural_residual', 1e-4):.0e}</td><td>{flag(a['structural_residual'] <= g.get('structural_residual', 1e-4))}</td><td>no surviving rule joins the named families</td></tr>",
          f"<tr><td>Behavioral probe ●<sup>2</sup></td><td class='num'>{a['grid_probe_max']:.2e}</td><td class='num'>≤ {g.get('grid_probe_max', 1e-3):.0e}</td><td>{flag(a['grid_probe_max'] <= g.get('grid_probe_max', 1e-3))}</td><td>output additively separable in each named pair</td></tr>",
          f"<tr><td>Off-rule rows ●<sup>3</sup></td><td class='num'>{offv}</td><td class='num'>= 0</td><td>{flag(a['stock_diff_off'] == 0.0)}</td><td>zero collateral outside the removed rules</td></tr>",
          f"<tr><td>On-rule agreement ●</td><td class='num'>{a['stock_diff_on']:.2e}</td><td class='num'>float32</td><td>{flag(True, info=True)}</td><td>edited file matches the exact functional edit</td></tr>",
          f"<tr><td>Holdout MSE ○</td><td class='num'>{a['holdout_mse_before']:.4g} → {a['holdout_mse_after']:.4g}</td><td class='num'>—</td><td>{flag(True, info=True)}</td><td>Δ {a['holdout_delta']:+.3g} ± {a['holdout_delta_ci']:.2g} (95% CI)</td></tr>"]
        action = f"""<h3 class='sec'>Evidence panel — {_e(a['mode'])} surgery <span class='kicker' style='margin-left:10px'>{a['atoms_removed']} rules removed · {a['trees_appended']} trees appended · output stays native</span></h3>
<table><tr><th>Check</th><th class='num'>Measured</th><th class='num'>Tolerance</th><th>Status</th><th>Reading</th></tr>{''.join(rows)}</table>
<div class='notes'><b>1</b> Re-parse the edited file and re-derive its full rule spectrum; value is the largest surviving coefficient on any rule joining a named family. Appended-tree leaves are float32 in the native format, so the certified zero is a ~1e-9 residual — disclosed, not hidden.
<b>2</b> Second differences over the full pair grid of the artifact's own thresholds at probe contexts; all-zero cells ⇔ no joint rule anywhere the artifact can express one.
<b>3</b> Appended trees emit exactly 0.0 outside their rule, so untouched rows are bit-identical on stock reload.
<b>4</b> Checks 1–2 are independently recomputed by <b>provable-verify</b> (own parser, own predictor, own transport; verifier checks 4–5).</div>"""
    pvo = ''
    if bundle['predicted_vs_observed']:
        rows = ''.join(f"<tr><td>{_e(r['quantity'])}</td><td class='num'>{r['predicted']:.6g}</td><td class='num'>{r['observed']:.6g}</td><td><span class='chip' style='background:{'#00693F' if 'within' in r['status'] else CLARET}'>{_e(r['status'])}</span></td></tr>" for r in bundle['predicted_vs_observed'])
        pvo = f"<h3 class='sec'>The instrument grades itself</h3><table><tr><th>Quantity</th><th class='num'>Predicted</th><th class='num'>Observed</th><th>Status</th></tr>{rows}</table>"
    refus = '; '.join(f"<code>{_e(r['code'])}</code> {_e(r['reason'])}" for r in bundle['refusals']) or 'None in this run.'
    limits = ''.join(f'<li>{_e(x)}</li>' for x in bundle['limits'])
    qr = _qr_svg(f"PROVABLE|{bundle['certificate_id']}|EVIDENCE-SHA256:{full_hash}")
    qr_block = qr or f"<div style='font-size:11px'>QR unavailable — evidence sha256:<br><code>{full_hash}</code></div>"
    estab = _e(v.get('establishes', 'see findings'))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{_e(bundle['certificate_id'])} — provable {_e(bundle['verb'])}</title><style>{CSS}</style></head><body><div class="page">
<div class="rule2"></div>
<div class="mast"><div class="word">PROVABLE<span class="dot">.</span></div><div style="text-align:right"><div class="kicker">{_e(bundle['verb'])} certificate</div><div class="certid">{_e(bundle['certificate_id'])}</div></div></div>
<div class="meta"><div class="kicker">{date}</div><div>policy <b>{_e(pol.get('name', 'policy'))}</b> <b>{pol['sha256'][:8]}…</b></div><div>{prov}</div><div>{bundle['model']['n_trees']} trees · {bundle['model']['n_atoms']} atoms read</div></div>
<div class="hero">{_seal(state, date, color)}<div><div class="kicker" style="color:{color}">Verdict</div><div class="head" style="color:{color}">{HEADLINE.get(state, state)}</div>{claim_html}</div></div>
<div class="scope"><div class="card yes"><h4>Establishes</h4><ul><li>{estab}</li><li>Exact prices paid, quoted before the edit</li></ul></div>
<div class="card no"><h4>Does not establish</h4><ul><li>Absence of proxy effects — other features may jointly reconstruct a protected attribute</li><li>Fairness of outcomes</li><li>Causal claims about the world</li><li>Behavior off the data distribution</li></ul></div></div>
{findings}{action}{pvo}
<h3 class="sec">Where the instrument declined</h3><div style="font-size:13.5px">{refus}</div>
<h3 class="sec">Standing limits</h3><div class="limits"><ul>{limits}</ul></div>
<div class="verify"><div style="flex:1"><div class="kicker" style="color:#9CE2D8">Verify without trusting this document</div><code>$ pip install provable-verify<br>$ python -m provable_verify evidence.json {_e(a.get('model_out', '&lt;model&gt;'))}</code><div style="font-size:11px;margin-top:8px;color:#B8AFA4">Independent checker: own parser, own predictor, own transport — numpy and stdlib only, zero shared code.</div></div>{qr_block}</div>
<div class="foot">● exact — arithmetic fact of the artifact, no error bar exists &nbsp;·&nbsp; ○ estimate, with 95% interval &nbsp;·&nbsp; scope: the artifact, not the world<br>
<code>{_e(bundle['certificate_id'])}</code> = PV- + sha256(canonical evidence)[:16] &nbsp;·&nbsp; QR encodes the certificate id and full evidence hash &nbsp;·&nbsp; created {_e(bundle['created_utc'])} &nbsp;·&nbsp; provable {_e(bundle['environment'].get('provable', ''))} / xgboost {_e(bundle['environment'].get('xgboost', ''))}</div>
<div class="rule2" style="margin-top:14px"></div>
</div></body></html>"""
