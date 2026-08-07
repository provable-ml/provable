# About provable {: style="display:none" }

<div class="pv-eyebrow">Exact audit &middot; certified discovery &middot; independent verification</div>

<p class="pv-thesis">A tree ensemble on binary inputs, gradient-boosted or otherwise, is exactly a finite sum of AND-rules: in closed form, with no data. Not an approximation of the model. The model.</p>

<div class="pv-eq"><i>F</i>(<i>x</i>) = <span class="big">&sum;</span><sub><i>A</i>&#8838;[<i>n</i>]</sub> <i>m</i><sub><i>A</i></sub> <span class="big">&prod;</span><sub><i>i</i>&isin;<i>A</i></sub> <i>x</i><sub><i>i</i></sub></div>
<div class="pv-eq-note">x &isin; {0,1}&#8319; &middot; coefficients readable from the leaves alone</div>

<p class="pv-thesis"><code>provable</code> turns that identity into operations you can run: read every rule and its exact weight, prove what never fires together, remove chosen rules without retraining, and know the accuracy cost before the edit. On raw data, the same rule vocabulary drives a certified-discovery sieve: an FDR-controlled ledger of effects that survive one honest pass, a verdict when nothing does, and a priced account of what no threshold scan could see.</p>

<p class="pv-thesis">Every operation emits a certificate, and the checker is not the engine: <code>provable-verify</code> re-implements its own parser, its own predictor, and its own transport from the xgboost JSON spec, and shares no code with <code>provable</code>. The first thing to run:</p>

```console
$ pip install provable-verify
$ git clone https://github.com/provable-ml/provable && cd provable
$ python -m provable_verify dossier_demo/evidence.json dossier_demo/model_v2.json
```

```text
provable-verify v0.9.0 (independent; numpy+stdlib; shares no code with provable)
  [PASS] 1 bundle hash = certificate id
  [PASS] 2 model file hash matches bundle
  [PASS] 3 policy hash matches bundle
  [PASS] 4 structural probe (own transport)
  [PASS] 5 behavioral probe (own predictor)
  [PASS] 6 verdict supported by own probes
  [PASS] 7 limits panel present
  measured: structural=3.522e-09  probe=7.451e-09
VERIFICATION: PASS
```

<div class="pv-cap">The checker needs numpy and the standard library, nothing else; independence is machine-checked, and CI fails on any verifier import outside that set. <a href="../scenarios/a4-verify/">Scenario A4</a> shows what happens to a forged certificate.</div>

<div class="pv-label">Status</div>

<ul class="pv-ledger">
  <li><span class="pv-key" style="min-width:auto">pip install provable</span><span class="pv-leader"></span><span class="pv-val"><a href="../manual/install/">0.9.0</a></span></li>
  <li><span class="pv-key" style="min-width:auto">Source</span><span class="pv-leader"></span><span class="pv-val"><a href="https://github.com/provable-ml/provable">GitHub</a> &middot; <a href="https://pypi.org/project/provable/">PyPI</a></span></li>
  <li><span class="pv-key" style="min-width:auto">License</span><span class="pv-leader"></span><span class="pv-val"><a href="https://github.com/provable-ml/provable/blob/main/LICENSE">MIT License</a></span></li>
  <li><span class="pv-key" style="min-width:auto">Papers</span><span class="pv-leader"></span><span class="pv-val"><a href="../cite/">a trilogy of companion preprints</a></span></li>
</ul>
