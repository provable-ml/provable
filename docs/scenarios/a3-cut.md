# A3 · Remove the rules without retraining, price quoted first

*Continues the running example from [A1](a1-read.md); the violation was found in [A2](a2-audit.md).*

Retraining under a constraint gives you a *different model* and a promise. Surgery gives you *the same model minus exactly the forbidden rules*: one small appended tree per removed atom (−coef on the conjunction, 0 elsewhere), so every prediction off the forbidden rules is **bit-identical** by construction, and the output file still loads in stock `xgboost`. The accuracy cost is computed from the spectrum *before* anything is touched.

```python
import xgboost as xgb
from provable import audit, enforce, Policy, between

policy = Policy([between('age', 'group_A')], name='fair_lending_demo')
b = enforce('model.json', policy, X_train, X_holdout, 'model_v2.json', y_holdout=y_holdout, out_dir='dossier_enforce')

act = b['action']
print(f"verdict: {b['verdict']['state']} -- certified absent: {b['verdict']['certified_absent']}")
print(f"atoms removed / trees appended : {act['atoms_removed']} / {act['trees_appended']}")
print(f"structural residual   #exact   : {act['structural_residual']:.3e}")
print(f"behavioral grid probe #exact   : {act['grid_probe_max']:.3e}")
print(f"stock reload diff on/off rule  : {act['stock_diff_on']:.3e} / {act['stock_diff_off']:.3e}")
print(f"holdout MSE before -> after    : {act['holdout_mse_before']:.4f} -> {act['holdout_mse_after']:.4f}  (delta {act['holdout_delta']:+.4f} +/- {act['holdout_delta_ci']:.4f})")
pv = b['predicted_vs_observed'][0]
print(f"price predicted vs realized    : {pv['predicted']:.4f} vs {pv['observed']:.4f}  ({pv['status']})")

m2 = xgb.Booster(); m2.load_model('model_v2.json')          # loads in STOCK xgboost

# Re-audit the edited file AT THE CERTIFICATION TOLERANCE. Surgery is float32
# write-back: the transported spectrum of model_v2 still carries ~1e-9 residue
# on the family (squared mass ~1e-17). tolerance=0 would flag that residue;
# the certificate states its gates (structural <= 1e-4) instead of hiding them.
pol2 = Policy([between('age', 'group_A')], name='fair_lending_demo', tolerance=1e-4)
a2 = audit('model_v2.json', pol2, X_val)
print(f"\nre-audit of model_v2.json      : {a2['verdict']['state']}  "
      f"(residual family mass {a2['findings'][0]['data_mass']:.1e})")
```

Output:

```text
verdict: PASS -- certified absent: ['age × group_A']
atoms removed / trees appended : 326 / 326
structural residual   #exact   : 5.000e-09
behavioral grid probe #exact   : 6.557e-07
stock reload diff on/off rule  : 5.347e-07 / 0.000e+00
holdout MSE before -> after    : 0.0908 -> 0.2722  (delta +0.1814 +/- 0.0190)
price predicted vs realized    : 0.1762 vs 0.1702  (within 25%)

re-audit of model_v2.json      : PASS  (residual family mass 2.6e-17)
```

## Reading it

Line by line, in the order a reviewer would ask:

- **Is the family really gone?** Two independent probes on the *edited file*. The structural probe re-transports `model_v2.json` and finds the largest surviving forbidden coefficient: `5.0e-09` against a gate of `1e-4`. The behavioral probe evaluates second differences on the full age×group threshold grid at probed contexts: `6.6e-07` against `1e-3`. Both are [`#exact`](../glossary.md#exact-vs-est): arithmetic facts of the file, not estimates.
- **Did anything else change?** Off-rule rows after a *stock xgboost reload*: `0.000e+00`: bit-identical, and that zero is a certification gate, not a rounding courtesy.
- **What did it cost?** Holdout MSE 0.0908 → 0.2722. The pattern was real signal in this world (we planted it), so removing it costs accuracy, which is precisely why the price matters: it was quoted from the spectrum as **0.1762** *before* the edit and realized at **0.1702** on holdout, within the 25% band the certificate grades itself against. You knew the bill before signing.
- **Does a fresh audit agree?** Re-auditing the edited file at the stated tolerance: PASS, residual family mass 2.6e-17. The comment in the code is the honest part: appended trees are written back in float32 by the native format, so "exact" always means *at the stated tolerance*; the gates are printed on the certificate rather than hidden in the fine print.

Everything above, plus the findings, the limits panel, and the environment, lands in `dossier_enforce/` as a canonical `evidence.json` (whose hash *is* the certificate id) with human renderings `certificate.md` / `certificate.html`. [A4](a4-verify.md) re-checks it with code that shares nothing with this package.

## What's guaranteed

Exact removal of the named rule families from the artifact, priced, with a still-native output file: within the stated float32 write-back gates, structurally and behaviorally probed, independently re-checkable. Not guaranteed, ever: that a proxy hasn't reconstructed the pattern, that outcomes are fair, that anything causal was removed. The limits panel on the certificate is non-removable; the verifier fails if it is stripped.

---

*What the papers add*: the appended-tree surgery theorem, the pricing identity, and the certificate-of-infeasibility construction for removal policies that can't be met on budget: Paper 1, [10.5281/zenodo.21819690](https://doi.org/10.5281/zenodo.21819690).
