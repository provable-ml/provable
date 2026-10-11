# Install

From PyPI:

```
pip install provable
pip install provable-verify
```

Working from a clone (editable installs of both distributions):

```
pip install -e .            # provable (engine): pulls numpy, pandas, scipy, scikit-learn,
pip install -e ./verify     # provable-verify (checker): numpy only     statsmodels, xgboost, tabulate, plotly, qrcode
```

After this, `import provable` works anywhere in the venv and the verifier runs as `python -m provable_verify <evidence.json> <model.json>` from any directory (file arguments are relative to your cwd; the demo dossier lives in `dossier_demo/`).

Tested on Python 3.10 to 3.14: at the minimum versions (numpy 1.26.4, pandas 2.2.3, scipy 1.13.1, scikit-learn 1.5.2, statsmodels 0.14.4) with xgboost 2.0.3, 2.1.4 and 3.0.5, and at current releases with xgboost 3.4. CI runs the same matrix. The verifier needs numpy only.

First thing to run after a clone: verify the shipped dossier with the independent checker:

```
python -m provable_verify dossier_demo/evidence.json dossier_demo/model_v2.json
```

## One repo, two distributions

| Path | Distribution | What it is | Depends on |
|---|---|---|---|
| `src/provable/` | `provable` | the instrument: load / spectrum / audit / enforce / discover, and the certificate layer | numpy ≥ 1.26, pandas ≥ 2.2, scipy ≥ 1.13, scikit-learn ≥ 1.5, statsmodels ≥ 0.14, xgboost ≥ 2.0, tabulate, plotly, qrcode |
| `verify/src/provable_verify/` | `provable-verify` | the independent checker examiners run | **numpy only** |

The verifier re-implements from the xgboost JSON spec its own parser, its own vectorized tree-walk predictor, and its own transport expansion. Independence is machine-checked: `tests/test_firewall.py` fails CI on any verifier import outside {stdlib, numpy} or any import of the engine.

The checker also ships standalone at [github.com/provable-ml/provable-verify](https://github.com/provable-ml/provable-verify), a byte-identical mirror of `verify/`; changes land in the main repository first and are mirrored there.
