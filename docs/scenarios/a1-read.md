# A1 · What is this model actually using?

You have a trained XGBoost model, perhaps one you didn't train. Feature-importance plots give you rankings; SHAP gives you sampled attributions. Neither tells you *the rules*: which exact conjunctions of thresholds the model fires, with what weights. `provable` reads that list, the [atom spectrum](../glossary.md#atoms), from the model file, exactly, with no sampling.

This is the running example the other A-scenarios continue: a small credit-style dataset with a pattern deliberately planted in the world (applicants under 35 in `group_A` are penalized) and a stock `XGBRegressor` trained on it.

```python
"""Running example for the A-scenarios: a small credit-style dataset with a planted
age x group_A pattern, and a stock XGBoost model trained on it. Seeded end to end."""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

rng = np.random.default_rng(0)
n = 9000
X = pd.DataFrame({
    'age':     rng.uniform(18, 80, n).round(1),
    'income':  (rng.lognormal(10.4, 0.5, n) / 1000).round(2),
    'tenure':  rng.uniform(0, 30, n).round(1),
    'group_A': (rng.random(n) < 0.4).astype(float),
    'util':    rng.random(n).round(3),
})
y = (0.006 * X['income'] + 0.015 * X['tenure'] - 0.4 * X['util']
     - 0.8 * ((X['age'] < 35) & (X['group_A'] == 1)) + rng.normal(0, 0.30, n)).values

tr, va, ho = slice(0, 6000), slice(6000, 7500), slice(7500, 9000)
X_train, X_val, X_holdout = X[tr], X[va], X[ho]
y_train, y_holdout = y[tr], y[ho]

model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, random_state=0, base_score=0.5)
model.fit(X_train, y_train)
model.save_model('model.json')
```

Now read the file:

```python
from provable import load

ad, spec = load('model.json')
print(f"atoms: {len(spec.atoms)}   features used: {[X.columns[i] for i in spec.features_used]}")
print("\ntop 10 rules by |coefficient|:")
for A, c in spec.top(10):
    rule = ' & '.join(f"{X.columns[f]}>={t:g}" for f, t in sorted(A))
    print(f"  {c:+.4f}   {rule}")
```

Output:

```text
atoms: 4860   features used: ['age', 'income', 'tenure', 'group_A', 'util']

top 10 rules by |coefficient|:
  +0.4406   income>=125.66 & util>=0.048
  -0.4081   group_A>=1
  +0.2553   income>=110.79 & util>=0.048
  +0.2047   age>=35.1 & group_A>=1
  -0.1706   group_A>=1 & util>=0.013
  +0.1556   age>=34.8 & group_A>=1
  -0.1478   income>=125.66 & util>=0.335
  -0.1455   age>=25 & tenure>=2.8
  +0.1330   tenure>=12.8
  -0.1309   income>=110.79
```

The listing was produced by the exact script above[^run]; run it and you get this table, byte for byte. The coefficients look nothing like the ones planted in `y`, and they should not: this is the model rewritten in a different basis. 200 trees became 4,860 signed AND-rules over `>=` thresholds, and the sum of those rules **is** the model: same predictions, to floating point (the measured identities are on [What "exact" means](../manual/exact.md)).

[^run]: All outputs in these scenarios are real captured runs of the code shown, in the stated environment ([Install](../manual/install.md)), with the seeds shown. The A-pages share this one setup.

## Reading it

The planted pattern is sitting in the open. The model learned the `age < 35 & group_A` penalty as a pair of atoms: `group_A>=1` at **−0.41** (everyone in the group starts penalized) partially refunded by `age>=35.1 & group_A>=1` at **+0.20** and `age>=34.8 & group_A>=1` at **+0.16** (the over-35 members get most of it back). A `>=`-vocabulary complement of "under 35 in group A", readable from the file alone: no data, no sampling, no attribution method.

Note what this listing is *not*: it is not an importance ranking, and it carries no claim about the world. It is an inventory: every rule in the artifact, exactly. Whether a rule is *allowed* is [A2](a2-audit.md)'s question; whether it reflects the *world* is the [bridge](bridge.md)'s.

## What's guaranteed

The spectrum is an identity of the artifact, tagged [`#exact`](../glossary.md#exact-vs-est) throughout: transported spectrum == stock margin to 2.4e-06 max abs on the shipped acceptance ledger, and == the reference Möbius calculus to 2.2e-16 on binary models.

---

*What the papers add*: the transport theorem behind this listing (leaf boxes → telescoped threshold atoms) and the calculus built on it: *Every Interaction Has a Price: An Exact Calculus for Tree Ensembles* (2026), [10.5281/zenodo.21819690](https://doi.org/10.5281/zenodo.21819690).
