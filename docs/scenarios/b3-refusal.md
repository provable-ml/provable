# B3 · It came back empty. Now what?

Every discovery tool you have used returns *something* (a top-10 list, a ranking, a heat map) because returning something is what tools do. This one has a third option, and it is the point of the whole design: **k = 0 is a verdict, not an error.** Here is the run whose correct answer is nothing: 25 candidate columns, `y` pure noise.

```python
"""B3 -- 25 candidate columns, no signal at all. The run whose correct answer is nothing."""
import numpy as np
import pandas as pd
from provable import CertifiedDiscovery

rng = np.random.default_rng(0)
n = 1200
X = pd.DataFrame(rng.normal(0, 1, (n, 25)), columns=[f'x{j}' for j in range(25)])
y = rng.normal(0, 1, n)

cd = CertifiedDiscovery(seed=0)
report = cd.run(X, y)
```

Output (trimmed to the load-bearing panels):

```text
================ certified-discovery report ================
+------+------+------+--------+-----------+-----------------------------------------------------------------------------------+------------+--------+---------+
|    N |   S1 |   S2 |   test |   P atoms | pool composition                                                                  | task       |   seed | refit   |
|------+------+------+--------+-----------+-----------------------------------------------------------------------------------+------------+--------+---------|
| 1200 |  450 |  450 |    300 |       451 | {'guideline': 0, 'cut': 200, 'cart': 42, 'onehot': 0, 'harvest': 39, 'conj': 170} | regression |      0 | train   |
+------+------+------+--------+-----------+-----------------------------------------------------------------------------------+------------+--------+---------+

-- sieve (RS proposal on S1: nu=0.50, R=4, entry |t|>=2.0) --
+---------+-------+---------+----------+-----------+
|   round |   new |   union |   S1 mse |   max |t| |
|---------+-------+---------+----------+-----------|
|       1 |   112 |     112 |  0.80411 |      5.02 |
|       2 |     5 |     117 |  0.69815 |      2.91 |
|       3 |     1 |     118 |  0.66332 |      2.09 |
|       4 |     1 |     119 |  0.64546 |      2.01 |
+---------+-------+---------+----------+-----------+

-- certification (one OLS pass on S2, BH q=0.10 over partial-t) --
+-----+---------+--------+----------+---------+-----------+-----------+-------------+-------------+------------+----------+----------+----------+
|   P |   union |   core |   k cert |   entry |   use bar |   z(khat) |   zbar core |   zbar pool |   sqrt2lnP |   min|t| |   med|t| |   max|t| |
|-----+---------+--------+----------+---------+-----------+-----------+-------------+-------------+------------+----------+----------+----------|
| 451 |     119 |     40 |        0 |       2 |         1 |       nan |        3.02 |        3.69 |        3.5 |      nan |      nan |      nan |
+-----+---------+--------+----------+---------+-----------+-----------+-------------+-------------+------------+----------+----------+----------+

-- dark-mass ledger on test (Var(y) = legible + dark-visible-to-engine + floor) --
+---------------+-----------------------+---------------------------------------------------+-------------------------------------+
|   Var(y_test) |   legible (certified) |   cert - engine (dark to cert, visible to engine) |   engine MSE (noise + dark to both) |
|---------------+-----------------------+---------------------------------------------------+-------------------------------------|
|        1.0298 |              -0.00021 |                                           0.00067 |                             1.02934 |
+---------------+-----------------------+---------------------------------------------------+-------------------------------------+

PHASE: I  (instrument refuses: nothing certifiable at q=0.1; refusal vs engine test MSE 1.03 vs 1.029)
elapsed: 1.2s
```

## Reading it

Notice how hard the run *tried*. On the proposal half, 119 atoms cleared the entry bar: in pure noise, with 451 candidates, some always will; that is what noise does, and the proposal stage is allowed to chase it because nothing it selects is a claim. The greedy core kept 40. Then the one S2 pass under BH(q = 0.10) certified **zero**. The overfitting happened where it is safe (S1 mse "improving" from 0.80 to 0.65 is the sieve fitting noise, on purpose, on the half whose job is to propose) and was refused where it counts.

The phase line is the verdict in one sentence: **`PHASE: I (instrument refuses)`**, and it shows its work: the refusal baseline (predict the mean) tests at 1.030 against the tuned XGBoost engine's 1.029. *The black box also found nothing here*; it just doesn't have a way to say so. The dark-mass ledger writes the same conclusion in variance units: legible ≈ 0, engine-visible-beyond-certified ≈ 0.0007, everything else floor.

So "nothing found" comes back informative three ways: a controlled statement (nothing certifiable at this FDR budget), a competitive benchmark (refusal ties the engine; this is [phase I](../glossary.md#phases), where honesty costs nothing because there is nothing to buy), and an accounting (no dark mass visible to the engine either). The [verdict grammar](../manual/verdicts.md) codes the general stance as R-01, *nothing certifiable at the agreed FDR budget*: never a crash, never a shrug, and never a decorated top-10 list of noise.

## What's guaranteed

Exactly what was guaranteed when k > 0: the FDR of the (empty) ledger. An empty certified list under a controlled procedure is a finding about the data at this budget and vocabulary: not a malfunction, and not proof that nothing exists ([B1](b1-claims.md) shows the diagnostics that say "signal remains"; here they stay quiet).

---

*What the papers add*: Paper 3 turns the empty list into a program: a hard bound on what could still be hiding, and a playbook that in the graded test converted zero findings into two at zero additional data cost: [SSRN 7247305](https://doi.org/10.2139/ssrn.7247305) (archived: [Zenodo 21819925](https://doi.org/10.5281/zenodo.21819925)).
