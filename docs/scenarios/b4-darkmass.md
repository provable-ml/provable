# B4 · How much is the model not seeing, and can it say so?

The certified model trails the black box. Everyone has seen the gap; the question is what the gap *is*. This scenario builds a world where most of the signal is a smooth diagonal wave in **two** variables, `sin(2π(x2+x3))`, something a depth-6 engine rides easily and axis-aligned threshold [atoms](../glossary.md#atoms) cannot name, plus one honest step on `x1`. Then it reads the instrument's own accounting of what it cannot see.

```python
"""B4 -- part of the signal is a smooth diagonal wave in TWO variables: the engine can
ride it, axis-aligned threshold atoms cannot name it. Where does the variance live?"""
import numpy as np
import pandas as pd
from provable import CertifiedDiscovery

rng = np.random.default_rng(0)
n = 6000
X = pd.DataFrame(rng.uniform(0, 1, (n, 20)), columns=[f'x{j}' for j in range(20)])
y = (1.0 * np.sin(2 * np.pi * (X['x2'] + X['x3'])) + 0.7 * (X['x1'] >= 0.5) + rng.normal(0, 0.35, n)).values

cd = CertifiedDiscovery(seed=0)
report = cd.run(X, y)
```

Output (trimmed to the panels this page reads):

```text
-- certified claims (S2 one-pass partial effects; raw predicate scale; top 15 by p) --
+--------------+-------+----------+---------+---------+------+----------+
| atom         | src   |   effect |   ci_lo |   ci_hi |    t |        p |
|--------------+-------+----------+---------+---------+------+----------|
| x1>=0.500711 | cart  |   0.7573 |  0.5396 |   0.975 | 6.82 | 1.16e-11 |
+--------------+-------+----------+---------+---------+------+----------+

-- halo (claim-free) -- mode=none | dof=0.0 | lam= | auto: residual spectrum consistent with noise (dividend-mass excess z=2.0 < 3) -> no halo

-- performance (* = canonical tier used by diagnostics; hindsight lam disclosed) --
+------------------------------+------------+-----------+-------------+-------+
| model                        |   test MSE |   test R2 |   train MSE |   dof |
|------------------------------+------------+-----------+-------------+-------|
| refusal (intercept)          |    0.76347 |   -0.0004 |     0.74295 |   1   |
| certified core [S2-fit]      |    0.6371  |    0.1652 |     0.6136  |   2   |
| certified core [train-fit] * |    0.63594 |    0.1667 |     0.61316 |   2   |
| two-tier (core + halo) *     |    0.63594 |    0.1667 |     0.61316 |   2   |
| OLS-all atoms                |    0.68999 |    0.0959 |     0.55652 | 373   |
| ridge* atoms [hindsight lam] |    0.65162 |    0.1462 |     0.59471 | 110.2 |
| xgboost [raw features]       |    0.23578 |    0.6911 |     0.03818 | nan   |
+------------------------------+------------+-----------+-------------+-------+

-- theory panel (trimmed to the two toll rows) --
| certification toll upper (stationarity)   |   0.0952  | per-obs, proxy | l1 off-support from ridge*; c=1 |
| certification toll lower (concealment)    |   0.00914 | per-obs, proxy | kappa_d=29 in (2, zbar]; c0=1   |

-- dark-mass ledger on test (Var(y) = legible + dark-visible-to-engine + floor) --
+---------------+-----------------------+---------------------------------------------------+-------------------------------------+
|   Var(y_test) |   legible (certified) |   cert - engine (dark to cert, visible to engine) |   engine MSE (noise + dark to both) |
|---------------+-----------------------+---------------------------------------------------+-------------------------------------|
|       0.76319 |               0.12725 |                                           0.40016 |                             0.23578 |
+---------------+-----------------------+---------------------------------------------------+-------------------------------------+

PHASE: II-b (estimated frame-blind margin M_C <= 0); engine lead 0.4 vs in-span toll cap 0.0952 (excess, if any, = off-span wall D_perp + proxy slack)
```

## Reading it

**The claim.** One: `x1>=0.5007` at 0.757, the planted step at the planted threshold, and nothing else. Correct: nothing else in this world is nameable by single-column thresholds or their pairwise conjunctions.

**The ledger.** Var(y) = 0.763 splits three ways: **legible** 0.127 (the certified step, ≈ 0.7² · ¼); **dark to certification but visible to the engine** 0.400, the diagonal wave, which the engine converts to accuracy (test MSE 0.236 ≈ the noise floor of 0.35²) while certification cannot even *address* it; **floor** 0.236 (noise plus whatever neither sees). More than half the world's variance is structure the honest model can neither claim nor use; instead of hiding that, the instrument itemizes it. That number is the honest price tag on "interpretable" here.

**The wave is invisible to every marginal scan.** Look at the halo line: *residual spectrum consistent with noise → no halo*. After the certified step, no single atom correlates with the residual above chance: a diagonal wave has near-zero marginal signal on axis-aligned cells. Every one-feature-at-a-time screen in your current stack would call this residual noise. The engine, probing jointly, extracts 0.400 of variance from it. That is what "dark mass" means: not weak signal but signal your frame cannot see, whose *size* is nonetheless measurable from the engine's lead.

**The two diagnoses.** The phase line does the differential: engine lead **0.400** vs. in-span [toll](../glossary.md#toll) cap **0.0952**. If the gap were the lawful price of honesty (the cost of certifying instead of merely fitting) it would sit under the toll cap. It is 4× over. The excess is *off-span*: the signal is outside what this vocabulary can express at any sample size. Different disease, different cure: more data would buy nothing here; a richer dictionary might ([B2](b2-vocab.md) is that cure working on a curable case; a diagonal wave is the incurable one for axis-aligned atoms; the fix would be a feature, `x2 + x3`, not a flag). Compare [B1](b1-claims.md), where the same line read lead 0.059 vs. cap 0.552: gap under the cap, honesty affordable, keep the frame. [Phase](../glossary.md#phases) II-b names this regime: engine ahead, and no frame-blind method on these atoms would close it either (M_C ≤ 0).

## What's guaranteed

The ledger's `legible` entry inherits the claim ledger's FDR control; the split itself is measured on held-out test data. Toll bounds, penumbra, and M_C are labeled plug-in estimates with disclosed proxies: [`#est`](../glossary.md#exact-vs-est), never dressed as `#exact`.

---

*What the papers add*: Paper 3 is this page as theory, with the dark-mass ceiling formula (how much true signal *could* hide from every scan, a sharp threshold in one number), the unmeasured-variable shadow forecast (calibrated to 0.02 AUC in the graded test), and where hidden structure lives: [SSRN 7247305](https://doi.org/10.2139/ssrn.7247305) (archived: [Zenodo 21819925](https://doi.org/10.5281/zenodo.21819925)). The toll bounds quoted by the phase line are Paper 2's: [SSRN 7247302](https://doi.org/10.2139/ssrn.7247302) (archived: [Zenodo 21819895](https://doi.org/10.5281/zenodo.21819895)).
