# B1 · You screened 300 columns. Which effects survive?

The everyday disaster: scan hundreds of candidate features (signals, biomarkers, factors), keep what looks strong, publish the list, and half of it is selection noise, because the same data that *chose* the effects also *graded* them. The sieve is built to make that impossible: it proposes on one half (S1), certifies with **one** OLS pass on the reserved half (S2) under [BH(q)](../glossary.md#fdr-and-bhq), and every quoted effect, CI, t, p comes from that single [selection-conditional](../glossary.md#selection-conditional) pass.

Here: 300 columns, 296 pure noise, four planted effects: three thresholds and one conjunction.

```python
"""B1 -- you screened 300 candidate columns; which effects survive?
296 are pure noise; four effects are planted (three thresholds, one conjunction)."""
import numpy as np
import pandas as pd
from provable import CertifiedDiscovery

rng = np.random.default_rng(0)
n = 6000
X = pd.DataFrame(rng.normal(0, 1, (n, 300)), columns=[f'x{j}' for j in range(300)])
y = (0.50 * (X['x7'] >= 0.6) - 0.45 * (X['x23'] >= 0.0) + 0.40 * (X['x55'] >= 1.0)
     + 0.60 * ((X['x101'] >= 0.3) & (X['x148'] >= 0.3)) + rng.normal(0, 1.0, n)).values

cd = CertifiedDiscovery(stability_reps=8, seed=0)
report = cd.run(X, y)
```

Output (the report prints itself; theory panel and dark-mass ledger trimmed here; [B4](b4-darkmass.md) reads those):

```text
================ certified-discovery report ================
+------+------+------+--------+-----------+-------------------------------------------------------------------------------------+------------+--------+---------+
|    N |   S1 |   S2 |   test |   P atoms | pool composition                                                                    | task       |   seed | refit   |
|------+------+------+--------+-----------+-------------------------------------------------------------------------------------+------------+--------+---------|
| 6000 | 2250 | 2250 |   1500 |      3348 | {'guideline': 0, 'cut': 2400, 'cart': 715, 'onehot': 0, 'harvest': 38, 'conj': 195} | regression |      0 | train   |
+------+------+------+--------+-----------+-------------------------------------------------------------------------------------+------------+--------+---------+

-- sieve (RS proposal on S1: nu=0.50, R=4, entry |t|>=2.0) --
+---------+-------+---------+----------+-----------+
|   round |   new |   union |   S1 mse |   max |t| |
|---------+-------+---------+----------+-----------|
|       1 |   300 |     300 |  0.84424 |      9.33 |
|       2 |   113 |     413 |  0.6397  |      5.58 |
|       3 |    24 |     437 |  0.55714 |      3.21 |
|       4 |    21 |     458 |  0.51244 |      2.51 |
+---------+-------+---------+----------+-----------+

-- certification (one OLS pass on S2, BH q=0.10 over partial-t) --
+------+---------+--------+----------+---------+-----------+-----------+-------------+-------------+------------+----------+----------+----------+
|    P |   union |   core |   k cert |   entry |   use bar |   z(khat) |   zbar core |   zbar pool |   sqrt2lnP |   min|t| |   med|t| |   max|t| |
|------+---------+--------+----------+---------+-----------+-----------+-------------+-------------+------------+----------+----------+----------|
| 3348 |     458 |    120 |        3 |       2 |         1 |      3.02 |        3.34 |        4.17 |       4.03 |     3.94 |     4.99 |     8.11 |
+------+---------+--------+----------+---------+-----------+-----------+-------------+-------------+------------+----------+----------+----------+

-- certified claims (S2 one-pass partial effects; raw predicate scale; top 15 by p) --
+----------------+---------+----------+---------+---------+------+----------+-----------+-------+-------+
| atom           | src     |   effect |   ci_lo |   ci_hi |    t |        p | tier      |   f_A |   f_F |
|----------------+---------+----------+---------+---------+------+----------+-----------+-------+-------|
| x7>=0.607282   | harvest |   0.5349 |  0.4055 |  0.6642 | 8.11 | 8.44e-16 | invariant |  1    |  1    |
| x55>=0.991179  | cart    |   0.4873 |  0.2958 |  0.6788 | 4.99 | 6.55e-07 | invariant |  1    |  1    |
| x101>=0.128156 | cut     |   0.1823 |  0.0917 |  0.2729 | 3.94 | 8.24e-05 | fragile   |  0.25 |  0.88 |
+----------------+---------+----------+---------+---------+------+----------+-----------+-------+-------+

-- residual scan on train after certified core (marginal |t|, sigma_r-studentized) --
+----------------+---------+--------------------+
| |t| band       |   atoms |   mass sum(t^2-1)+ |
|----------------+---------+--------------------|
| (0, 1]         |    2032 |                0   |
| (1, 2]         |     964 |             1036.1 |
| (2, 4.17447]   |     299 |             1708.9 |
| (4.17447, inf] |      53 |             2326.1 |
+----------------+---------+--------------------+
(… theory panel, dark-mass ledger, predicted-vs-observed trimmed …)

PHASE: II-b (estimated frame-blind margin M_C <= 0); engine lead 0.059 vs in-span toll cap 0.552 (excess, if any, = off-span wall D_perp + proxy slack)
stability: B=8 replicates, certified-set sizes [5, 8, 5, 7, 4, 7, 7, 4]
elapsed: 47.2s
```

## Reading it

Out of 3,348 candidate atoms built from 300 columns, the ledger certifies **three claims**. `x7>=0.607` and `x55>=0.991` are exactly the planted thresholds, effects within a CI of the planted 0.50 and 0.40, **invariant** across all 8 replicates. `x101>=0.128` is the conjunction's footprint certified as a main effect: real family, [fragile](../glossary.md#stability-tiers) identity (`f_A=0.25`, `f_F=0.88`). The sieve knows *something* lives at x101, and says out loud that it hasn't pinned the exact cell. None of the 296 noise columns certified.

## Where is x23?

The fourth planted effect, `x23 >= 0.0` at −0.45, is missing, and the instrument does not hide it. Look inside the S2 pass:

```python
import numpy as np
tv = np.asarray(cd.core_res_.tvalues)[1:]
for pos, j in enumerate(cd.core_):
    if 'x23' in cd.atom_names_[j] and '&' not in cd.atom_names_[j]:
        print(f"{cd.atom_names_[j]:18s} S2 partial t = {tv[pos]:+.2f}")
```

```text
x23>=-0.117574     S2 partial t = -2.36
x23>=0.143936      S2 partial t = -2.51
```

The dictionary carried two near-duplicate cuts straddling the true 0.0 threshold; **both** entered the core, and the effect's evidence split between them: two partial t's of ≈ −2.4 where one representative would have carried ≈ −7. BH grades *identities*, not families, and declined both. That is the conservative direction: the certified list stays clean; the miss is priced, not papered over.

And the report announces it, twice. The residual scan shows **53 atoms above the certification bar** (mass 2326) still uncertified, and the completion check prints `stationary: False`, the instrument's own statement that signal remains beyond this ledger. The stability line agrees: replicate certified-set sizes run 4–8 against the primary 3: different splits resolve the split-evidence tie differently. A `PASS`-shaped silence would have been a lie; this is a refusal to overclaim, with the leftover located.

## What's guaranteed

The FDR of the three-claim ledger at q = 0.10, controlled by BH over one selection-conditional S2 pass, full stop. Not guaranteed: completeness. The diagnostics exist precisely because the guarantee is one-sided.

---

*What the papers add*: Paper 3 is the theory of this page's second half: why a procedure misses (search effort, data shortage, or geometry), what each miss would cost in rows to buy back (the naive quote overprices ~3×), and which misses can never be bought at any budget: [SSRN 7247305](https://doi.org/10.2139/ssrn.7247305) (archived: [Zenodo 21819925](https://doi.org/10.5281/zenodo.21819925)).
