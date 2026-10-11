# C1 · From a panel to a citable claim

*Opens the C track: the certificate layer over the sieve of [B1](b1-claims.md). This page's `demo_book` is reused by C2 and C3.*

You ran a study. The sieve certified something. Now you want the result to be *citable*: a record that states exactly what was claimed, at what error rate, over which family, tested on which rows, so that a later reader, or a later you, can reuse it without silently re-spending the data. `certify` is the one-call form: it runs the instrument of [B1](b1-claims.md), issues a [certificate](../manual/certificates.md) into a book, and returns the standard report.

```python
import numpy as np, pandas as pd
import provable as pv

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(6000, 8)), columns=[f"x{i}" for i in range(8)])
y = 0.8 * (X["x0"] >= 0.6) - 0.7 * ((X["x1"] >= 0) & (X["x2"] >= 0)) + rng.normal(size=6000)

rep = pv.certify(X, y, q=0.10, ledger="demo_book", dataset_tag="demo", seed=0)
print(rep.render())
```

Output (provable 0.10.0, xgboost 3.4.0, seed 0; the structure is what to read, the exact decimals depend on your xgboost build):

```text
-- certify: standard report --
+----------------+------------------+
| manifest       |                  |
|----------------+------------------|
| x_sha          | b7961df3aa784bf8 |
| y_sha          | 51eb99d5af5c0096 |
| dictionary_sha | d15a2e43ee8cbd53 |
| config_sha     | 1e43e0092612370e |
| instrument_sha | b7c63882317237f6 |
| seed           | 0                |
| provable       | 0.10.0           |
+----------------+------------------+
+------+-----+--------+----------+--------+-----+
|    n |   P |   core |   k cert |   khat |   q |
|------+-----+--------+----------+--------+-----|
| 6000 | 279 |     63 |        3 |      3 | 0.1 |
+------+-----+--------+----------+--------+-----+
engine vs certified core (test R2): engine 0.1405 vs core 0.18; lead -0.03945
-- certified claims (selective intervals at alpha_sel = khat*q/m = 0.00476; FCR) --
+--------------------------------+----------+---------+---------+-------------+-------------+--------+-----------+
| atom                           |   effect |   ci_lo |   ci_hi |   ci_lo_sel |   ci_hi_sel |      t |         p |
|--------------------------------+----------+---------+---------+-------------+-------------+--------+-----------|
| x0>=0.598249                   |   0.6972 |  0.3741 |  1.02   |      0.2317 |      1.163  |  4.232 | 2.412e-05 |
| (x1>=-0.152471)&(x2>=0.141445) |  -0.4578 | -0.6705 | -0.2451 |     -0.7643 |     -0.1513 | -4.221 | 2.533e-05 |
| (x1>=0.152503)&(x2>=-0.131637) |  -0.3319 | -0.5423 | -0.1215 |     -0.6351 |     -0.0287 | -3.093 | 0.002006  |
+--------------------------------+----------+---------+---------+-------------+-------------+--------+-----------+
-- atom card (est; marginal per-atom effects on S2 and held-out test, never the proposal rows; net across cells via rep.core_fit()) --
+--------------------------------+-------------+-----------+-------------+----------+---------+------------+-----------+------------+--------------+
| atom                           |   direction |   fire_s2 |   fire_test |   eff_s2 |   se_s2 |   eff_test |   se_test |   agree_s2 |   agree_test |
|--------------------------------+-------------+-----------+-------------+----------+---------+------------+-----------+------------+--------------|
| x0>=0.598249                   |           1 |    0.2707 |       0.28  |   0.8109 | 0.05098 |     0.8747 |   0.06102 |     0.7143 |       0.7381 |
| (x1>=-0.152471)&(x2>=0.141445) |          -1 |    0.2453 |       0.246 |  -0.6233 | 0.05198 |    -0.593  |   0.0662  |     0.6612 |       0.6585 |
| (x1>=0.152503)&(x2>=-0.131637) |          -1 |    0.2444 |       0.25  |  -0.5934 | 0.05263 |    -0.5145 |   0.06593 |     0.6455 |       0.6293 |
+--------------------------------+-------------+-----------+-------------+----------+---------+------------+-----------+------------+--------------+
-- penumbra (Thm 6a band; null floor subtracted, est) --
+-----------------+------------------+--------------+-------------+--------------------------------------+
|   raw band mass |   m0 (est nulls) |   null floor |   corrected | grade                                |
|-----------------+------------------+--------------+-------------+--------------------------------------|
|          148.14 |              279 |       133.66 |       14.48 | est: Storey null count at lambda=0.5 |
+-----------------+------------------+--------------+-------------+--------------------------------------+
ledger: 1 carrier certificate appended to demo_book (CD-3C9272B344A2BC67)
book after this run: charged 0.1000 of budget 0.1000, remaining 0.0000
```

## Reading it, block by block

**The manifest** pins what ran: content hashes of the data (`x_sha`, `y_sha`), the built [atom](../glossary.md#atoms) dictionary, the configuration, and the instrument's own source (`instrument_sha` is the sha of the facade module that produced this report). Two runs with the same manifest are the same experiment; any silent change to data, dictionary, config, or tooling moves a hash.

**The state line** is [B1](b1-claims.md)'s gate in six numbers: 6000 rows, a pool of 279 atoms, a core of 63 entering the one-pass S2 regression, and khat = 3 certified at BH q = 0.10.

**engine vs certified core** is one finding, printed every run: the free gradient-boosted engine against the frozen-dictionary certified core, on held-out test R2. Here the lead is *negative*: the certified three-atom core beats the engine, because this world really is three threshold rules and the engine spends capacity elsewhere. On real panels the lead is often positive; a large positive lead says mass is dark to the dictionary but visible to the engine, which is [B4](b4-darkmass.md)'s subject.

**The claims** are partial effects in one joint regression over the 63 core members, so they read jointly, not as standalone effects. The planted `x0` cut is recovered as one atom; the planted `(x1>=0)&(x2>=0)` interaction is recovered as *two* overlapping conjunction cells whose effects split the planted mass between them. That is normal: the dictionary proposes many near-duplicate shards of one true rule, and the joint pass distributes the mass. Read families, not rows.

**Selective intervals** are the second CI pair. The marginal `ci_lo, ci_hi` are ordinary 95% intervals and are biased for a *selected* set; `ci_lo_sel, ci_hi_sel` are at alpha_sel = khat·q/m, the FCR-calibrated level for claims that survived selection. Here khat·q/m = 3·0.10/63 ≈ 0.005, so the selective intervals are *wider* than the marginal ones. When khat is large relative to m, alpha_sel can exceed 0.05 and the selective intervals come out *narrower*; both directions are correct, the level tracks the selection.

**The atom card** sits between the claims and the penumbra, labeled est and never budget-bearing. Its header is the contract: these are *marginal per-atom* effects (E[y | fired] minus E[y | not fired], with firing read on S2 and on held-out test, never on the proposal rows that selected the atoms), and the net across cells is `rep.core_fit()`, not a sum of card rows. The two interaction shards are the case that bites: each carries a strong negative when-fired effect, but a row firing both is one physical state, and only the certified tier's fitted value nets it. `rep.core_fit()` returns exactly that column (the certified tier is one OLS of the target on the certified atoms alone), and `rep.core_fit(X_new)` prices fresh rows through the stored dictionary. Declare `unit_cost=` in the certify call and the card adds a net line in the certified direction. Grading those committed directions later on fresh data is [C5](c5-grade.md).

**The corrected penumbra** takes [B4](b4-darkmass.md)'s raw evidence-band mass and subtracts the estimated null floor (a Storey null count times the exact per-null band mass). What remains is band mass above what pure noise would put there. With a data-adaptive dictionary part of that residue is selection mass from the dictionary build itself; the report's JSON carries that caveat in the block's `note` field.

**The book line** is the point of the exercise: one carrier [certificate](../manual/certificates.md) with a content-addressed id (`CD-…`), carrying the claim, the guarantee (FDR, q, the full core family with its p-vector), the fingerprinted test and proposal [lineage](../glossary.md#lineage), the certified support, and a Gram descriptor. The account is charged 0.10 of 0.10: the book's whole error budget is spent on this one family. [C3](c3-book.md) is about that account.

## The refusal you want

`certify` accepts the manifest you *expected*. On any mismatch it refuses before touching the book:

```python
try:
    pv.certify(X, y, q=0.10, ledger="demo_book", dataset_tag="demo", seed=0,
               expect_manifest={"x_sha": "0000000000000000"})
except pv.ManifestMismatch as e:
    print(f"ManifestMismatch: {e}")
print("register ops in demo_book/ledger.jsonl:",
      sum(1 for _ in open("demo_book/ledger.jsonl")))
```

```text
ManifestMismatch: refused before any ledger write: manifest mismatch (x_sha: expected 0000000000000000, saw b7961df3aa784bf8)
register ops in demo_book/ledger.jsonl: 1
```

The ledger still holds exactly the one certificate. Wire the previous run's manifest into the next run's `expect_manifest` and a changed input can never silently spend the book.

## What is guaranteed, and what is diagnostic

The guarantee is the gate's: expected false-discovery proportion at most q over the declared core family, on rows the selection never saw, plus FCR-calibrated intervals on the certified set. Everything else in the report, the engine comparison, the penumbra, the diagnostics inside `rep.engine`, is instrumentation: exact where marked `#exact`, estimated where marked `est`, and never part of the claim.

Two panels can be run against two books and reused later without either spending the other; when a certificate is consumed again, admission and pricing take over, which is [C2](c2-reuse.md). An empty run (khat = 0) issues nothing, charges nothing, and reports a refusal line; [B3](b3-refusal.md) is about what an empty run is worth.
