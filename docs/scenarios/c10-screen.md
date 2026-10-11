# C10 · Screen the column before you buy it

*[C7](c7-purchase.md) priced what a missing column would have to be; a vendor now claims to have one. `screen` is the test that claim faces before any money or any budget moves: the reliability statistic of the paper's screening law, its exact null, its one-sided band, the operating requirement `m_req`, and — with the [C9](c9-characterize.md) findings on the table — the build / screen-then-buy / refuse trichotomy. Like the other two verbs it is free of acquisition: a PASS is a statement about a sample column, not a purchase, and nothing here writes to any book. Everything is on a synthetic panel.*

The statistic is Def 4.3: residualize the candidate against the licensed span, standardize it, couple it to the **aim block** `w`, and form `R = m_eff ||g_hat||^2 / sigma_hat^2` with `lam_hat^2 = (R - J) / (m_eff - 1)`. Three laws make it usable at a distance: `E[lam_hat^2] = rho^2` exactly (Thm 4.6(ii)); `R ~ chi2_J` exactly under the null, whatever the scale (Cor 4.7); and the fluctuation of `xi(lam_hat)` is `1/sqrt(m_eff)` to first order on the stabilized scale of Lem 4.8, which is where the one-sided band lives (Thm 4.11(i)).

The **floor** `lam_min` is mandatory and single: either the reliability price a columns-leg purchase order computed (`po=`, the `lam_A` of C7, with `lam*` printed beside it), or a declared value with its derivation, labelled `DECLARED`. Passing both refuses. The **verdict rule** is Def 4.4's band form: `PASS` when the one-sided lower end clears the floor, `REJECT` when the upper end fails it, `CONTINUE` between — and at `J = 1` a sign that flips between the two halves of the split rejects a candidate whose amplitude alone would have survived.

```python
import numpy as np
import pandas as pd
import provable as pv

rng = np.random.default_rng(7)
n = 7500
X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
base = (X["f0"] >= 0.3).astype(float) * 0.80 + ((X["f1"] >= 0) & (X["f2"] >= 0)).astype(float) * 0.12
y = (rng.random(n) < np.clip(base + 0.03, 0, 1)).astype(float)
clock = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
rep = pv.certify(X[:5000], y[:5000], q=0.10, seed=0,
                 clock=(clock[:5000], clock[:5000] + pd.Timedelta(hours=1)))
sh = pv.shadow(rep, X[5002:].reset_index(drop=True), y[5002:], clock[5002:])
ch = pv.characterize(sh, atoms={"archive": {"book_flow": {
    "values": 0.3 * sh.rows["e"].to_numpy() + rng.normal(size=sh.m),
    "lag_class": "post-decision"}}})

e = sh.rows["e"].to_numpy()
candidates = {
    "vendor_a": {"values": 0.5 * e + rng.normal(size=sh.m),
                 "lag_class": "decision-time", "archive_built": False},
    "vendor_b": {"values": rng.normal(size=sh.m),
                 "lag_class": "decision-time", "archive_built": False},
    "own_build": {"values": 0.35 * e + rng.normal(size=sh.m),
                  "lag_class": "decision-time", "archive_built": True},
}
sc = pv.screen(sh, candidates, floor=(0.12, "the aim floor this study declared"), findings=ch)
print(sc)
```

```text
-- Screening (Defs 4.3-4.4; band ends one-sided at 0.05, Lem 4.8 / Thm 4.11(i); split interleaved-day) --
+-------------+--------------+---------------+--------------------------+------------------+----------------------------------+--------+------------+-----------+---------+------------------+
| candidate   |   lam screen |   lam confirm |   lam full (descriptive) | verdict band     |   p_null (exact chi2_J, Cor 4.7) |   DEFF | sign s/c   | verdict   |     gap | m_req eff/rows   |
|-------------+--------------+---------------+--------------------------+------------------+----------------------------------+--------+------------+-----------+---------+------------------|
| vendor_a    |       0.1148 |        0.1379 |                   0.1282 | [0.0915, 0.1842] |                         6.61e-07 |   1    | +1/+1      | CONTINUE  |  0.0179 | 6,710 / 13,430   |
| vendor_b    |       0      |        0      |                   0      | [0.0000, 0.0466] |                         0.952    |   1    | -1/+1      | REJECT    | -0.12   | 453 / 906        |
| own_build   |       0.1323 |        0.0974 |                   0.1167 | [0.0500, 0.1448] |                         0.000434 |   1.04 | +1/+1      | CONTINUE  | -0.0226 | 4,534 / 9,442    |
+-------------+--------------+---------------+--------------------------+------------------+----------------------------------+--------+------------+-----------+---------+------------------+
counts by verdict: CONTINUE 2, REJECT 1
floor lam_min = 0.1200 [DECLARED]; derivation: the aim floor this study declared
aim block: the shadow itself, standardized (J = 1); the Screening says so because no columns leg was passed
m_req is the operating requirement of Def 4.4, achieved by the threshold test at t* = lam_min^2 + dlam^2 (Thm 4.9(a)); the plug-in band rule inherits it to first order (Thm 4.9(f)). Under a split the m_eff figure reads on the verdict slice and the rows figure carries the split factor.
+--------+-----------------+----------------+
|   dlam |   m_req (m_eff) |   m_req (rows) |
|--------+-----------------+----------------|
|   0.02 |           5,561 |         11,130 |
|   0.05 |           1,318 |          2,638 |
|   0.1  |             550 |          1,101 |
+--------+-----------------+----------------+
trichotomy (Cor 4.15, Table 3): floor 0.1200 vs lam_arch 0.0861 vs lam_cap 0.9971 -> screen, then buy (completion-contingent; the middle verdict is never certifiable, Cor 4.15)
archive-built floor by exhibition: 0.0974 (largest confirmed lam_hat among archive-built candidates; the exhibiting band certifies the build verdict, Cor 4.15)
PASS is one-sided valid for the raw reliability: aim error understates only (Thm 4.11(i)). REJECT is decision-valid for the declared promise only (Thm 4.11(iv)).
[K = 3: the screen-slice ranking values are maxima over K selections and print descriptive; the verdict bands sit on the disjoint confirm slice, honest because the selection is independent of the confirm statistic (Thm 5.7); an unsplit winner quote would overstate by up to sqrt(2 log K) noise scale (Thm 5.6(iv))]
[lam_cap is the upper envelope of Def 4.14: lam_arch^2 plus the walled remainder share, the walled aim budget bounded by the aim projection of the identified sum; est]
```

Reading it, top to bottom:

- **Three `lam_hat` columns, one honest one.** With `K > 1` candidates, ranking happens on the screen half of the interleaved-day split and every verdict band sits on the disjoint confirm half — honest because the selection is independent of the confirm statistic (Thm 5.7). The full-slice value prints as *descriptive only*: quoting the winner's unsplit number would overstate by up to `sqrt(2 log K)` noise scale (Thm 5.6(iv)). At `K = 1` the split is unnecessary — the full-slice band is the honest quote and the halves serve only the sign gate.
- **`p_null`** is the exact chi-square tail of the confirm-slice `R` (Cor 4.7): `vendor_b` is comfortably a null and its band's upper end sits under the floor, so it is `REJECT` on amplitude — no asymptotics were consumed anywhere in that sentence.
- **`m_req`** is the operating requirement of Def 4.4 at the candidate's own gap, and the grid below it at the declared `dlam` values: the row count at which the threshold test at `t* = lam_min^2 + dlam^2` meets both the `eps_miss` and `eps_clear` budgets (Thm 4.9(a)); the shipped band rule inherits the same requirement to first order (Thm 4.9(f)). Both `CONTINUE` rows say the same practical thing: the gap each candidate actually shows needs far more rows than this slice carries, which is precisely what `CONTINUE` means. When the floor and the resolution collide (`lam_min <= dlam`), the grid flags the corner: the clear side degenerates to exact null testing (Thm 4.9(e)) and the requirement follows the corner row of Thm 4.9(b).
- **The trichotomy line** (Cor 4.15, Table 3) places the floor against the C9 findings: below `lam_arch`, *build, do not buy* — an archive-constructible realizer exists; above `lam_cap` — the upper envelope `lam_arch^2` plus the walled remainder share — *refuse*, and a pre-acquisition refusal is an amplitude statement, never a reliability statement; between, *screen then buy*, and the middle verdict is never certifiable before completion. Here the declared floor sits just above `lam_arch`, so the run lands in the middle, and the `archive_built` exhibit's confirmed `lam_hat` prints as the exhibition floor the build verdict would rest on.
- **The two validity clauses** close every read: `PASS` is one-sided valid for the raw reliability (aim error only understates, Thm 4.11(i)); `REJECT` is decision-valid for the declared promise only (Thm 4.11(iv)) — a rejected column may still be a fine column for someone else's aim.

## Screening against a purchase order

Pass the C7 order instead of a declared floor and both the floor and the aim change meaning: `lam_min` becomes the order's guaranteed reliability `lam_A` at the declared promise, and the aim block becomes the order's own state-whitened feasible-dark block `Psi_hat` with its `J` — rebuilt deterministically from the Report the Shadow consumed and applied to the slice rows in the Report's own encoding, so the screen tests the candidate against *the direction the order priced*, not against the residual at large.

```python
po = pv.purchase_order(rep, promise=pv.PurchasePromise(M0=0.01, rho=0.10, delta=0.10))
sc = pv.screen(sh, {"vendor_a": {"values": 0.5 * e + rng.normal(size=sh.m),
                                 "lag_class": "decision-time"}}, po=po)
print(sc)
```

```text
-- Screening (Defs 4.3-4.4; band ends one-sided at 0.05, Lem 4.8 / Thm 4.11(i); split interleaved-day) --
+-------------+--------------+---------------+--------------------------+------------------+----------------------------------+--------+------------+-----------+---------+------------------+
| candidate   |   lam screen |   lam confirm |   lam full (descriptive) | verdict band     |   p_null (exact chi2_J, Cor 4.7) |   DEFF | sign s/c   | verdict   |     gap | m_req eff/rows   |
|-------------+--------------+---------------+--------------------------+------------------+----------------------------------+--------+------------+-----------+---------+------------------|
| vendor_a    |            0 |        0.0226 |                        0 | [0.0000, 0.0331] |                            0.673 |   1.01 | -1/+1      | REJECT    | -0.5634 | 10 / 10          |
+-------------+--------------+---------------+--------------------------+------------------+----------------------------------+--------+------------+-----------+---------+------------------+
counts by verdict: REJECT 1
floor lam_min = 0.5634 [purchase order: guaranteed reliability lam_A at the declared promise (unlock at 1-delta)]; lam* = 0.3509 beside it
aim block: state-whitened feasible-dark block Psi_hat from the purchase order (J = 1)
m_req is the operating requirement of Def 4.4, achieved by the threshold test at t* = lam_min^2 + dlam^2 (Thm 4.9(a)); the plug-in band rule inherits it to first order (Thm 4.9(f)). Under a split the m_eff figure reads on the verdict slice and the rows figure carries the split factor.
+--------+-----------------+----------------+
|   dlam |   m_req (m_eff) |   m_req (rows) |
|--------+-----------------+----------------|
|   0.02 |           3,656 |          3,701 |
|   0.05 |             635 |            643 |
|   0.1  |             180 |            182 |
+--------+-----------------+----------------+
trichotomy (Cor 4.15, Table 3): floor 0.5634 vs lam_arch n/a vs lam_cap n/a -> not available: pass findings= from characterize to place lam_arch and lam_cap on the line
PASS is one-sided valid for the raw reliability: aim error understates only (Thm 4.11(i)). REJECT is decision-valid for the declared promise only (Thm 4.11(iv)).
```

The same vendor column that looked alive against the residual is a clean `REJECT` here: it tracks the shadow, but not the feasible-dark direction the promise actually needs, and the floor a real unlock requires (`lam_A = 0.56`) is far above anything it shows. That distinction — a good column for the residual, the wrong column for the promise — is the whole reason the aim block comes from the order.

## Refusals

Admissibility is Assumption 4.5: every candidate carries a decision-time attestation — its lag class or clock offset relative to the shadow's clock — or is refused by name before any statistic is computed. The floor is mandatory; so is its uniqueness. Frames are screened one member at a time.

```python
try:
    pv.screen(sh, {"bare": {"values": np.zeros(sh.m)}}, floor=(0.12, "declared"))
except ValueError as err:
    print(err)
print()
try:
    pv.screen(sh, {"z": {"values": np.zeros(sh.m), "lag_class": "decision-time"}})
except ValueError as err:
    print(err)
```

```text
screen refused: candidate 'bare' carries no decision-time attestation (its lag class or clock offset relative to the shadow's clock; Ass 4.5)

screen refused: no floor; pass floor=(value, derivation) or a purchase order with a priced columns leg (one of the two is mandatory)
```

A column is taken complete or not at all. A candidate with a `NaN` or an infinity anywhere is refused by name, with the count, before any statistic; the same rule holds in `characterize` (archive and state atoms), `alternate`, `pursue` and `select` (candidate and start columns), and it is the inquiry-layer form of the engine's R-07 refusal on missing values. How a gap is filled (an imputation, a missingness indicator, dropping the rows) is a registration decision made before the verb, and the verb does not make it.

```python
z = np.zeros(sh.m); z[::97] = np.nan
try:
    pv.screen(sh, {"z": {"values": z, "lag_class": "decision-time"}}, floor=(0.12, "declared"))
except ValueError as err:
    print(err)
```

```text
screen refused: candidate 'z' carries 26 missing or non-finite value(s) in 2498 rows; the verbs take complete columns, and how a gap is filled (imputation, a missingness indicator, dropping the rows) is a registration decision made before the verb
```

What `screen` never does: it never names the column (row 9 of the [C9](c9-characterize.md) charter is walled at every rung, and the Screening record carries no field for an identity), and it never turns a PASS into a purchase — completion, if it comes, goes through the acquisition layer with its own book and its own budget.
