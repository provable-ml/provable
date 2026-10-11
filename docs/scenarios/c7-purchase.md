# C7 · The cure is a column you do not have

*You certified something, but the model still sees more than your columns can name. The rows leg ([the purchase order](../glossary.md#atoms)) prices more of the same data; the columns leg prices the feature the panel lacks: how reliable a new column would have to be, how it converts to rows, and where the wall becomes terminal. Everything here is on a synthetic panel; nothing is planted beyond what the code shows.*

`purchase_order(rep)` with no promise prices the rows leg only (see [C1](c1-certify.md)/[B4](b4-darkmass.md) for what "certified" and "dark mass" mean). Pass a **promise** and the columns leg activates: a target dark mass `M0` you would like to make certifiable, a localization tolerance `rho`, and an unlock confidence `delta`.

```python
import numpy as np
import pandas as pd
import provable as pv

rng = np.random.default_rng(7)
n = 6000
X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
base = (X["f0"] >= 0.3).astype(float) * 0.80 + ((X["f1"] >= 0) & (X["f2"] >= 0)).astype(float) * 0.12
y = (rng.random(n) < np.clip(base + 0.03, 0, 1)).astype(float)

rep = pv.certify(X, y, q=0.10, seed=0)
po = pv.purchase_order(rep, promise=pv.PurchasePromise(M0=0.05, rho=0.10, delta=0.10))
print(po.columns)
```

```text
-- purchase order (columns leg; est, plug-in; promise M0=0.05, rho=0.1, delta=0.1, estimand=E2-realized) --
+-----------------+-----+---------+------------+------------+--------+----------+
| feasible-dark   |   J | share   |   captured |   gammahat |    eta | DK sin   |
|-----------------+-----+---------+------------+------------+--------+----------|
| tolerance       |   1 | >=0.9   |     0.3268 |     0.0336 | 0.1204 |          |
+-----------------+-----+---------+------------+------------+--------+----------+
debiased visibility: nu_deb=3.07102 (+/-0.9092); ||Deps||^2_deb=3.00609 (+/-0.1814)
reachability ceiling pihat(1-pihat)nu_deb = 0.709411; M0 = 0.05
+--------------------------------------------------------------+----------------+---------------+------------+
| reliability price                                            | value          | feasibility   | lam        |
|--------------------------------------------------------------+----------------+---------------+------------|
| guaranteed reliability lam_A (unlock at 1-delta)             | lam^2=0.0529   | feasible      | lam=0.23   |
| reliability-law lam* (fixed hbar, M* = reachability ceiling) | lam*^2=0.01933 |               | lam*=0.139 |
+--------------------------------------------------------------+----------------+---------------+------------+
column count r_min: not priced (declare kappa and B in the promise to price it)
rows-route (columns<->rows exchange): 23 rows would carry the same feasible visibility at rho=0.1, delta=0.1
never-ceiling (plug-in from nu_deb): AUC<=0.8924, accuracy<=0.8215 (in-panel estimate)
```

Reading it, top to bottom:

- **feasible-dark** is where the reachable dark mass lives: a `J`-dimensional subspace of the estimated Gram (here `J=1`, a tolerance set capturing the required share of the debiased contrast energy). It is the target a new column has to align with.
- **debiased visibility** `nu_deb` is `po.state.nu`, the estimated-Gram corrected visibility of the panel. The `+/-` band is the plug-in interval; the correction subtracts the `P`-dimensional estimation bias and uses the correlated quadratic-form variance, so `nu_deb` is not the raw `Dhat' Ghat^-1 Dhat`.
- **reachability ceiling** is `pihat(1-pihat)*nu_deb`: the most mass the augmented panel could ever make legible. Every price below is quoted only because `M0` sits under it.
- **reliability price** is the headline. `lam_A` is the reliability a new column must carry to unlock `M0` at confidence `1-delta` (guaranteed form); `lam*` is the reliability-law value at fixed evidence, taken against the same reachability ceiling the guard uses, so it reaches zero exactly where the guard fires. A column measuring the feasible-dark direction with squared reliability at least `lam_A^2` clears the promise; below it, no count of such columns does.
- **rows-route** is the columns/rows exchange: how many additional rows would carry the same feasible visibility, so you can compare buying a column against buying data.
- **never-ceiling** is the accuracy and AUC a perfect reader of this panel plus the feasible-dark direction could reach, plug-in from `nu_deb`.

`po.state.nu` exposes the same corrected visibility directly:

```python
print(round(po.state.nu, 4))   # 3.071
```

## The guard: when the promise is past the wall

Ask for more mass than the panel could ever surface and the order refuses to price a fantasy:

```python
po = pv.purchase_order(rep, promise=pv.PurchasePromise(M0=2.0, rho=0.10))
print(po.columns)
```

```text
reachability ceiling pihat(1-pihat)nu_deb = 0.709411; M0 = 2
TERMINAL-REMAINDER GUARD FIRED (PG Thm 7.4): M0 exceeds the ceiling by 1.29059; that excess is terminally dark, no admissible column resolves it. Acquisition NOT priced.
```

The terminal-remainder guard is the columns-leg counterpart of a refusal: the excess of `M0` over the reachability ceiling is mass that stays dark under any admissible acquisition, so the order quotes the gap and prices nothing. This is what stops the columns leg from selling a reliability that would never certify.

## What the columns leg is, and is not

The columns leg prices **statistical properties** a column would need (its reliability toward the feasible-dark direction, and how that converts to rows), never a real-world identity. It is binary-`Y` by construction; a continuous target is refused before pricing, because turning it into a label is a registration decision, not one the purchase layer makes. And like the rest of the layer, it issues no certificate and charges no budget: it is a quote you read before deciding, not a purchase.
