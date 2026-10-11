# C8 · The part your model still gets wrong

*Certification told you what the panel can say. The shadow is the record of what it could not: the certified predictor's own residual on rows it never fitted, kept per row with its clock, so the two verbs after this one ([C9](c9-characterize.md), [C10](c10-screen.md)) have something lawful to interrogate. `shadow` is free of acquisition: it issues nothing, charges nothing, and writes nothing to any book. Everything here is on a synthetic panel; nothing is planted beyond what the code shows.*

The paper behind this page calls the per-row quantity the owned residual `u = Y - f_D`: the part of the outcome the certified class does not carry. Its empirical second moment estimates the answerable aggregate `Var(u) = Mhat + Mbar` (Thm 2.4(iii)) — the identified sum of what better fitting could still reach and what no function of the panel ever will. The package prints it as `||e||^2_n`; `e` is the paper's `u`, row by row.

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

fit = slice(0, 5000)
rep = pv.certify(X[fit], y[fit.start:fit.stop], q=0.10, seed=0,
                 clock=(clock[fit], clock[fit] + pd.Timedelta(hours=1)))

sh = pv.shadow(rep, X[5002:].reset_index(drop=True), y[5002:], clock[5002:])
print(sh)
```

```text
-- Shadow (unit panel; estimand E2-realized; free of acquisition, nothing issued or charged) --
rows m = 2498; m_eff = 2498.0 (DEFF 0.857, day-cluster kernel; plug-in, Thm 8.5(iii)); clock step 0 days 01:00:00
era 2026-07-28 10:00:00+00:00 .. 2026-11-09 11:00:00+00:00  [Ass 5.2: an estimation warranty, never an identification clause]
fit manifest consumed: x_sha=86e8dfe5d2e945a1, y_sha=461b5673eb2e7e35, dictionary_sha=35d4a786c216de56, config_sha=1e43e0092612370e, seed=0
mean shift = -0.00223975 [exact on the slice; ranked rules are immune to it, any p > 0.5 rule is not]
shadow second moment ||e||^2_n = 0.0888409 [est of the answerable aggregate Var(u) = Mhat + Mbar, Thm 2.4(iii)]
```

Reading it, top to bottom:

- **rows and `m_eff`** — the slice size and its dependence-adjusted effective size. `m_eff = m / max(DEFF, 1)`: the design effect is measured under the declared cluster kernel (here day clusters, the cluster special case of Thm 8.5(iii)) and printed as measured, but only ever *shrinks* the effective count. Every band downstream reads on `m_eff`, never on raw rows.
- **era** — the slice's clock span, carried under Assumption 5.2: an *estimation warranty* (the archive functionals the later verbs estimate are stable enough over this span to admit a common estimand), never an identification clause. Nothing about the era makes any claim causal.
- **fit manifest consumed** — the Report's own fingerprints, restated so the Shadow is pinned to exactly one certified fit. `f_D` is the Report's certified-class predictor scored through its stored dictionary; the Shadow never refits anything.
- **mean shift** — the exact mean of `e` on the slice. The line states the operational consequence: a *ranked* decision rule built on `f_D` is immune to a pure level shift, while any rule thresholding at a fixed probability is not.
- **`||e||^2_n`** — the headline: the est of the answerable aggregate. It is the total the C9 ledger will decompose, and the denominator every share downstream is a share *of*.

The target must be binary; a continuous slice target is refused by name, because binarizing it is a registration decision the inquiry layer does not make (the same clause the columns leg holds in [C7](c7-purchase.md)). When the Report certified nothing (`khat = 0`), the predictor falls back to the training base rate and the Shadow says so in the read — the shadow of an empty certificate is still a lawful record.

## The refusal: fitted rows are not a slice

Out-of-fit is asserted by row identity, not by promise. Two guards, both refusing by name: the slice's source fingerprint against the Report's `x_sha`, and — when the Report was issued against a book, so its lineage carries clock envelopes — the slice's clock window against the fitted envelope under the same dataset tag, the conservative clock form of the same row identity.

```python
rep = pv.certify(X[fit], y[:5000], q=0.10, seed=0, ledger="./book",
                 clock=(clock[fit], clock[fit] + pd.Timedelta(hours=1)))
try:
    pv.shadow(rep, X[fit], y[:5000], clock[fit])
except ValueError as err:
    print(err)
print()
try:
    pv.shadow(rep, X[100:200].reset_index(drop=True), y[100:200], clock[100:200])
except ValueError as err:
    print(err)
```

```text
shadow refused: overlap by row identity; the slice source fingerprint equals the Report's x_sha, so these are the fitted rows, not an out-of-fit slice

shadow refused: overlap by row identity; the slice clock window [2026-01-05 04:00:00+00:00 .. 2026-01-09 07:00:00+00:00] intersects the Report's fitted lineage envelope [2026-01-01 02:00:00+00:00 .. 2026-07-28 04:00:00+00:00] under the same dataset tag 'panel' (the conservative clock taint of the row identity)
```

The second refusal is deliberately conservative: those hundred rows have a different fingerprint, but their clocks sit inside the fitted envelope, and a Shadow scored on rows the fit could have touched is not a record of out-of-fit error. A ledger-free Report (`ledger=None`, or the manifest-locked refit of C7) carries no envelopes, so only the fingerprint guard binds there — the page you are reading is the notice.

## Frames: several shadows, one clock

`pv.Shadow.join([...])` aligns the shadows of several certified targets on the clock key: aggregation to the coarsest declared clock by the mean, common rows only, every member's manifest carried. Mixed clocks that do not nest into the coarsest are refused by name, and so are members whose eras do not overlap — the frame clause the cross-sectional row of the charter requires (Cor 7.7). What a frame is *for* is the next page: [C9](c9-characterize.md) files one charter per member and answers the frame-wide/group/idiosyncratic question the single-unit charter must leave open. `screen` takes frames one member at a time.
