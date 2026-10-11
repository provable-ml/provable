# C11 · Grow the shadow inside the archive

*[C9](c9-characterize.md) read the shadow's profile one family at a time. `alternate` builds with it: two declared archive classes are fitted to the shadow in alternation, each to the other's partial residual, until the pair has absorbed everything their joint span can hold. It is the build half of the loop, and it stays inside the archive: its fixed point is the joint projection, at most the archive ceiling, and nothing here moves the wall. Free of acquisition: nothing issued, nothing charged, nothing written to any book. Everything is on a synthetic panel; the diurnal term is planted in the code below.*

Fitting one class to the residual of the other and back is backfitting, and in residual form it is alternating projection onto the two classes' orthogonal complements. Two classical facts make it a lawful instrument rather than a heuristic. The iteration converges to the residual of the joint least-squares fit on the union of the two spans (von Neumann). And the excess residual mass, the distance to that fixed point, contracts at every half-step after the first by at least `c^2`, where `c` is the cosine of the Friedrichs angle between the two spans with their shared span removed (Kayalar and Weinert 1988; the angle between the complements equals the angle between the classes, Deutsch 1995). The verb computes `c` from the slice geometry, asserts the bound on every half-step it runs, and reports the trajectory.

```python
import numpy as np
import pandas as pd
import provable as pv

rng = np.random.default_rng(7)
n = 7500
X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
clock = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
hour = clock.hour.to_numpy()
base = (X["f0"] >= 0.3).astype(float) * 0.80 + ((X["f1"] >= 0) & (X["f2"] >= 0)).astype(float) * 0.12
diurnal = 0.08 * np.cos(2 * np.pi * hour / 24)
y = (rng.random(n) < np.clip(base + 0.03 + diurnal, 0, 1)).astype(float)
rep = pv.certify(X[:5000], y[:5000], q=0.10, seed=0,
                 clock=(clock[:5000], clock[:5000] + pd.Timedelta(hours=1)))
sh = pv.shadow(rep, X[5002:].reset_index(drop=True), y[5002:], clock[5002:])

t = np.arange(sh.m)
h = pd.to_datetime(sh.rows["clock"]).dt.hour.to_numpy()
e = sh.rows["e"].to_numpy()
lines = np.column_stack([np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * h / 24),
                         np.cos(2 * np.pi * h / 12), np.sin(2 * np.pi * h / 12)])
lags = np.column_stack([np.concatenate([np.zeros(k), e[:-k]]) for k in (1, 2, 3, 24)])
alt = pv.alternate(sh, {"diurnal": {"values": lines, "lag_class": "decision-time"},
                        "own_lags": {"values": lags, "lag_class": "post-decision"}})
print(alt)
alt.exit_html("exit_chart.html")
```

```text
-- Alternation (build half; free of acquisition, nothing issued or charged) --
classes diurnal (dim 4) and own_lags (dim 4); joint span dim 8; shared span dim 0 [exact, slice geometry]
Friedrichs cosine c = 0.3212 (angle 71.26 deg); contraction bound per half-step after the first: c^2 = 0.1032 [theorem: Kayalar and Weinert 1988; Deutsch 1995 for the complements]
+-------------+----------+-------------+--------------+---------+-------------+--------------------------+
|   half-step | class    |   excess in |   excess out |   ratio |   bound c^2 |   explained share so far |
|-------------+----------+-------------+--------------+---------+-------------+--------------------------|
|           1 | diurnal  | 0.00244189  |  5.54973e-05 |  0.0227 |             |                   0.0291 |
|           2 | own_lags | 5.54973e-05 |  5.15098e-06 |  0.0928 |      0.1032 |                   0.0297 |
|           3 | diurnal  | 5.15098e-06 |  5.27667e-07 |  0.1024 |      0.1032 |                   0.0298 |
|           4 | own_lags | 5.27667e-07 |  5.44049e-08 |  0.1031 |      0.1032 |                   0.0298 |
|           5 | diurnal  | 5.44049e-08 |  5.61295e-09 |  0.1032 |      0.1032 |                   0.0298 |
|           6 | own_lags | 5.61295e-09 |  5.79124e-10 |  0.1032 |      0.1032 |                   0.0298 |
+-------------+----------+-------------+--------------+---------+-------------+--------------------------+
excess = ||r_j - r_inf||^2_n with r_inf the joint projection's residual [exact]; explained share = fitted drop / ||e||^2_n [est of the population share]
shares of ||e||^2_n: diurnal alone 0.0291, own_lags alone 0.0008, joint 0.0298 (the fixed point); remainder 0.9702
converged in 3 rounds (6 half-steps) to excess <= 1e-06 x initial; the ratio bound held at every half-step after the first [asserted on this run]
[no shared span: the split of the fitted value between the two classes is identified]
[alternation is a build algorithm inside the archive: its fixed point is the joint projection, at most the archive ceiling; nothing here moves the wall]
```

Reading it, top to bottom:

- **dims and shared span** - the two classes' ranks after the licensed span (the intercept and the shadow's own `p_hat`) is removed, the rank of their union, and the dimension of their intersection. A nonzero shared span does not stop the alternation; it is reported because on it the *split* of the fitted value between the two classes is not identified (concurvity), only the sum is.
- **`c` and `c^2`** - the Friedrichs cosine and the theorem's contraction bound. `c = 0` is the orthogonal or nested case, where one round finishes the job; `c` close to one is a narrow tunnel, many rounds.
- **the half-step ledger** - excess mass entering and leaving each half-step, their ratio, and the bound the ratio must respect from the second half-step on. On this panel the ratios settle on the bound within three rounds: the bound is sharp, and the trajectory rides it.
- **shares** - each class alone, then the joint fixed point, then the remainder, all as shares of `||e||^2_n`. The joint share is what the pair can build; it is an estimate of the archive-built mass the two classes reach, never a statement about what lies beyond the archive.

## The EXIT chart

`alt.exit_html(path)` writes one self-contained page: the excess mass leaving each half-step against the excess entering it, both as fractions of the initial excess on log axes; the diagonal, where no progress is made; the theorem's bound line `y = c^2 x`; and the trajectory drawn as the usual staircase between the diagonal and the half-steps. The picture is the one iterative decoding borrowed from: two components exchanging what the other could not compute, a tunnel whose width is set by the angle between them, and a trajectory that walks down it. `alt.exit_svg()` returns the chart alone and `alt.exit_points()` the numbers behind it.

Two things the chart cannot show. The tunnel is always open here, because in finite dimension the Friedrichs angle is positive; the chart reports how fast, never whether. And it says nothing about acquisition: the acquire half has no computable transfer curve before the columns are bought, which is the subject of the next page, [C12](c12-pursue.md).
