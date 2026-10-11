# C12 · Buy the column, absorb it, repeat

*[C10](c10-screen.md) tested one vendor's claim. `pursue` runs the whole acquire half of the loop: several priced candidate columns against a declared requirement, round after round, each round placed by the trichotomy before any column is credited, each acquisition absorbed before the next candidate is read. It is forward stagewise pursuit with prices, and it is explicit about which of its parts are theorem and which are design. Free of acquisition in the same sense as the other verbs: the candidates are declared inputs, the verb prices and orders them, it buys nothing and issues nothing. Everything is on a synthetic panel; the vendors are planted in the code below.*

The accounting is theorem. An acquisition moves mass across the wall by exactly the increment of the archive mean, the increments telescope along any chain of archives, each one depends on the order in which the columns arrive and their sum does not (Thm 9.4(i) iterated; the L2 chain rule). Their sum is bounded by the external mass of the starting archive, a walled quantity that never prints. The rule is design: at each round the remaining requirement is placed by the trichotomy (build when the exhibited archive-built share from `findings` covers it, impossible when it exceeds the residual mass, acquire between), every remaining candidate is residualized against the licensed span and the columns already held so that only what the archive cannot compute is credited, its coefficients are fitted on the screen slice and its drop confirmed on the disjoint confirm slice with a clustered plug-in band, and the candidate with the largest confirmed drop per unit price among those whose lower band end clears their price is absorbed. Prices are declared on the same scale as the drops, a share of `||e||^2_n`; the exchange rate is the caller's.

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
e = sh.rows["e"].to_numpy()
ch = pv.characterize(sh, atoms={"archive": {"book_flow": {
    "values": 0.3 * e + rng.normal(size=sh.m), "lag_class": "post-decision"}}})

candidates = {
    "vendor_a": {"values": 3.0 * e + rng.normal(size=sh.m), "lag_class": "decision-time", "price": 0.05},
    "vendor_b": {"values": rng.normal(size=sh.m), "lag_class": "decision-time", "price": 0.0},
    "vendor_c": {"values": 1.5 * e + rng.normal(size=sh.m), "lag_class": "decision-time", "price": 0.01},
}
pu = pv.pursue(sh, candidates, requirement=(0.30, "the study's declared floor, as a share of the residual mass"), findings=ch)
print(pu)
```

```text
-- Pursuit (acquire half; forward stagewise, priced; split interleaved-day; bands one-sided at 0.05) --
requirement gamma = 0.3000 of ||e||^2_n [DECLARED]; derivation: the study's declared floor, as a share of the residual mass; archive-built share lam_arch^2 = 0.0074 from findings
+---------+---------------------------------------------------------------------------------------+------------+---------+------------------+------------------+-----------------------+-------------------+------------------+------------------+
|   round | verdict                                                                               | acquired   |   price |   drop (confirm) |   drop lower end |   drop (full, descr.) |   residual before |   residual after |   remaining req. |
|---------+---------------------------------------------------------------------------------------+------------+---------+------------------+------------------+-----------------------+-------------------+------------------+------------------|
|       0 | acquire (walled band: above the archive-built share, below the residual mass; Prop 2) | vendor_c   |    0.01 |           0.1649 |           0.1101 |                0.1701 |            1      |           0.8298 |           0.1351 |
|       1 | acquire (walled band: above the archive-built share, below the residual mass; Prop 2) | vendor_a   |    0.05 |           0.332  |           0.2663 |                0.3174 |            0.8298 |           0.5124 |           0      |
|       2 | met                                                                                   |            |         |                  |                  |                       |            0.5124 |                  |           0      |
+---------+---------------------------------------------------------------------------------------+------------+---------+------------------+------------------+-----------------------+-------------------+------------------+------------------+
+---------+-------------+------------------+-------------+---------+------------------+--------------------------------------------------+
|   round | candidate   |   drop (confirm) |   lower end |   price | drop per price   | status                                           |
|---------+-------------+------------------+-------------+---------+------------------+--------------------------------------------------|
|       0 | vendor_a    |           0.443  |      0.3458 |    0.05 | 8.86             | clears its price                                 |
|       0 | vendor_b    |          -0.0007 |     -0.0034 |    0    | free             | does not clear its price at the band's lower end |
|       0 | vendor_c    |           0.1649 |      0.1101 |    0.01 | 16.5             | clears its price                                 |
|       1 | vendor_a    |           0.332  |      0.2663 |    0.05 | 6.64             | clears its price                                 |
|       1 | vendor_b    |           0.0004 |     -0.001  |    0    | free             | does not clear its price at the band's lower end |
+---------+-------------+------------------+-------------+---------+------------------+--------------------------------------------------+
outcome: met: the requirement is covered by the confirmed drops of the acquisitions
sum of confirmed drops = 0.4970 of ||e||^2_n at total price 0.0600; full-slice descriptive total = 0.4876, which telescopes exactly and does not depend on the order [theorem]; remaining residual share 0.5124
the theorem bound on the sum of increments, M_ext of the starting archive, is walled (Thm 2.11) and does not print; the printable envelope is the residual share above
[design: greedy selection by confirmed drop per unit price; each increment depends on the order, the full-slice total does not (the L2 chain rule); a different order can reach the requirement at a different price]
[honesty: coefficients fitted on the screen slice, drops confirmed on the disjoint confirm slice with a clustered plug-in band; the full-slice drop prints descriptive only]
[the acquisitions are declared inputs; this verb prices and orders them, it buys nothing and issues nothing]
```

Reading it:

- **round 0** - the requirement sits above the archive-built share and below the residual mass, so the verdict is acquire. Two vendors clear their price at the lower band end; the free noise column does not, and free is not enough: a candidate must confirm a positive drop on rows its coefficients never saw. `vendor_c` is cheaper per unit of drop and goes first.
- **round 1** - `vendor_a` is read again, now residualized against `vendor_c`, and its confirmed drop is smaller than it was in round 0. That is the order dependence of the increments, visible in one table. It still clears its price and is absorbed.
- **round 2** - the confirmed drops cover the requirement: `met`. The verb stops; the remaining residual is not an invitation.
- **the two totals** - the confirmed sum is what the honesty split certifies; the full-slice total is descriptive, telescopes exactly, and is the same in every order (the lock suite checks both orders against the joint projection).

## Stopping, and what a stall means

Five outcomes close a pursuit: `met`, `build` (the archive covers what is left, no purchase), `impossible` (the remainder exceeds the residual mass; no column of any law reaches it), `exhausted` (no candidate left), and `stall`. A stall is the one to read carefully: every remaining candidate failed to clear its price. Whether that is because the residual is noise or because the right column was not on the table is exactly the split Theorem 2.11 walls; the verb says so rather than guessing. What it can say is the envelope: the residual share left, which bounds every further drop.

The greedy order is not optimal over sequences; a different order can reach the same requirement at a different price, and the verb prints the caveat every time. The loop it belongs to is measure ([C9](c9-characterize.md)), absorb ([C11](c11-alternate.md)), acquire (this page), absorb again.
