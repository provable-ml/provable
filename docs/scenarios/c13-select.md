# C13 · Which columns, in which order, and when to stop

*[C12](c12-pursue.md) bought columns the panel did not own. `select` runs the same loop over the columns it does own, as a feature-selection procedure for a logistic class. The admitted columns are the archive; every other declared column is a candidate at a price; the acquire step is run by a candidate's model-free value, the absorb step by class enrichment. It is descriptive and free of acquisition like the other verbs. Everything here is synthetic; the planted structure is in the code.*

Two ledgers per candidate, and the identity between them. The class is a logistic fit on the admitted columns at the current enrichment level (linear, then squares, then pairwise products). The flexible learner, gradient-boosted trees on the same columns, exhibits the archive ceiling on the admitted set; the gap between the class's confirmed log-loss and the learner's is the class shadow, what the logistic form leaves of the admitted information. A candidate's **model-free value** is the learner's confirmed log-loss drop when the column joins the admitted set, an exhibition lower bound on `I(Y; X_j | admitted)`; its **class gain** is the logistic fit's confirmed drop. At every admission the accounting holds exactly: the class's reduction equals the model-free value admitted minus the change in the class shadow, and the absorb step is where the difference is recovered, by raising the enrichment level while the shadow's lower band end stays above zero and the enriched class stays inside its parameter budget: one parameter per `epv` events on the screen slice (the smaller class count; ten events per parameter at the default). Pairwise products grow as the square of the admitted count, so on a wide admitted set the budget is what stops the enrichment, not the shadow; the record then says so, with the level the budget could not carry and the shadow left, which is unabsorbed by budget, not closed.

Honesty. The screen slice fits, the confirm slice scores, on the interleaved-day split. Ranking uses a cross-fitted value inside the screen slice, so the maximum over candidates never touches the confirm slice; the winner is admitted when the larger of its two confirmed values clears its price at the lower band end. The per-candidate statuses in the table use a union-bound band over the candidates of the round and print descriptive. Prices are `none` (the band alone), `aic` (`1/n` per parameter on the mean log-loss scale), `bic` (`log n / 2n`), or a declared mapping; the classical criteria are this loop at particular prices.

## A planted world where stepwise cannot see the first column

Four true columns among twelve: one enters through its square, two linearly, one through a threshold. Forward stepwise on in-sample deviance cannot see the squared one, because its linear coefficient is zero.

```python
import numpy as np
import pandas as pd
import provable as pv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def world(seed, n, N):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, N)), columns=[f"x{i}" for i in range(N)])
    eta = 1.2 * (X["x0"] ** 2 - 1) + 1.0 * X["x1"] - 0.8 * X["x2"] + 1.2 * (X["x3"] > 1.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y, pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")

def stepwise(X, y, penalty):
    n = len(y); cols, remaining = [], list(X.columns)
    def deviance(c):
        p = np.full(n, y.mean()) if not c else LogisticRegression(C=1e3, max_iter=5000).fit(X[c], y).predict_proba(X[c])[:, 1]
        return 2 * n * log_loss(y, p)
    cur = deviance(cols)
    while remaining:
        best = min(remaining, key=lambda c: deviance(cols + [c])); d = deviance(cols + [best])
        if cur - d > penalty: cols.append(best); remaining.remove(best); cur = d
        else: break
    return cols, LogisticRegression(C=1e3, max_iter=5000).fit(X[cols], y)

X, y, clock = world(0, 6000, 12)
fit, test = slice(0, 4000), slice(4000, 6000)
sel = pv.select(X.iloc[fit], y[fit], clock[fit], price="bic")
print(sel)
cols, m = stepwise(X.iloc[fit], y[fit], penalty=np.log(4000))
print(f"loop      {sel.admitted} enrichment {sel.level_enrich}: held-out log-loss {log_loss(y[test], sel.predict_proba(X.iloc[test])):.4f}")
print(f"stepwise  {cols} (forward, BIC, in-sample): held-out log-loss {log_loss(y[test], m.predict_proba(X.iloc[test][cols])[:, 1]):.4f}")
```

```text
-- Selection (the loop on owned columns; logistic class; split interleaved-day; bands one-sided at 0.05; price rule bic) --
base-rate log-loss on the confirm slice 0.69313; exhibited ceiling on all candidates 0.51263 (flexible learner, confirm slice) [est]
+---------+-----------+------------+------------------------+------------------------+---------+---------------+--------------+----------------------+----------+
|   round | verdict   | admitted   | model-free value       | class gain             |   price |   loss before |   loss after |   class shadow after |   enrich |
|---------+-----------+------------+------------------------+------------------------+---------+---------------+--------------+----------------------+----------|
|       0 | acquire   | x0         | +0.06413 (lo +0.04901) | -0.00005 (lo -0.00054) | 0.00104 |       0.69313 |      0.61461 |             -0.01439 |        1 |
|       1 | acquire   | x1         | +0.05741 (lo +0.04333) | +0.05498 (lo +0.04199) | 0.00207 |       0.61461 |      0.55964 |             -0.01195 |        1 |
|       2 | acquire   | x2         | +0.04362 (lo +0.02939) | +0.04485 (lo +0.03194) | 0.00207 |       0.55964 |      0.51479 |             -0.01318 |        1 |
|       3 | acquire   | x3         | +0.01384 (lo +0.00602) | +0.01294 (lo +0.00758) | 0.00207 |       0.51479 |      0.50185 |             -0.01227 |        1 |
|       4 | stall     |            |                        |                        |         |       0.50185 |              |                      |          |
+---------+-----------+------------+------------------------+------------------------+---------+---------------+--------------+----------------------+----------+
candidates at the last round (4); verdict bands at level 0.05 over the candidates on the confirm slice; ranking by the cross-fitted screen-slice value:
+-------------+-------------+--------------------+-------------+--------------+-------------+---------+--------------------------------------------+
| candidate   |   rank stat |   model-free value |   lower end |   class gain |   lower end |   price | status                                     |
|-------------+-------------+--------------------+-------------+--------------+-------------+---------+--------------------------------------------|
| x4          |    -0.00099 |           -0.00335 |    -0.01057 |      0.00094 |    -9e-05   | 0.00207 | no exhibited value beyond the admitted set |
| x5          |     0.00206 |           -0.00242 |    -0.01106 |      0.00043 |    -0.00067 | 0.00207 | no exhibited value beyond the admitted set |
| x6          |    -0.0024  |            0.00368 |    -0.00355 |      0.0022  |    -0.0003  | 0.00207 | no exhibited value beyond the admitted set |
| x7          |    -0.00027 |           -0.00365 |    -0.01138 |     -0.00014 |    -0.00052 | 0.00207 | no exhibited value beyond the admitted set |
| x8          |    -0.00349 |           -0.00177 |    -0.00856 |     -0.00034 |    -0.0016  | 0.00207 | no exhibited value beyond the admitted set |
| x9          |    -0.00077 |           -0.0029  |    -0.01105 |     -0.00021 |    -0.00334 | 0.00207 | no exhibited value beyond the admitted set |
| x10         |    -0.00079 |           -0.00355 |    -0.01062 |      4e-05   |    -0.00186 | 0.00207 | no exhibited value beyond the admitted set |
| x11         |    -0.00043 |            0.0011  |    -0.00668 |     -0.00093 |    -0.0024  | 0.00207 | no exhibited value beyond the admitted set |
+-------------+-------------+--------------------+-------------+--------------+-------------+---------+--------------------------------------------+
admitted ['x0', 'x1', 'x2', 'x3'] at enrichment level 1; final confirm log-loss 0.50185 (reduction +0.19128 from the base rate)
class-fit parameter budget 100 (997 events on the screen slice at 10 events per parameter); parameters in the final class 8
outcome: stall: the top-ranked candidate does not clear its price at the lower band end on the confirm slice and no candidate is real at the union-bound level; whether the residual is noise or an unadmitted column is walled
[design: ranking by the cross-fitted value inside the screen slice, admission by the winner's confirmed value on the confirm slice at the lower band end; per-candidate statuses use a union-bound band over the candidates of the round; when the winner fails at its own band the highest-ranked candidate real at the union-bound level is admitted instead, so a stall means no candidate is real at level/K]
[theorem: the confirmed reduction of the class equals the model-free value admitted minus the change in the class shadow; the model-free value is an exhibition lower bound on the conditional mutual information, never its value]
[design: the enrichment menu is squares then pairwise products; the learner is gradient-boosted trees at a fixed modest setting; both are declared, not derived]
[design: the enriched class is held to one parameter per 10 events on the screen slice (997 events, 100 parameters); a level the budget cannot carry is not fitted, and the shadow it would have addressed is reported as unabsorbed by budget]
loop      ['x0', 'x1', 'x2', 'x3'] enrichment 1: held-out log-loss 0.4755
stepwise  ['x1', 'x2', 'x3', 'x4', 'x6'] (forward, BIC, in-sample): held-out log-loss 0.5977
```

Reading it:

- **round 0** - `x0` is admitted on its model-free value alone; its class gain is zero, the status reads *real, unabsorbed by the class*, and the absorb step raises the enrichment to level 1 (squares), where the class takes it: the loss falls from the base rate to 0.60 in one round.
- **rounds 1 to 3** - the linear and threshold columns follow, each with both ledgers positive.
- **round 4** - the top-ranked remaining candidate does not clear its price on the confirm slice; every noise column reads *no exhibited value beyond the admitted set*, and the verb stalls without guessing whether the residual is noise or an unadmitted column.
- **the baseline** - forward stepwise BIC admits the three visible columns and two noise columns and never sees `x0`; on held-out rows the loop's log-loss is 0.48 against 0.60.

Over five seeds of this world (4,000 fitting rows, 2,000 held out): the loop admitted no noise column in any seed (precision 1.00, recall 0.90), held-out log-loss 0.480 against 0.594 for stepwise AIC and 0.593 for stepwise BIC, accuracy 0.766 against 0.677 and 0.681.

## Five relevant columns among forty

Same construction with forty candidates and five true columns (one squared, three linear, one threshold), five seeds. The loop recovered exactly the five in four seeds and four of the five in the fifth, with no noise column admitted in any seed (precision 1.00, recall 0.96); stepwise AIC admitted between two and eight noise columns per seed (precision 0.45); stepwise BIC admitted no noise column and never the squared one (recall 0.80). Held-out log-loss 0.483 for the loop against 0.580 and 0.576; accuracy 0.763 against 0.692 and 0.695. Each loop run took about fifty seconds on two cores.

## What the verb does not claim

The model-free value is an exhibition, never the conditional mutual information itself; a column the learner cannot use on these rows reads as valueless, and a pair of columns that matter only jointly reads as two valueless columns, since the loop is greedy and admits one at a time. The learner and the enrichment menu are declared choices. A stall does not say the remaining columns are noise; it says none clears its price under this learner and this class, which is the same wall the paper draws.
