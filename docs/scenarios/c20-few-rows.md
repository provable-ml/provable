# C20 · When looking gains: too few training rows to choose

*Synthetic world throughout, built by the first block below. The code blocks, run in order, are the demo script, also shipped as [c20_few_rows_demo.py](c20_few_rows_demo.py); its last block repeats the study over 120 worlds and takes a few minutes. The clock and the custody salt were fixed when this page was generated, so the holdout ids and the model hashes are specific to that run; every other number is reproduced by the code shown.*

Nothing changes in this world. The holdout gains because there is more of it: 1,000 rows against 400 training rows, and 30 candidates that differ by small amounts. Cross-validation on 400 rows cannot tell the close candidates apart; 800 holdout rows can. The look buys a better model and pays with a wider certificate.

## In short

| | |
|---|---|
| data | 1,400 rows; 100 features, independent standard normal |
| target | continuous, so regression: y = 0.25 (x0 + x1 + x2 + x3 + x4) + noise, with standard normal noise; the other 95 features carry nothing |
| split | the first 400 rows train, the next 1,000 are the holdout; every row is drawn the same way, so the split is random in law |
| candidates | 30 least-squares fits on the training rows: the k features most correlated with y on those rows, k = 1 to 30 |
| live score | mean squared error, exact: 1 + b0^2 + sum over j of (b_j - beta_j)^2 for a fit with intercept b0 and coefficients b_j; the best possible is 1 |
| without looking | 5-fold cross-validation on the training rows, the ranking redone inside each fold, picks k = 11; live MSE 1.137; the holdout certifies at most 1.306 on all 1,000 rows |
| looking | 30 yes or no answers, 19.80 bits, pick k = 4, the best of the 30 here; live MSE 1.069; the exam slice certifies at most 1.366 on 200 rows |
| over 60 worlds | live MSE 1.0449 without looking, 1.0223 looking, 1.0212 for the best candidate; bound 1.1238 against 1.1821 |
| with 4,000 training rows, 60 worlds | live MSE 1.0024 without looking and 1.0021 looking; bound 1.0667 against 1.1578 |

The model gets better and its certificate gets worse. Both hold at once; which one matters is the analyst's call.

## The study

### 1. The world and the candidates

`world(rep, m)` draws world `rep` with `m` training rows and 1,000 holdout rows. `candidates` ranks the features by their correlation with y on the rows it is given and fits least squares on the top k, for k = 1 to 30. The features are independent standard normal, so a fit's live mean squared error is exact, `live_mse`, and no extra rows are needed to score it.

```python
import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
import provable as pv
from provable import certificates as ct

os.chdir(tempfile.mkdtemp(prefix="few_rows_"))
P, K = 100, 30
beta = np.r_[np.full(5, 0.25), np.zeros(P - 5)]
names = [f"x{j}" for j in range(P)]


def world(rep, m):
    rng = np.random.default_rng(3000 + rep)
    Xs = rng.normal(size=(m, P))
    ys = Xs @ beta + rng.normal(size=m)
    Xh = rng.normal(size=(1000, P))
    yh = Xh @ beta + rng.normal(size=1000)
    panel = pd.DataFrame(np.vstack([Xs, Xh]), columns=names).assign(y=np.r_[ys, yh])
    panel["start_utc"] = pd.date_range("2026-01-01", periods=len(panel), freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    train, hold = panel.iloc[:m], panel.iloc[m:]
    S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=f.index)) for f in (train, hold))
    return rng, train, hold, S, H


def candidates(X, y):
    order = np.argsort(-np.abs((X - X.mean(0)).T @ (y - y.mean())))
    fits = []
    for k in range(1, K + 1):
        keep = order[:k]
        fits.append((keep, np.linalg.lstsq(np.column_stack([np.ones(len(y)), X[:, keep]]), y, rcond=None)[0]))
    return fits


def cv_mse(rng, X, y):
    sq = np.zeros(K)
    for fold in np.array_split(rng.permutation(len(y)), 5):
        rest = np.setdiff1d(np.arange(len(y)), fold)
        for k, (keep, b) in enumerate(candidates(X[rest], y[rest])):
            sq[k] += ((y[fold] - b[0] - X[fold][:, keep] @ b[1:]) ** 2).sum()
    return sq / len(y)


def live_mse(keep, b):
    full = np.zeros(P)
    full[keep] = b[1:]
    return 1.0 + b[0] ** 2 + float(((full - beta) ** 2).sum())


def predictor(keep, b):
    cols = [names[j] for j in keep]
    return lambda Z: b[0] + Z[cols].to_numpy() @ b[1:]


rng, train, hold, S, H = world(0, m=400)
Xs, ys = train[names].to_numpy(), train["y"].to_numpy()
fits = candidates(Xs, ys)
live = [live_mse(keep, b) for keep, b in fits]
base = float(ys.mean())
```

### 2. Without looking

```python
cv = cv_mse(rng, Xs, ys)
k_cv = int(np.argmin(cv)) + 1
ho_a = pv.seal(hold[names], hold["y"], "book_no_look", lineage=H, baseline=base, baseline_lineage=S,
               exam_share=0.0, label="no look")
cert_a = ho_a.open(predictor(*fits[k_cv - 1]), lineage=S, label=f"k={k_cv}")
print(f"cross-validation picks k = {k_cv}")
print(cert_a.claim)
```

```text
cross-validation picks k = 11
performance: mean squared loss of model 306ebf6d8188 at most 1.30563 (FWER 0.05, CLT approximation; HO-E053BCA7A7CAF546, tuning slice, 0.00 bits spent)
```

Cross-validation picks k = 11. The holdout, sealed without an exam slice and opened at once at the clean bar, certifies a mean squared error of at most 1.306 on all 1,000 rows, for 0 bits.

### 3. Looking

```python
ho_b = pv.seal(hold[names], hold["y"], "book_look", lineage=H, baseline=base, baseline_lineage=S, label="look")
answers = [ho_b.ask(predictor(*fits[k - 1]), lineage=S, label=f"k={k}") for k in range(1, K + 1)]
k_look = int(next(r["label"] for r in ho_b.transcript().rows if r["seq"] == ho_b.state["incumbent"])[2:])
cert_b = ho_b.open(predictor(*fits[k_look - 1]), lineage=S, label=f"k={k_look}")
print(f"{len(answers)} answers, {sum(a.accepted for a in answers)} accepts, {ho_b.state['bits']:.2f} bits; the incumbent is k = {k_look}")
print(cert_b.claim)
```

```text
30 answers, 4 accepts, 19.80 bits; the incumbent is k = 4
performance: mean squared loss of model 5c53e34f7bfc at most 1.36638 (FWER 0.05, CLT approximation; HO-F64538D6D4F41B54, exam slice, 19.80 bits spent)
```

The 30 candidates are asked in order of k. Four are accepted, and the incumbent is k = 4. The exam slice certifies it at most 1.366, on 200 rows.

### 4. What each analyst saw

```python
signal = lambda j: "signal" if j < 5 else "noise"
print(tabulate([[k, f"x{fits[k - 1][0][-1]}", signal(fits[k - 1][0][-1]), cv[k - 1], "ACCEPT" if answers[k - 1].accepted else "reject",
                 answers[k - 1].bits, live[k - 1]] for k in range(1, 16)],
               headers=["k", "feature added", "", "CV MSE (training rows)", "holdout answer", "bits", "live MSE"],
               tablefmt="psql", floatfmt=".4f"))
```

```text
+-----+-----------------+--------+--------------------------+------------------+--------+------------+
|   k | feature added   |        |   CV MSE (training rows) | holdout answer   |   bits |   live MSE |
|-----+-----------------+--------+--------------------------+------------------+--------+------------|
|   1 | x0              | signal |                   1.2814 | ACCEPT           | 1.0000 |     1.2662 |
|   2 | x1              | signal |                   1.2551 | ACCEPT           | 0.4150 |     1.1953 |
|   3 | x2              | signal |                   1.2091 | ACCEPT           | 0.2630 |     1.1339 |
|   4 | x4              | signal |                   1.1561 | ACCEPT           | 0.1926 |     1.0692 |
|   5 | x96             | noise  |                   1.1800 | reject           | 3.3219 |     1.0980 |
|   6 | x65             | noise  |                   1.1796 | reject           | 2.0000 |     1.1347 |
|   7 | x63             | noise  |                   1.1674 | reject           | 1.4854 |     1.1551 |
|   8 | x3              | signal |                   1.2082 | reject           | 1.1926 |     1.1056 |
|   9 | x14             | noise  |                   1.1946 | reject           | 1.0000 |     1.1149 |
|  10 | x94             | noise  |                   1.1911 | reject           | 0.8625 |     1.1234 |
|  11 | x31             | noise  |                   1.1557 | reject           | 0.7590 |     1.1366 |
|  12 | x46             | noise  |                   1.1688 | reject           | 0.6781 |     1.1388 |
|  13 | x49             | noise  |                   1.1863 | reject           | 0.6130 |     1.1475 |
|  14 | x67             | noise  |                   1.1892 | reject           | 0.5594 |     1.1541 |
|  15 | x95             | noise  |                   1.2176 | reject           | 0.5146 |     1.1574 |
+-----+-----------------+--------+--------------------------+------------------+--------+------------+
```

The first fifteen candidates. The CV column is what the analyst who does not look sees: beyond k = 4 the estimates stay within a few hundredths of each other and move up and down, and k = 11 edges out k = 4 by 0.0004. The answer column is what the analyst who looks sees: the first four candidates, each adding a signal feature, are accepted; every later one carries at least one noise feature and is rejected, including k = 8, which adds x3 after three noise features. The bits column prices each answer: accepts after accepts are cheap, and the first reject after them costs 3.32 bits. The last column is what neither sees.

```python
k_best = int(np.argmin(live)) + 1
print(tabulate([["no look", k_cv, 0.0, cert_a.artifact["bound"], live[k_cv - 1]],
                ["look", k_look, ho_b.state["bits"], cert_b.artifact["bound"], live[k_look - 1]],
                ["best of the 30", k_best, None, None, live[k_best - 1]]],
               headers=["route", "k", "bits", "95% bound", "live MSE"], tablefmt="psql", floatfmt=".4f"))
```

```text
+----------------+-----+---------+-------------+------------+
| route          |   k |    bits |   95% bound |   live MSE |
|----------------+-----+---------+-------------+------------|
| no look        |  11 |  0.0000 |      1.3056 |     1.1366 |
| look           |   4 | 19.7958 |      1.3664 |     1.0692 |
| best of the 30 |   4 |         |             |     1.0692 |
+----------------+-----+---------+-------------+------------+
```

Looking cut the live mean squared error from 1.137 to 1.069, the best of the 30, and raised the bound from 1.306 to 1.366, because the certificate now rests on 200 rows instead of 1,000.

### 5. With 4,000 training rows

```python
def study(rep, m):
    """Both routes on world rep with m training rows: {route: (k, bits, bound, live MSE)}."""
    rng, train, hold, S, H = world(rep, m)
    X, y = train[names].to_numpy(), train["y"].to_numpy()
    fits = candidates(X, y)
    k_cv = int(np.argmin(cv_mse(rng, X, y))) + 1
    books = tempfile.mkdtemp(dir=".")
    no_look = pv.seal(hold[names], hold["y"], os.path.join(books, "a"), lineage=H, baseline=float(y.mean()),
                      baseline_lineage=S, exam_share=0.0, seed=rep)
    look = pv.seal(hold[names], hold["y"], os.path.join(books, "b"), lineage=H, baseline=float(y.mean()),
                   baseline_lineage=S, seed=rep)
    for k in range(1, K + 1):
        look.ask(predictor(*fits[k - 1]), lineage=S, label=f"k={k}")
    k_look = int(next(r["label"] for r in look.transcript().rows if r["seq"] == look.state["incumbent"])[2:])
    live = [live_mse(keep, b) for keep, b in fits]
    out = {}
    for route, ho, k in (("no look", no_look, k_cv), ("look", look, k_look)):
        cert = ho.open(predictor(*fits[k - 1]), lineage=S, label=f"k={k}")
        out[route] = (k, ho.state["bits"], cert.artifact["bound"], live[k - 1])
    out["best of the 30"] = (int(np.argmin(live)) + 1, None, None, min(live))
    return out


one = study(0, 4000)
print(tabulate([[route, *v] for route, v in one.items()], headers=["route", "k", "bits", "95% bound", "live MSE"],
               tablefmt="psql", floatfmt=".4f"))
```

```text
+----------------+-----+---------+-------------+------------+
| route          |   k |    bits |   95% bound |   live MSE |
|----------------+-----+---------+-------------+------------|
| no look        |   5 |  0.0000 |      1.0582 |     1.0006 |
| look           |   5 | 22.2983 |      1.1017 |     1.0006 |
| best of the 30 |   5 |         |             |     1.0006 |
+----------------+-----+---------+-------------+------------+
```

`study(rep, m)` runs both routes on world `rep` with `m` training rows. With 4,000, cross-validation and the holdout agree on world 0: both pick k = 5, and looking costs 22.30 bits and widens the bound from 1.058 to 1.102 for nothing.

### 6. Sixty worlds of each

```python
table = []
for m in (400, 4000):
    runs = [study(rep, m) for rep in range(60)]
    for route in ("no look", "look", "best of the 30"):
        r = [run[route] for run in runs]
        priced = r[0][2] is not None
        table.append([m, route, np.mean([x[3] for x in r]), np.mean([x[2] for x in r]) if priced else None,
                      f"{sum(x[2] >= x[3] for x in r)} of 60" if priced else "", np.mean([x[1] for x in r]) if priced else None,
                      np.mean([x[0] for x in r])])
    gain = [run["no look"][3] - run["look"][3] for run in runs]
    print(f"{m:,} training rows: the look chose a better model in {sum(g > 1e-12 for g in gain)} worlds, a worse one in {sum(g < -1e-12 for g in gain)}, the same in {sum(abs(g) <= 1e-12 for g in gain)}")
    print(f"{m:,} training rows: the best of the 30 chosen in", {route: sum(run[route][0] == run["best of the 30"][0] for run in runs) for route in ("no look", "look")}, "worlds of 60")
print(tabulate(table, headers=["training rows", "route", "mean live MSE", "mean 95% bound", "bound held", "mean bits", "mean k"],
               tablefmt="psql", floatfmt=".4f"))
```

```text
400 training rows: the look chose a better model in 38 worlds, a worse one in 2, the same in 20
400 training rows: the best of the 30 chosen in {'no look': 19, 'look': 56} worlds of 60
4,000 training rows: the look chose a better model in 12 worlds, a worse one in 10, the same in 38
4,000 training rows: the best of the 30 chosen in {'no look': 47, 'look': 49} worlds of 60
+-----------------+----------------+-----------------+------------------+--------------+-------------+----------+
|   training rows | route          |   mean live MSE |   mean 95% bound | bound held   |   mean bits |   mean k |
|-----------------+----------------+-----------------+------------------+--------------+-------------+----------|
|             400 | no look        |          1.0449 |           1.1238 | 57 of 60     |      0.0000 |   6.0833 |
|             400 | look           |          1.0223 |           1.1821 | 55 of 60     |     21.8209 |   5.0667 |
|             400 | best of the 30 |          1.0212 |                  |              |             |   5.1000 |
|            4000 | no look        |          1.0024 |           1.0667 | 54 of 60     |      0.0000 |   5.4500 |
|            4000 | look           |          1.0021 |           1.1578 | 54 of 60     |     22.5266 |   5.4333 |
|            4000 | best of the 30 |          1.0015 |                  |              |             |   5.0000 |
+-----------------+----------------+-----------------+------------------+--------------+-------------+----------+
```

With 400 training rows, looking chose a better model in 38 worlds of 60 and a worse one in 2, and the best of the 30 in 56 worlds against 19 for cross-validation. It closed 95 percent of the gap between cross-validation and the best candidate (1.0449 to 1.0223, against 1.0212), and the mean bound rose from 1.1238 to 1.1821. With 4,000 training rows the two routes are a toss-up, better in 12 worlds and worse in 10, the best of the 30 in 49 worlds against 47, and the mean live score moved by 0.0003 while the bound rose from 1.0667 to 1.1578. The bound held in 54 to 57 worlds of 60: the bar is the Gaussian form, a central-limit approximation, and on squared errors it holds a little less often than its nominal 95 percent, as [C15](c15-transcript.md) notes.

The gain here is the holdout's rows used as validation rows, which is what looking is. It exists when the analyst's own rows cannot rank the candidates, and it costs certificate width whether it exists or not. [C19](c19-changed-world.md) is the other way a look gains: there the holdout knows a law the training rows never saw, and the bound falls as well. [C21](c21-looking-hurts.md) is a case where looking hurts.
