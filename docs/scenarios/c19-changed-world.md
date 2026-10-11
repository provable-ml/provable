# C19 · When looking gains: the holdout comes from a changed world

*Synthetic world throughout, built by the first block below. The code blocks, run in order, are the demo script, also shipped as [c19_changed_world_demo.py](c19_changed_world_demo.py); its last block repeats the study over 40 worlds and takes several minutes. The clock and the custody salt were fixed when this page was generated, so the holdout ids and the model hashes are specific to that run; every other number is reproduced by the code shown.*

When the holdout is drawn like the training rows and those rows are enough to rank the candidates, the holdout's answers have nothing to add to cross-validation on the training rows, and looking buys nothing but a wider certificate. Step 5 shows that case on these rows. This page changes one thing: at the first holdout row the coefficient of x1 changes sign, and it stays changed.

## In short

| | |
|---|---|
| data | 3,000 hourly rows; six features, independent standard normal |
| target | binary, so classification: y = 1 when 0.9 x0 + b x1 + 0.8 x2 [x3 > 0] + noise > 0, with standard normal noise |
| the change | b = -0.6 on the training rows; b = +0.6 from the first holdout row on |
| split | by time, not shuffled: rows 1 to 2,000 train, rows 2,001 to 3,000 are the holdout |
| candidates | six models fit on the training rows only: logistic regression on x0, on x0 and x1, and on all six features; boosted trees of depth 2, 3 and 5 |
| without looking | 5-fold cross-validation on the training rows picks the logistic regression on all six features; the holdout certifies its mean Brier loss at most 0.289; live, it scores 0.268 and errs on 39.6 percent of rows |
| looking | six yes or no answers from the holdout, 5.61 bits, pick the logistic regression on x0 alone; the exam slice certifies it at most 0.245; live, it scores 0.197 and errs on 30.6 percent |
| over 40 worlds | live Brier 0.2558 without looking and 0.1969 looking, the best of the six in every world; bound 0.2814 against 0.2345 |
| without the change, 40 worlds | live Brier 0.1595 without looking and 0.1601 looking; bound 0.1781 against 0.2070, for 7.52 bits |

Cross-validation ranks the candidates under the law of the training rows, the only law it can see. The holdout's answers come from the new law. The price of looking is what it is in [C18](c18-end-to-end.md), bits and a certificate taken on a fifth of the rows, and here the gain is larger than the price: the bound falls with the live score.

## The study

### 1. The world and the candidates

`world(rep, b_new)` builds world `rep`: 2,000 training rows under b = -0.6 and 1,000 holdout rows under `b_new`. The features and the noise depend only on `rep`, so for one `rep` the changed and the unchanged world share the training rows, the candidates, and the holdout's features and noise, and differ only in the coefficient of x1 on the holdout rows. `later` stands in for the rows after the holdout, 200,000 under each law; they play no part in any choice and only score the chosen models live.

```python
import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
import provable as pv
from provable import certificates as ct

os.chdir(tempfile.mkdtemp(prefix="changed_world_"))
feats = [f"x{i}" for i in range(6)]


def draw(rng, n):
    return pd.DataFrame(rng.normal(size=(n, 6)), columns=feats), rng.normal(size=n)


def label(X, noise, b1):
    return X.assign(y=(0.9 * X["x0"] + b1 * X["x1"] + 0.8 * X["x2"] * (X["x3"] > 0) + noise > 0).astype(float))


def world(rep, b_new):
    rng = np.random.default_rng(2000 + rep)
    panel = pd.concat([label(*draw(rng, 2000), -0.6), label(*draw(rng, 1000), b_new)], ignore_index=True)
    panel["start_utc"] = pd.date_range("2026-01-01", periods=3000, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    train, hold = panel.iloc[:2000], panel.iloc[2000:]
    S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=f.index)) for f in (train, hold))
    return train, hold, S, H


later = {b: label(*draw(np.random.default_rng(999), 200_000), b) for b in (0.6, -0.6)}
models = {"logit x0": (LogisticRegression, {}, ["x0"]),
          "logit x0 x1": (LogisticRegression, {}, ["x0", "x1"]),
          "logit all": (LogisticRegression, {}, feats),
          "gbm depth 2": (GradientBoostingClassifier, {"max_depth": 2, "random_state": 0}, feats),
          "gbm depth 3": (GradientBoostingClassifier, {"max_depth": 3, "random_state": 0}, feats),
          "gbm depth 5": (GradientBoostingClassifier, {"max_depth": 5, "random_state": 0}, feats)}


def fit(name, rows):
    make, kw, cols = models[name]
    m = make(**kw).fit(rows[cols], rows["y"])
    return lambda X: m.predict_proba(X[cols])[:, 1]


def brier(f, rows):
    return float(((rows["y"].to_numpy() - f(rows)) ** 2).mean())


def error(f, rows):
    return float(((f(rows) > 0.5) != rows["y"].to_numpy()).mean())


def cv_brier(name, rows, rep):
    return float(np.mean([brier(fit(name, rows.iloc[a]), rows.iloc[b]) for a, b in KFold(5, shuffle=True, random_state=rep).split(rows)]))


train, hold, S, H = world(0, b_new=0.6)
candidates = {name: fit(name, train) for name in models}
base = float(train["y"].mean())
```

### 2. Without looking

```python
cv = {name: cv_brier(name, train, 0) for name in models}
print(tabulate(list(cv.items()), headers=["model", "5-fold CV Brier on the training rows"], tablefmt="psql", floatfmt=".4f"))
pick = min(cv, key=cv.get)
ho_a = pv.seal(hold[feats], hold["y"], "book_no_look", lineage=H, loss="brier", baseline=base,
               baseline_lineage=S, exam_share=0.0, label="no look")
cert_a = ho_a.open(candidates[pick], lineage=S, label=pick)
print(cert_a.claim)
```

```text
+-------------+----------------------------------------+
| model       |   5-fold CV Brier on the training rows |
|-------------+----------------------------------------|
| logit x0    |                                 0.1956 |
| logit x0 x1 |                                 0.1679 |
| logit all   |                                 0.1573 |
| gbm depth 2 |                                 0.1590 |
| gbm depth 3 |                                 0.1589 |
| gbm depth 5 |                                 0.1674 |
+-------------+----------------------------------------+
performance: mean brier loss of model dede2c9d7f85 at most 0.288907 (FWER 0.05, exact, betting bound; HO-D538B551825F180A, tuning slice, 0.00 bits spent, probabilities clipped to [0, 1])
```

Cross-validation picks the logistic regression on all six features, with the boosted trees of depth 2 and 3 close behind. The analyst seals the holdout without an exam slice and opens it at once for that model: no answers, 0 bits, the clean level, and, the Brier loss lying in [0, 1], the exact betting bound: 0.289 on the mean Brier loss. The certificate is correct, and it cannot show that the model beats predicting 0.5 for every row, which scores 0.25. Choosing another model in the light of that number would be a look, and nothing has priced it.

### 3. Looking

```python
ho_b = pv.seal(hold[feats], hold["y"], "book_look", lineage=H, loss="brier", baseline=base,
               baseline_lineage=S, label="look")
for name, f in candidates.items():
    print(f"{name:12s}", ho_b.ask(f, lineage=S, label=name))
chosen = next(r["label"] for r in ho_b.transcript().rows if r["seq"] == ho_b.state["incumbent"])
cert_b = ho_b.open(candidates[chosen], lineage=S, label=chosen)
print(cert_b.claim)
```

```text
logit x0     ask 1 (log #1): ACCEPT | 1.000 bits; spent 1.00; z_L 1.960; s(L) 0.704; tuning open
logit x0 x1  ask 2 (log #2): REJECT | 2.000 bits; spent 3.00; z_L 2.498; s(L) 0.434; tuning open
logit all    ask 3 (log #3): REJECT | 1.000 bits; spent 4.00; z_L 2.734; s(L) 0.362; tuning open
gbm depth 2  ask 4 (log #4): REJECT | 0.678 bits; spent 4.68; z_L 2.886; s(L) 0.325; tuning open
gbm depth 3  ask 5 (log #5): REJECT | 0.515 bits; spent 5.19; z_L 2.996; s(L) 0.301; tuning open
gbm depth 5  ask 6 (log #6): REJECT | 0.415 bits; spent 5.61; z_L 3.083; s(L) 0.285; tuning open
performance: mean brier loss of model 33c1ac24fa24 at most 0.244846 (FWER 0.05, exact, betting bound; HO-99D8FBD7FF313FD9, exam slice, 5.61 bits spent, probabilities clipped to [0, 1])
```

Sealed with the defaults, as in C18: 800 tuning rows answer, 200 exam rows certify, and the training base rate is the first incumbent. The logistic regression on x0 beats it and is accepted for 1 bit. Every later candidate uses x1, with the sign the old rows taught it, and is rejected. Six answers cost 5.61 bits. The exam slice opens once, at the clean level, for the incumbent: a bound of 0.245.

### 4. Live

```python
live = later[0.6]
best = min(models, key=lambda name: brier(candidates[name], live))
rows = [["no look", pick, 0.0, cert_a.artifact["bound"]], ["look", chosen, ho_b.state["bits"], cert_b.artifact["bound"]], ["best of the six", best, None, None]]
print(tabulate([r + [brier(candidates[r[1]], live), error(candidates[r[1]], live)] for r in rows],
               headers=["route", "model", "bits", "95% bound", "live Brier", "live error"], tablefmt="psql", floatfmt=".4f"))
```

```text
+-----------------+-----------+--------+-------------+--------------+--------------+
| route           | model     |   bits |   95% bound |   live Brier |   live error |
|-----------------+-----------+--------+-------------+--------------+--------------|
| no look         | logit all | 0.0000 |      0.2889 |       0.2675 |       0.3963 |
| look            | logit x0  | 5.6077 |      0.2448 |       0.1967 |       0.3056 |
| best of the six | logit x0  |        |             |       0.1967 |       0.3056 |
+-----------------+-----------+--------+-------------+--------------+--------------+
```

On 200,000 rows from after the holdout, looking cut the Brier score from 0.268 to 0.197 and the error rate from 39.6 to 30.6 percent, and its model is the best of the six. The bound fell too, from 0.289 to 0.245. The look cost 5.61 bits and a certificate on 200 rows instead of 1,000, and what it bought is larger.

### 5. The same rows without the change

```python
def study(rep):
    """Both routes on world rep, with the change and without it: the training rows, the candidates and the
    cross-validated pick are shared, and the holdout differs only in the coefficient of x1."""
    train, _, S, _ = world(rep, 0.6)
    candidates = {name: fit(name, train) for name in models}
    cv = {name: cv_brier(name, train, rep) for name in models}
    pick, base, out = min(cv, key=cv.get), float(train["y"].mean()), {}
    for b_new in (0.6, -0.6):
        _, hold, _, H = world(rep, b_new)
        books = tempfile.mkdtemp(dir=".")
        no_look = pv.seal(hold[feats], hold["y"], os.path.join(books, "a"), lineage=H, loss="brier", baseline=base,
                          baseline_lineage=S, exam_share=0.0, seed=rep)
        look = pv.seal(hold[feats], hold["y"], os.path.join(books, "b"), lineage=H, loss="brier", baseline=base,
                       baseline_lineage=S, seed=rep)
        for name, f in candidates.items():
            look.ask(f, lineage=S, label=name)
        chosen = next(r["label"] for r in look.transcript().rows if r["seq"] == look.state["incumbent"])
        for route, ho, name in (("no look", no_look, pick), ("look", look, chosen)):
            cert = ho.open(candidates[name], lineage=S, label=name)
            out[b_new, route] = (name, ho.state["bits"], cert.artifact["bound"], brier(candidates[name], later[b_new]))
        best = min(models, key=lambda name: brier(candidates[name], later[b_new]))
        out[b_new, "best of the six"] = (best, None, None, brier(candidates[best], later[b_new]))
    return out


one = study(0)
print(tabulate([[("changed" if b == 0.6 else "unchanged"), route, *v] for (b, route), v in one.items()],
               headers=["holdout", "route", "model", "bits", "95% bound", "live Brier"], tablefmt="psql", floatfmt=".4f"))
```

```text
+-----------+-----------------+-------------+--------+-------------+--------------+
| holdout   | route           | model       |   bits |   95% bound |   live Brier |
|-----------+-----------------+-------------+--------+-------------+--------------|
| changed   | no look         | logit all   | 0.0000 |      0.2889 |       0.2675 |
| changed   | look            | logit x0    | 5.6077 |      0.2448 |       0.1967 |
| changed   | best of the six | logit x0    |        |             |       0.1967 |
| unchanged | no look         | logit all   | 0.0000 |      0.1700 |       0.1615 |
| unchanged | look            | logit all   | 7.6781 |      0.2195 |       0.1615 |
| unchanged | best of the six | gbm depth 2 |        |             |       0.1611 |
+-----------+-----------------+-------------+--------+-------------+--------------+
```

`study(rep)` runs both routes on world `rep`, with the change and without it; on world 0 its changed rows reproduce steps 2 to 4. Without the change the holdout's answers end on the logistic regression on all six features, the model cross-validation picked, so looking changes nothing live (0.1615 either way) and costs 7.68 bits and a bound of 0.2195 instead of 0.1700. The best of the six, the boosted trees of depth 2 at 0.1611, was picked by neither route.

### 6. Forty worlds

```python
runs = [study(rep) for rep in range(40)]
table = []
for b_new, what in ((0.6, "changed"), (-0.6, "unchanged")):
    for route in ("no look", "look", "best of the six"):
        r = [run[b_new, route] for run in runs]
        held = f"{sum(x[2] >= x[3] for x in r)} of 40" if r[0][2] is not None else ""
        table.append([what, route, np.mean([x[3] for x in r]), np.mean([x[2] for x in r]) if r[0][2] is not None else None, held,
                      np.mean([x[1] for x in r]) if r[0][1] is not None else None])
print(tabulate(table, headers=["holdout", "route", "mean live Brier", "mean 95% bound", "bound held", "mean bits"], tablefmt="psql", floatfmt=".4f"))
for b_new, what in ((0.6, "changed"), (-0.6, "unchanged")):
    print(f"{what:9s} models chosen:", {route: pd.Series([run[b_new, route][0] for run in runs]).value_counts().to_dict() for route in ("no look", "look")})
    print(f"{what:9s} the best of the six chosen in:", {route: sum(run[b_new, route][0] == run[b_new, "best of the six"][0] for run in runs) for route in ("no look", "look")}, "worlds of 40")
```

```text
+-----------+-----------------+-------------------+------------------+--------------+-------------+
| holdout   | route           |   mean live Brier |   mean 95% bound | bound held   |   mean bits |
|-----------+-----------------+-------------------+------------------+--------------+-------------|
| changed   | no look         |            0.2558 |           0.2814 | 40 of 40     |      0.0000 |
| changed   | look            |            0.1969 |           0.2345 | 40 of 40     |      5.6077 |
| changed   | best of the six |            0.1969 |                  |              |             |
| unchanged | no look         |            0.1595 |           0.1781 | 40 of 40     |      0.0000 |
| unchanged | look            |            0.1601 |           0.2070 | 40 of 40     |      7.5203 |
| unchanged | best of the six |            0.1591 |                  |              |             |
+-----------+-----------------+-------------------+------------------+--------------+-------------+
changed   models chosen: {'no look': {'gbm depth 2': 25, 'logit all': 11, 'gbm depth 3': 4}, 'look': {'logit x0': 40}}
changed   the best of the six chosen in: {'no look': 0, 'look': 40} worlds of 40
unchanged models chosen: {'no look': {'gbm depth 2': 25, 'logit all': 11, 'gbm depth 3': 4}, 'look': {'logit all': 26, 'gbm depth 2': 13, 'gbm depth 3': 1}}
unchanged the best of the six chosen in: {'no look': 28, 'look': 17} worlds of 40
```

With the change, looking picked the logistic regression on x0 in all 40 worlds, the best of the six under the new law, which cross-validation picked in none; the mean live Brier score fell from 0.2558 to 0.1969, and the mean bound from 0.2814 to 0.2345. Without the change, the two routes picked among the same three close models, and looking lost 0.0006 in live score while the bound widened from 0.1781 to 0.2070. Cross-validation picked the best of the six in 28 worlds and the look in 17: the look accepts a candidate only when it wins by one paired standard error on 800 rows, so among models this close it often keeps the one asked first. The bound held in all 40 worlds on every route; the betting bound misses with probability at most 0.05 whatever the shape of a loss within its range.

The holdout's answers carry what the training rows cannot: how each candidate does under the law it will meet. When that law has moved, they pay for themselves. [C20](c20-few-rows.md) is the other way a look gains, with no change of law at all, and [C21](c21-looking-hurts.md) is a case where looking hurts.
