# C21 · When looking hurts: tuning on the holdout score

*Synthetic world throughout, built by the first block below. The code blocks, run in order, are the demo script, also shipped as [c21_looking_hurts_demo.py](c21_looking_hurts_demo.py); its last block repeats the study over 60 worlds and takes about a minute. The clock and the custody salt were fixed when this page was generated, so the holdout ids and the model hashes are specific to that run; every other number is reproduced by the code shown.*

In this world cross-validation on the training rows already finds the five features that carry signal, so the holdout has nothing left to add. An analyst who keeps looking anyway tries every remaining feature and keeps each one that lowers the holdout error. The features kept are noise that won the comparison by chance: the holdout error falls, the live error rises, and the number reported from the holdout is wrong. The same loop run through `ask` keeps far fewer of them, stops when the tuning slice retires, and its certificate holds.

## In short

| | |
|---|---|
| data | 1,400 rows; 200 features, independent standard normal |
| target | continuous, so regression: y = 0.5 (x0 + x1 + x2 + x3 + x4) + noise, with standard normal noise; the other 195 features carry nothing |
| split | the first 400 rows train, the next 1,000 are the holdout; every row is drawn the same way, so the split is random in law |
| starting model | as in [C20](c20-few-rows.md): least squares on the top k features by correlation with y on the training rows, k chosen by 5-fold cross-validation; here k = 6, the five signal features and x79 |
| the loop | try each of the other 194 features in order of training correlation, refit least squares on the training rows, keep the feature if the model then scores better on the holdout |
| without looking | live MSE 1.017; the holdout certifies at most 1.080 |
| looking, no price | keeps 46 features, all noise; the holdout error falls from 1.010 to 0.943 while the live error rises to 1.081; the bound reported from the holdout, 1.007, is below the live error |
| looking through `ask` | 96 answers and 39.79 bits, then the tuning slice retires; keeps 6 features, all noise; live MSE 1.029; the exam slice certifies at most 1.158 |
| over 60 worlds | live MSE 1.0207 without looking, 1.0855 looking without a price, 1.0271 through `ask`; the bound reported without a price held in 7 worlds of 60, the exam bound in 56 |

Looking without a price cost 0.065 in live mean squared error on average and made the model worse in all 60 worlds, while the holdout said it had improved. Through `ask` the same loop cost 0.006, about a tenth as much, and the certificate held.

## The study

### 1. The world and the starting model

`world(rep)` draws world `rep`: 400 training rows and 1,000 holdout rows of 200 features. The features are independent standard normal, so a fit's live mean squared error is exact, `live_mse`. The starting model is the analyst of C20: the top k features by training correlation, with k chosen by 5-fold cross-validation, the ranking redone inside each fold. `rest` holds the 194 features the loop will try, in order of training correlation.

```python
import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
import provable as pv
from provable import certificates as ct
from provable.certificates import transcript as tr

os.chdir(tempfile.mkdtemp(prefix="looking_hurts_"))
P, K = 200, 30
beta = np.r_[np.full(5, 0.5), np.zeros(P - 5)]
names = [f"x{j}" for j in range(P)]


def world(rep, m=400):
    rng = np.random.default_rng(5000 + rep)
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


def ranked(X, y):
    return np.argsort(-np.abs((X - X.mean(0)).T @ (y - y.mean())))


def fit(X, y, keep):
    return np.linalg.lstsq(np.column_stack([np.ones(len(y)), X[:, keep]]), y, rcond=None)[0]


def cv_pick(rng, X, y):
    sq = np.zeros(K)
    for fold in np.array_split(rng.permutation(len(y)), 5):
        rest = np.setdiff1d(np.arange(len(y)), fold)
        order = ranked(X[rest], y[rest])
        for k in range(1, K + 1):
            b = fit(X[rest], y[rest], order[:k])
            sq[k - 1] += ((y[fold] - b[0] - X[fold][:, order[:k]] @ b[1:]) ** 2).sum()
    return int(np.argmin(sq)) + 1


def live_mse(keep, b):
    full = np.zeros(P)
    full[list(keep)] = b[1:]
    return 1.0 + b[0] ** 2 + float(((full - beta) ** 2).sum())


def predictor(keep, b):
    cols = [names[j] for j in keep]
    return lambda Z: b[0] + Z[cols].to_numpy() @ b[1:]


rng, train, hold, S, H = world(0)
Xs, ys = train[names].to_numpy(), train["y"].to_numpy()
Xh, yh = hold[names].to_numpy(), hold["y"].to_numpy()
order = ranked(Xs, ys)
k_cv = cv_pick(rng, Xs, ys)
start = [int(j) for j in order[:k_cv]]
b_start = fit(Xs, ys, start)
rest = [int(j) for j in order[k_cv:]]
print(f"cross-validation picks k = {k_cv}: {', '.join(names[j] for j in start)}; {len(rest)} features left to try")
```

```text
cross-validation picks k = 6: x3, x0, x4, x1, x2, x79; 194 features left to try
```

### 2. Without looking

```python
ho_a = pv.seal(hold[names], hold["y"], "book_no_look", lineage=H, baseline=float(ys.mean()), baseline_lineage=S,
               exam_share=0.0, label="no look")
cert_a = ho_a.open(predictor(start, b_start), lineage=S, label=f"k={k_cv}")
print(cert_a.claim)
```

```text
performance: mean squared loss of model dbc6884844b7 at most 1.08002 (FWER 0.05, CLT approximation; HO-C06A388DF9B6ED83, tuning slice, 0.00 bits spent)
```

The holdout, sealed without an exam slice and opened once at the clean bar for the starting model: 0 bits, a bound of 1.080 on the mean squared error.

### 3. Looking without a price

```python
def holdout_mse(keep, b):
    return float(((yh - b[0] - Xh[:, keep] @ b[1:]) ** 2).mean())


keep_u, b_u = list(start), b_start
for j in rest:
    b = fit(Xs, ys, keep_u + [j])
    if holdout_mse(keep_u + [j], b) < holdout_mse(keep_u, b_u):
        keep_u, b_u = keep_u + [j], b
loss = (yh - b_u[0] - Xh[:, keep_u] @ b_u[1:]) ** 2
reported = loss.mean() + tr.z_bar(0.05, 0) * loss.std(ddof=1) / np.sqrt(len(loss))
added = keep_u[len(start):]
print(f"kept {len(added)} of {len(rest)} features tried, {sum(j >= 5 for j in added)} of them noise")
print(f"holdout MSE {holdout_mse(start, b_start):.4f} -> {holdout_mse(keep_u, b_u):.4f}; reported bound at the clean bar {reported:.4f}")
```

```text
kept 46 of 194 features tried, 46 of them noise
holdout MSE 1.0098 -> 0.9426; reported bound at the clean bar 1.0070
```

This analyst holds the holdout rows and scores every candidate on all 1,000 of them. Of the 194 features tried, 46 lowered the holdout error and were kept, and all 46 are noise. The holdout error fell from 1.010 to 0.943, and the bound computed from it at the clean bar, 1.007, is the number this analyst would report. Each comparison is fair on its own. The loop as a whole is not, because the holdout score that chose the 46 features is then used to measure the model built from them.

### 4. The same loop through `ask`

```python
ho_b = pv.seal(hold[names], hold["y"], "book_look", lineage=H, baseline=float(ys.mean()), baseline_lineage=S, label="look")
first = ho_b.ask(predictor(start, b_start), lineage=S, label=f"k={k_cv}")
keep_v, b_v = list(start), b_start
answers = [first]
for j in rest:
    b = fit(Xs, ys, keep_v + [j])
    try:
        a = ho_b.ask(predictor(keep_v + [j], b), lineage=S, label=f"+{names[j]}")
    except ValueError as e:
        print(f"after {len(answers)} answers: {e}")
        break
    answers.append(a)
    if a.accepted:
        keep_v, b_v = keep_v + [j], b
added_v = keep_v[len(start):]
print(f"{len(answers)} answers, {sum(a.accepted for a in answers)} accepts, {ho_b.state['bits']:.2f} bits; "
      f"kept {len(added_v)} features, {sum(j >= 5 for j in added_v)} of them noise")
print(ho_b)
cert_b = ho_b.open(predictor(keep_v, b_v), lineage=S, label="incumbent")
print(cert_b.claim)
```

```text
after 96 answers: ask refused: the tuning slice of HO-8F1DBE7E2732E991 retired at #96 (threshold: s(L) 0.0489 <= retire_at 0.05); seal the next block for further questions
96 answers, 7 accepts, 39.79 bits; kept 6 features, 6 of them noise
-- holdout HO-8F1DBE7E2732E991 --
+----------------------+------------------------------+
| holdout              | HO-8F1DBE7E2732E991          |
| mode / route         | comparative / exam           |
| claim                | risk (squared loss)          |
| rows (tuning / exam) | 800 / 200                    |
| delta                | 0.05                         |
| code / eta           | kt / 1                       |
| asks (accepts)       | 96 (7)                       |
| bits spent           | 39.793                       |
| transcript bar z_L   | 7.4345                       |
| clean share s(L)     | 0.0489                       |
| retirement           | s(L) <= 0.05 (at 38.94 bits) |
| tuning slice         | retired                      |
| exam slice           | sealed                       |
| certificates         | none                         |
+----------------------+------------------------------+
performance: mean squared loss of model 14cab26f0d89 at most 1.15798 (FWER 0.05, CLT approximation; HO-8F1DBE7E2732E991, exam slice, 39.79 bits spent)
```

Each candidate is the incumbent plus one feature, fit on the training rows, and the holdout answers ACCEPT only when it beats the incumbent by one paired standard error on the 800 tuning rows. Seven of the 96 answers were accepts: the starting model and six noise features. Every answer was priced under the sealed code. At 39.79 bits the clean share fell to 0.049, below the retirement threshold of 0.05, the tuning slice retired with that answer, and the next ask was refused at no cost. The exam slice, which answered none of the 96 questions, opens once at the clean bar for the incumbent: a bound of 1.158.

### 5. Live

```python
rows = [["no look", k_cv, 0.0, cert_a.artifact["estimate"], cert_a.artifact["bound"], live_mse(start, b_start)],
        ["look, no price", len(keep_u), None, holdout_mse(keep_u, b_u), reported, live_mse(keep_u, b_u)],
        ["look through ask", len(keep_v), ho_b.state["bits"], cert_b.artifact["estimate"], cert_b.artifact["bound"], live_mse(keep_v, b_v)]]
print(tabulate(rows, headers=["route", "features", "bits", "reported MSE", "95% bound", "live MSE"], tablefmt="psql", floatfmt=".4f"))
```

```text
+------------------+------------+---------+----------------+-------------+------------+
| route            |   features |    bits |   reported MSE |   95% bound |   live MSE |
|------------------+------------+---------+----------------+-------------+------------|
| no look          |          6 |  0.0000 |         1.0098 |      1.0800 |     1.0168 |
| look, no price   |         52 |         |         0.9426 |      1.0070 |     1.0806 |
| look through ask |         12 | 39.7934 |         0.9941 |      1.1580 |     1.0289 |
+------------------+------------+---------+----------------+-------------+------------+
```

The reported column is what each route reports: the holdout estimate behind its bound, on all 1,000 rows without an exam slice and on the 200 exam rows through `ask`. Without a price, looking took the live error from 1.017 to 1.081 while the reported error went from 1.010 to 0.943, and the reported bound, 1.007, sits 0.074 below the live error. Through `ask` the live error rose to 1.029, and the bound of 1.158 holds.

### 6. Sixty worlds

```python
def study(rep):
    """The three routes on world rep: {route: (features, bits, reported MSE, bound, live MSE)}."""
    rng, train, hold, S, H = world(rep)
    Xs, ys = train[names].to_numpy(), train["y"].to_numpy()
    Xh, yh = hold[names].to_numpy(), hold["y"].to_numpy()
    order = ranked(Xs, ys)
    k_cv = cv_pick(rng, Xs, ys)
    start, rest = [int(j) for j in order[:k_cv]], [int(j) for j in order[k_cv:]]
    hmse = lambda keep, b: float(((yh - b[0] - Xh[:, keep] @ b[1:]) ** 2).mean())
    books = tempfile.mkdtemp(dir=".")
    out = {}
    b0 = fit(Xs, ys, start)
    no_look = pv.seal(hold[names], hold["y"], os.path.join(books, "a"), lineage=H, baseline=float(ys.mean()),
                      baseline_lineage=S, exam_share=0.0, seed=rep)
    cert = no_look.open(predictor(start, b0), lineage=S)
    out["no look"] = (start, 0.0, cert.artifact["estimate"], cert.artifact["bound"], live_mse(start, b0))
    keep, bk = list(start), b0
    for j in rest:
        b = fit(Xs, ys, keep + [j])
        if hmse(keep + [j], b) < hmse(keep, bk):
            keep, bk = keep + [j], b
    loss = (yh - bk[0] - Xh[:, keep] @ bk[1:]) ** 2
    out["look, no price"] = (keep, None, loss.mean(), loss.mean() + tr.z_bar(0.05, 0) * loss.std(ddof=1) / np.sqrt(len(loss)), live_mse(keep, bk))
    look = pv.seal(hold[names], hold["y"], os.path.join(books, "b"), lineage=H, baseline=float(ys.mean()),
                   baseline_lineage=S, seed=rep)
    keep, bk, retired = list(start), b0, False
    look.ask(predictor(start, b0), lineage=S)
    for j in rest:
        b = fit(Xs, ys, keep + [j])
        try:
            if look.ask(predictor(keep + [j], b), lineage=S).accepted:
                keep, bk = keep + [j], b
        except ValueError:
            retired = True
            break
    cert = look.open(predictor(keep, bk), lineage=S)
    out["look through ask"] = (keep, look.state["bits"], cert.artifact["estimate"], cert.artifact["bound"], live_mse(keep, bk))
    return out, len(start), retired


runs = [study(rep) for rep in range(60)]
table = []
for route in ("no look", "look, no price", "look through ask"):
    r = [run[route] for run, _, _ in runs]
    noise = np.mean([sum(j >= 5 for j in x[0][n:]) for (run, n, _), x in zip(runs, r)])
    table.append([route, np.mean([x[2] for x in r]), np.mean([x[4] for x in r]), np.mean([x[3] for x in r]), f"{sum(x[3] >= x[4] for x in r)} of 60",
                  noise, np.mean([x[1] for x in r]) if r[0][1] is not None else None])
print(tabulate(table, headers=["route", "mean reported MSE", "mean live MSE", "mean 95% bound", "bound held", "noise features kept", "mean bits"],
               tablefmt="psql", floatfmt=".4f"))
for route in ("look, no price", "look through ask"):
    d = [run[route][4] - run["no look"][4] for run, _, _ in runs]
    print(f"{route:16s} against no look: live worse in {sum(x > 1e-12 for x in d)} worlds, better in {sum(x < -1e-12 for x in d)}")
print(f"the tuning slice retired before the loop ended in {sum(ret for _, _, ret in runs)} worlds of 60")
```

```text
+------------------+---------------------+-----------------+------------------+--------------+-----------------------+-------------+
| route            |   mean reported MSE |   mean live MSE |   mean 95% bound | bound held   |   noise features kept |   mean bits |
|------------------+---------------------+-----------------+------------------+--------------+-----------------------+-------------|
| no look          |              1.0122 |          1.0207 |           1.0860 | 54 of 60     |                0.0000 |      0.0000 |
| look, no price   |              0.9568 |          1.0855 |           1.0269 | 7 of 60      |               52.4500 |             |
| look through ask |              1.0137 |          1.0271 |           1.1746 | 56 of 60     |                6.3167 |     40.3979 |
+------------------+---------------------+-----------------+------------------+--------------+-----------------------+-------------+
look, no price   against no look: live worse in 60 worlds, better in 0
look through ask against no look: live worse in 59 worlds, better in 1
the tuning slice retired before the loop ended in 60 worlds of 60
```

Without a price, the loop kept about 52 noise features on average, the reported error fell from 1.0122 to 0.9568 while the live error rose from 1.0207 to 1.0855, and the reported bound held in 7 worlds of 60. Through `ask` the loop kept about 6 noise features, the tuning slice retired before the loop ended in every world, at about 40 bits, the live error rose by 0.006, and the exam bound held in 56 worlds of 60, about what the Gaussian bar gives on squared errors at 200 rows, as [C20](c20-few-rows.md) notes. In 59 worlds of 60 both looking routes ended worse live than not looking; through `ask` the loss was a tenth of the other.

Looking hurts when the holdout's score drives many decisions and nothing prices them: every comparison a noise feature wins by chance is kept, and the score that selected the features can no longer measure the model. The price does not stop all of it. The comparison rule turned most noise features away, the retirement rule ended the loop, and the exam slice, untouched by the loop, certified what was left. [C19](c19-changed-world.md) and [C20](c20-few-rows.md) are the cases where looking gains.
