# C23 · Which choices to tune on the holdout: the transport screen

*Synthetic world throughout, built by the first block below. The code blocks below, run in order, are the demo script, also shipped as [c23_route_demo.py](c23_route_demo.py); its last block repeats the study over 100 worlds, and the whole script runs in about half a minute. The clock and the custody salt were fixed when this page was generated, so the holdout and route ids, the model hashes and the dates are specific to that run; every other number is reproduced by the code shown.*

Some choices have the same effect in every stretch of time, like a feature that always helps by the same amount. Others help in some states of the world and hurt in others, like a coefficient that is right in one state only. A holdout covers a few weeks, and an analyst usually remembers what those weeks were like. For a choice whose effect depends on the state, that memory steers the choice from outside the holdout's log, and the certificate describes the holdout weeks rather than the weeks to come. `pv.route` reads the choice's effect on the training weeks only, week by week, and routes it: through the holdout (VAULT) when the worst bias memory could add stays within one clean bar of holdout noise, and into a forward registration (FORWARD) otherwise.

## In short

| | |
|---|---|
| data | 52 weeks of hourly rows: y = 0.4 x0 + 0.3 x1 + noise in weeks of state A, and y = 0.4 x0 + noise in weeks of state B; each week is in state A or B with probability 1/2, independently |
| split | the first 48 weeks train; the next 4 are the holdout, with an exam slice of a quarter of its rows |
| choices | x0: the constant against least squares on x0. x1: least squares on x0 against least squares on x0 and x1 fitted on the training weeks in state A |
| the screen | every training week scored by models fitted on the other weeks; the spread of the weekly effect bounded at 0.95 (Q-profile); VAULT when that bound times sqrt(2 ln 2), the worst bias memory of two states could add, stays within one clean bar of holdout noise and the fluctuation statistic stays below its critical value |
| world 29 | x0: the spread is at most 0.042 and the memory bias at most 0.049, within the bar of 0.061: VAULT. x1: the spread is at most 0.117 and the memory bias up to 0.138, against a bar of 0.046: FORWARD |
| through the holdout anyway | three of world 29's four holdout weeks were in state A; the holdout accepts x1 and certifies a gain of at least 0.046, while its long-run gain is -0.001 |
| over 100 worlds | x0 routed to VAULT in 79, and its certificates carried over: the bound exceeded the long-run gain in 5 of 99. x1 routed FORWARD in all 100; tuned through the holdout anyway it was certified in 40, with a mean gain of 0.048 on the exam rows against a long-run gain of 0.000, and its bound exceeded the long-run gain in 11 of the 40 |

## The study

### 1. The world and the choices

`world(rep)` draws world `rep`: 52 weeks of 168 hourly rows, each week in state A or B. A choice is a pair of fitted models, the default and the alternative. For x0 the default is the constant and the alternative least squares on x0. For x1 the default is least squares on x0, and the alternative adds x1 at the coefficient it has in state A, fitted on the training weeks in that state: right in half the weeks, wrong in the other half. `crossfit` scores every training week with the pair fitted on the other 47 weeks and returns the default's loss minus the alternative's, row by row. `transported` is the exact long-run risk of a fitted model, with half the weeks in each state.

```python
import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
import provable as pv
from provable import certificates as ct

os.chdir(tempfile.mkdtemp(prefix="route_"))
per, M_S, M_H = 168, 48, 4
a1 = 0.3


def world(rep):
    rng = np.random.default_rng(23000 + rep)
    M = M_S + M_H
    state = rng.choice(["A", "B"], size=M)
    panel = pd.DataFrame(rng.normal(size=(M * per, 2)), columns=["x0", "x1"])
    panel["week"] = np.repeat(np.arange(M), per)
    panel["state"] = np.repeat(state, per)
    panel["y"] = 0.4 * panel["x0"] + np.where(panel["state"] == "A", a1, 0.0) * panel["x1"] + rng.normal(size=M * per)
    panel["start_utc"] = pd.date_range("2026-01-05", periods=M * per, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    return panel


def ols(frame, cols):
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in cols])
    b = np.linalg.lstsq(X, frame["y"].to_numpy(), rcond=None)[0]
    return np.r_[b[0], [b[1 + cols.index(c)] if c in cols else 0.0 for c in ("x0", "x1")]]


def fit(choice, frame):
    """The two alternatives of a choice, fitted on frame: (the default, the alternative)."""
    if choice == "x0":
        return ols(frame, []), ols(frame, ["x0"])
    return ols(frame, ["x0"]), ols(frame[frame["state"] == "A"], ["x0", "x1"])


def loss(b, frame):
    return (frame["y"].to_numpy() - b[0] - b[1] * frame["x0"].to_numpy() - b[2] * frame["x1"].to_numpy()) ** 2


def crossfit(choice, train):
    """The default's loss minus the alternative's on every training row, each week scored by models fitted on the other weeks."""
    d = np.empty(len(train))
    for wk in train["week"].unique():
        here = (train["week"] == wk).to_numpy()
        base, alt = fit(choice, train[~here])
        d[here] = loss(base, train[here]) - loss(alt, train[here])
    return d


def transported(b):
    """The live risk of a fitted model over the long run, half the weeks in each state."""
    return 1.0 + b[0] ** 2 + (0.4 - b[1]) ** 2 + 0.5 * (a1 - b[2]) ** 2 + 0.5 * b[2] ** 2


panel = world(29)
train, hold = panel[panel["week"] < M_S], panel[panel["week"] >= M_S]
S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "weeks", rows=f.index)) for f in (train, hold))
print(f"{M_S} training weeks ({(train.groupby('week')['state'].first() == 'A').sum()} in state A), {M_H} holdout weeks in states {''.join(hold.groupby('week')['state'].first())}")
```

```text
48 training weeks (28 in state A), 4 holdout weeks in states BAAA
```

### 2. The screen, on the training weeks

```python
holdouts, routes = {}, {}
for choice in ("x0", "x1"):
    base, alt = fit(choice, train)
    book = ct.Ledger(f"book_{choice}", budget=0.10)
    holdouts[choice] = pv.seal(hold[["x0", "x1"]], hold["y"], book, lineage=H, claim="gain", exam_share=0.25,
                               baseline=lambda Z, b=base: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"], baseline_lineage=S, label=choice)
    routes[choice] = pv.route(crossfit(choice, train), train["week"], holdout=holdouts[choice], lineage=S, ledger=book,
                              label="x0 in" if choice == "x0" else "x1 at its state-A coefficient")
    print(routes[choice])
```

```text
-- route: x0 in --
+--------------------------------+---------------------------------------------------------------------------------------+
| verdict                        | VAULT: the worst memory bias 0.04892 is within one clean bar of holdout noise 0.06099 |
| rows, blocks                   | 8064, 48                                                                              |
| mean effect (a minus b)        | 0.151634                                                                              |
| Cochran's Q (p)                | 45.73 (0.525) on 47 df                                                                |
| spread of the effect tau_c     | 0 (DerSimonian-Laird), at most 0.04155 at 0.95 (Q-profile)                            |
| worst memory bias              | tau_c^UCB sqrt(2 ln 2) = 0.04892                                                      |
| one clean bar of holdout noise | z_0 se_H = 1.6449 x 0.03708 = 0.06099 (h = 504)                                       |
| fluctuation statistic          | 1.3002 against 2.6367 (window 0.3 of the rows)                                        |
| recorded                       | RT-3A9A6AB97D5823FF in book_x0                                                        |
+--------------------------------+---------------------------------------------------------------------------------------+
-- route: x1 at its state-A coefficient --
+--------------------------------+--------------------------------------------------------------------------------------+
| verdict                        | FORWARD: the worst memory bias 0.1379 exceeds one clean bar of holdout noise 0.04569 |
| rows, blocks                   | 8064, 48                                                                             |
| mean effect (a minus b)        | 0.0111127                                                                            |
| Cochran's Q (p)                | 233.4 (1.55e-26) on 47 df                                                            |
| spread of the effect tau_c     | 0.09315 (DerSimonian-Laird), at most 0.1171 at 0.95 (Q-profile)                      |
| worst memory bias              | tau_c^UCB sqrt(2 ln 2) = 0.1379                                                      |
| one clean bar of holdout noise | z_0 se_H = 1.6449 x 0.02778 = 0.04569 (h = 504)                                      |
| fluctuation statistic          | 2.4152 against 2.6367 (window 0.3 of the rows)                                       |
| recorded                       | RT-09322F36E5B9CA85 in book_x1                                                       |
+--------------------------------+--------------------------------------------------------------------------------------+
```

Each choice gets its own holdout, sealed on the same four weeks in its own book, with the choice's default as the baseline. The holdout is sealed before the screen because the screen takes the number of rows that would answer the choice from its tuning slice, 504, and refuses rows still in its custody. The screen spends no bits and charges nothing; with `ledger=` it is recorded in the book as the choice's transport status.

For x0 the 48 weekly effects agree within their noise (Cochran's Q gives p = 0.53), and the Q-profile bound puts the spread across weeks at no more than 0.042. Memory of the holdout weeks could then bias a certificate by at most 0.042 times sqrt(2 ln 2) = 0.049, less than one clean bar of holdout noise, z_0 times the standard error of the effect on 504 rows, 0.061. For x1 the weekly effects split into two groups, as the world is built: near +0.09 = 0.3^2 in state A and near -0.09 in state B (Q gives p = 2e-26). The spread is at most 0.117, and the memory bias could reach 0.138, three times the bar. The fluctuation statistic, the largest moving-window sum of the demeaned effects, stays below its critical value for both choices. The states change from week to week at random, and a window of 30 percent of the rows averages over many weeks; the weekly Q sees what the window misses, while the window catches a slow drift or a lasting break.

### 3. Through the holdout anyway

```python
for choice in ("x0", "x1"):
    base, alt = fit(choice, train)
    ho = holdouts[choice]
    ans = ho.ask(lambda Z, b=alt: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"], lineage=S, label="alternative")
    line = f"{choice}: route {routes[choice].verdict}; the holdout answers {ans}"
    if ans.accepted:
        cert = ho.open(lambda Z, b=alt: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"], lineage=S)
        line += f"; certified gain at least {cert.artifact['bound']:.4f}, long-run gain {transported(base) - transported(alt):.4f}"
    print(line)
```

```text
x0: route VAULT; the holdout answers ask 1 (log #1): ACCEPT | 1.000 bits; spent 1.00; z_L 1.960; s(L) 0.704; tuning open; certified gain at least 0.0359, long-run gain 0.1599
x1: route FORWARD; the holdout answers ask 1 (log #1): ACCEPT | 1.000 bits; spent 1.00; z_L 1.960; s(L) 0.704; tuning open; certified gain at least 0.0460, long-run gain -0.0011
```

Both alternatives are accepted on the tuning slice and certified on the exam slice. The certificate for x0 bounds a gain the choice keeps: at least 0.036, against a long-run gain of 0.160. The certificate for x1 is true of the four holdout weeks, three of which were in state A, where x1 helps: a gain of at least 0.046 on their rows. Over the long run, half the weeks in each state, the gain is -0.001. Nothing in the holdout's log is wrong. The certificate answers a question about the holdout weeks, and a choice whose effect depends on the state turns it into a question about which states those weeks happened to be in.

### 4. A hundred worlds

```python
def study(rep):
    panel = world(rep)
    train, hold = panel[panel["week"] < M_S], panel[panel["week"] >= M_S]
    S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "weeks", rows=f.index)) for f in (train, hold))
    out = {"mostly A": int((hold.groupby("week")["state"].first() == "A").sum()) >= 3}
    for choice in ("x0", "x1"):
        base, alt = fit(choice, train)
        book = ct.Ledger(tempfile.mkdtemp(dir="."), budget=0.10)
        ho = pv.seal(hold[["x0", "x1"]], hold["y"], book, lineage=H, claim="gain", exam_share=0.25, seed=rep,
                     baseline=lambda Z, b=base: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"], baseline_lineage=S)
        f = lambda Z, b=alt: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"]
        out[choice + " route"] = pv.route(crossfit(choice, train), train["week"], holdout=ho, lineage=S).verdict
        out[choice + " long run"] = transported(base) - transported(alt)
        cert = ho.open(f, lineage=S) if ho.ask(f, lineage=S).accepted else None
        out[choice + " estimate"] = None if cert is None else cert.artifact["estimate"]
        out[choice + " bound"] = None if cert is None else cert.artifact["bound"]
    return out


r = pd.DataFrame([study(rep) for rep in range(100)])
rows = []
for label, choice, keep in (("x0 in", "x0", r.index == r.index), ("x1, asked every time", "x1", r.index == r.index), ("x1, asked when memory says mostly A", "x1", r["mostly A"])):
    s = r[keep]
    cert = s[s[choice + " bound"].notna()]
    rows.append([label, f"{(s[choice + ' route'] == 'VAULT').sum()} of {len(s)}", f"{len(cert)} of {len(s)}", cert[choice + " estimate"].mean(), cert[choice + " bound"].mean(),
                 cert[choice + " long run"].mean(), int((cert[choice + " bound"] > cert[choice + " long run"]).sum())])
print(tabulate(rows, headers=["choice", "routed to VAULT", "certified", "mean gain on the exam rows", "mean certified bound", "mean long-run gain", "bound above the long-run gain"], tablefmt="psql", floatfmt=".4f"))
```

```text
+-------------------------------------+-------------------+-------------+------------------------------+------------------------+----------------------+---------------------------------+
| choice                              | routed to VAULT   | certified   |   mean gain on the exam rows |   mean certified bound |   mean long-run gain |   bound above the long-run gain |
|-------------------------------------+-------------------+-------------+------------------------------+------------------------+----------------------+---------------------------------|
| x0 in                               | 79 of 100         | 99 of 100   |                       0.1619 |                 0.0532 |               0.1599 |                               5 |
| x1, asked every time                | 0 of 100          | 40 of 100   |                       0.0478 |                -0.0316 |               0.0003 |                              11 |
| x1, asked when memory says mostly A | 0 of 41           | 37 of 41    |                       0.0545 |                -0.0247 |               0.0001 |                              11 |
+-------------------------------------+-------------------+-------------+------------------------------+------------------------+----------------------+---------------------------------+
```

The screen routed x0 to VAULT in 79 worlds and x1 FORWARD in all 100. Through the holdout, the certificates for x0 carried over to the long run: the mean gain on the exam rows, 0.162, matches the long-run 0.160, and the bound exceeded the long-run gain in 5 of 99, as a bound at 0.95 does. The certificates for x1 did not. The holdout accepted x1 in 40 worlds, mostly those whose holdout weeks favoured it, so the mean gain on their exam rows was 0.048 against a long-run gain of 0.000, and the bound exceeded the long-run gain in 11. An analyst who remembers the holdout weeks and submits x1 only when three or four of them were in state A submitted in 41 worlds and was certified in 37. That memory costs no bits in the holdout's log, which is why the screen has to be run before the choice reaches the holdout.

The bound behind the rule: if the weekly effects are independent and vary about their mean with spread tau_c, a choice steered by any information about the states of the holdout weeks biases the certificate for the long-run effect by at most tau_c times sqrt(2 I), I the information the choice carries about the states in nats, and by at most tau_c sqrt(2 ln r) when memory tells r states apart in each week, however many weeks the holdout covers. With tau_c = 0 the holdout's estimate does not depend on memory at all. The rule compares the upper bound on tau_c, not its estimate, with the holdout's noise, so a short history, whose bound is wide, routes forward. That is the conservative failure, and it is why x0 went FORWARD in 21 worlds of 100: a fine-grained choice sent forward costs time, not validity. A choice routed FORWARD is registered forward, its predictions issued before their outcomes, and a watch follows it, as in [C22](c22-drift-watch.md). [Certificates & the book](../manual/certificates.md) gives the reference for `route`.
