# C22 · When the world drifts: certify, refit, watch

*Synthetic world throughout, built by the first block below. The code blocks below, run in order, are the demo script, also shipped as [c22_drift_watch_demo.py](c22_drift_watch_demo.py); its last block repeats the study over 120 worlds, and the whole script runs in about ten seconds. The clock and the custody salt were fixed when this page was generated, so the holdout, certificate and watch ids, the model hashes and the dates are specific to that run; every other number is reproduced by the code shown.*

In this world the level of the target wanders as a random walk, so the latest rows say the most about the rows to come. The analyst predicts with the mean of the last 35 rows, holds out the latest 200 rows of history, and certifies the model fitted before them. Then they refit through the holdout, deploy the refit model, and carry the certificate's bound to it. Under drift the bound describes the refit model only for a while: on average its live risk passes the certified model's holdout score at lead (h + 1)/2 rows, here 100.5. `pv.watch` follows the deployed rows against the certified level and raises an alarm once the evidence that the level is breached reaches 1/delta.

## In short

| | |
|---|---|
| data | 3,500 hourly rows: y = level + noise, noise with standard deviation 1; the level is a random walk whose steps have standard deviation 0.05 per row |
| model | the mean of the last 35 rows, close to the best window for this drift, the square root of 3 times sigma / tau, 34.6 rows |
| split | 1,800 training rows, then the latest 200 rows of history as the holdout, then 1,500 deployed rows |
| certificate | the model fitted on the training rows, on all 200 holdout rows at the clean level, squared error clipped to [0, 16], so the bound is exact (the betting bound): mean loss at most 1.664 |
| after a refit | over 20,000 simulated worlds, the refit model's live risk at lead j minus the certified model's holdout score is tau^2 (j - (h + 1)/2) to within 0.0002 at every lead shown: the score is a conservative report for the refit model up to lead 100, and not after |
| the watch | on the refit model's deployed rows, fed one day at a time: its live risk passes the certified level at lead 55, and the watch raises its alarm at lead 464 |
| over 60 worlds | the live risk passed the certified level in all 60, at a median lead of 138; the watch never alarmed before that lead, and it alarmed in 49 worlds (53 with a horizon of 1,500 rows) at a median delay of 376 rows (271); without drift the bound held in all 60 and no watch raised an alarm |

## The study

### 1. The world and the model

`world(rep)` draws world `rep`: the level, a random walk, and 3,500 hourly rows of y around it. The model is a constant, the mean of the last 35 rows it was fitted on: `f_S` is fitted on the training rows, which end where the holdout begins, and `f_n` is refitted through the last holdout row. `clipped_risk` is the live risk of a row on the certificate's range, the mean of min(r^2, 16) for a residual r with the given mean and variance; given the level at the row before, a deployed row's residual has mean the distance from that level to the prediction and variance tau^2 + sigma^2.

```python
import os, tempfile
import numpy as np, pandas as pd
from scipy.stats import norm
from tabulate import tabulate
import provable as pv
from provable import certificates as ct

os.chdir(tempfile.mkdtemp(prefix="drift_"))
sigma, tau, w = 1.0, 0.05, 35
n_hist, h, J = 2000, 200, 1500
m = n_hist - h
RANGE = (0.0, 16.0)


def world(rep, tau=tau):
    rng = np.random.default_rng(22000 + rep)
    level = np.cumsum(rng.normal(scale=tau, size=n_hist + J))
    panel = pd.DataFrame({"row": np.arange(n_hist + J), "y": level + rng.normal(scale=sigma, size=n_hist + J)})
    panel["start_utc"] = pd.date_range("2026-01-01", periods=n_hist + J, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    return level, panel


def clipped_risk(gap, var, c=RANGE[1]):
    """E min(r^2, c) for r ~ N(gap, var): the live risk of a row, on the certificate's range."""
    s, a = np.sqrt(var), np.sqrt(c)
    lo, hi = (-a - gap) / s, (a - gap) / s
    inside = norm.cdf(hi) - norm.cdf(lo)
    return (gap ** 2 + var) * inside + 2 * gap * s * (norm.pdf(lo) - norm.pdf(hi)) + var * (lo * norm.pdf(lo) - hi * norm.pdf(hi)) + c * (1 - inside)


level, panel = world(0)
train, hold, live = panel.iloc[:m], panel.iloc[m:n_hist], panel.iloc[n_hist:]
S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "drift", rows=f.index)) for f in (train, hold))
f_S = float(train["y"].iloc[-w:].mean())
f_n = float(panel["y"].iloc[n_hist - w:n_hist].mean())
print(f"trailing mean of the last {w} rows: {f_S:.4f} at the end of training, {f_n:.4f} refit through the holdout")
```

```text
trailing mean of the last 35 rows: 0.8592 at the end of training, 1.3710 refit through the holdout
```

### 2. Certify on the latest rows

```python
ho = pv.seal(hold[["row"]], hold["y"], ct.Ledger("drift_book", budget=0.10), lineage=H, baseline=float(train["y"].mean()),
             baseline_lineage=S, exam_share=0.0, loss_range=RANGE, clock=(hold["start_utc"], hold["end_utc"]))
cert = ho.open(lambda Z: np.full(len(Z), f_S), lineage=S, label="trailing mean")
U = cert.artifact["bound"]
print(cert.claim)
```

```text
performance: mean squared loss of model 542882bef296 at most 1.66364 (FWER 0.05, exact, betting bound; HO-DE4484E29145B232, tuning slice, 0.00 bits spent, loss clipped to [0, 16])
```

The holdout is the latest 200 rows of history, sealed without an exam slice and opened once, with no question asked: 0 bits, so the level is the clean one and the certificate uses all 200 rows. The loss range [0, 16] is declared at the seal, because the watch needs bounded losses and certificate and watch must speak of the same loss; a squared error above 16 is a residual of four standard deviations. With a range declared, the certificate takes the exact betting bound. It is about `f_S` on the law of the holdout rows: its mean squared loss there is at most 1.664 at FWER 0.05.

### 3. What the certificate says about the refit model

```python
def leads_table(reps=20000, leads=(1, 25, 50, 75, 100, 101, 125, 150, 200, 300)):
    rng = np.random.default_rng(22)
    steps = rng.normal(scale=tau, size=(reps, w + h))
    lev = np.cumsum(steps, axis=1)
    y = lev + rng.normal(scale=sigma, size=(reps, w + h))
    fs = y[:, :w].mean(axis=1)
    fn = y[:, h:].mean(axis=1)
    score = ((y[:, w:] - fs[:, None]) ** 2).mean(axis=1)
    rows = []
    for j in leads:
        live = (lev[:, -1] - fn) ** 2 + j * tau ** 2 + sigma ** 2
        rows.append([j, live.mean(), score.mean(), live.mean() - score.mean(), tau ** 2 * (j - (h + 1) / 2)])
    return rows


print(tabulate(leads_table(), headers=["lead j", "live risk of the refit model", "holdout score of f_S", "difference", "tau^2 (j - (h+1)/2)"], tablefmt="psql", floatfmt=".4f"))
```

```text
+----------+--------------------------------+------------------------+--------------+-----------------------+
|   lead j |   live risk of the refit model |   holdout score of f_S |   difference |   tau^2 (j - (h+1)/2) |
|----------+--------------------------------+------------------------+--------------+-----------------------|
|        1 |                         1.0587 |                 1.3077 |      -0.2489 |               -0.2488 |
|       25 |                         1.1187 |                 1.3077 |      -0.1889 |               -0.1888 |
|       50 |                         1.1812 |                 1.3077 |      -0.1264 |               -0.1263 |
|       75 |                         1.2437 |                 1.3077 |      -0.0639 |               -0.0638 |
|      100 |                         1.3062 |                 1.3077 |      -0.0014 |               -0.0013 |
|      101 |                         1.3087 |                 1.3077 |       0.0011 |                0.0013 |
|      125 |                         1.3687 |                 1.3077 |       0.0611 |                0.0613 |
|      150 |                         1.4312 |                 1.3077 |       0.1236 |                0.1238 |
|      200 |                         1.5562 |                 1.3077 |       0.2486 |                0.2488 |
|      300 |                         1.8062 |                 1.3077 |       0.4986 |                0.4988 |
+----------+--------------------------------+------------------------+--------------+-----------------------+
```

Under this local-level drift the expected risks are exact. The certified model's holdout score averages sigma^2 + sigma^2/w + tau^2 ((h + 1)/2 + c_w), where (h + 1)/2 is the mean distance of the holdout rows from the last training row and c_w = (w - 1)(2w - 1)/(6w) is the staleness of the window itself. The refit model's live risk at lead j averages sigma^2 + sigma^2/w + tau^2 (j + c_w). The difference is tau^2 (j - (h + 1)/2) whatever w and the length of the history, and the 20,000 simulated worlds above reproduce it to within 0.0002 at every lead. So the holdout score of the model fitted before the holdout is a conservative report for the refit model up to lead (h + 1)/2 = 100.5, and an optimistic one after it. Two practices follow: certify again on a holdout rolled forward at least every h/2 rows, or add the allowance tau^2 (j - (h + 1)/2) to the report. Both need tau. The watch needs neither: it reads the deployed rows.

### 4. Watch the deployed rows

```python
wa = pv.watch(cert, "drift_book")
lead = 0
while lead < J:
    day = live.iloc[lead:lead + 24]
    st = wa.update(day["y"], np.full(len(day), f_n), times=day["start_utc"])
    lead += len(day)
    if st.status != "watching":
        break
risk = clipped_risk(level[n_hist - 1:n_hist + J - 1] - f_n, tau ** 2 + sigma ** 2)
cross = int(np.argmax(risk > U)) + 1
print(st)
print(f"the refit model's live risk first exceeds the certified level {U:.4f} at lead {cross}; the watch alarmed at lead {st.alarm['row']}")
a0, a1 = st.alarm["start"], st.alarm["row"]
gap = risk - U
print(f"leads {cross} to {a0 - 1}: live risk above the certified level at {(gap[cross - 1:a0 - 1] > 0).mean():.0%} of leads, by {gap[cross - 1:a0 - 1].mean():.2f} on average; leads {a0} to {a1}: by {gap[a0 - 1:a1].mean():.2f}")
print(pv.Ledger("drift_book").statement())
```

```text
-- watch WA-C6FFA247473EB287 --
+-----------------------------+--------------------------------------------------------------------------------------------+
| watch                       | WA-C6FFA247473EB287                                                                        |
| certificate                 | CD-A522E42BCBF09466                                                                        |
| claim                       | mean loss at most 1.66364 (squared loss)                                                   |
| rows monitored              | 464 of 480 received                                                                        |
| mean loss                   | 1.91144                                                                                    |
| evidence (alarm at 1/delta) | 27.26 of 20                                                                                |
| status                      | alarm                                                                                      |
| alarm                       | row 464 (2026-04-13T15:00:00+00:00); the evidence points to a breach starting near row 421 |
| last row's time             | 2026-04-13T15:00:00+00:00                                                                  |
+-----------------------------+--------------------------------------------------------------------------------------------+
the refit model's live risk first exceeds the certified level 1.6636 at lead 55; the watch alarmed at lead 464
leads 55 to 420: live risk above the certified level at 65% of leads, by 0.08 on average; leads 421 to 464: by 1.74
-- ledger statement: drift_book --
+--------------------------------------------+-----+
| budget (target)                            | 0.1 |
| charged (additive, active book)            | 0.1 |
| reserved (registered charges open)         | 0   |
| remaining                                  | 0   |
| active certificates                        | 1   |
| voided                                     | 0   |
| unknown-lineage items (not budget-bearing) | 0   |
+--------------------------------------------+-----+
-- active certificates --
+---------------------+-------------+-------------+---------+--------------------------------------------------+
| id                  | kind        | guarantee   |   level | claim                                            |
|---------------------+-------------+-------------+---------+--------------------------------------------------|
| CD-A522E42BCBF09466 | performance | FWER        |    0.05 | performance: mean squared loss of model 542882be |
+---------------------+-------------+-------------+---------+--------------------------------------------------+
-- holdouts (bits = codelength of each holdout's own log; the transcript bar z_L is taken at delta 2^-bits; s(L) = (z_0/z_L)^2 is the clean share) --
+---------------------+-------------------+------------------+---------------------------+--------+-------+--------+-----------+--------+
| holdout             | mode/route        | tune/exam rows   | asks (accepts) | checks   |   bits |   z_L |   s(L) | tuning    | exam   |
|---------------------+-------------------+------------------+---------------------------+--------+-------+--------+-----------+--------|
| HO-DE4484E29145B232 | comparative/kraft | 200/0            | 0 (0)                     |      0 | 1.645 |      1 | certified | none   |
+---------------------+-------------------+------------------+---------------------------+--------+-------+--------+-----------+--------+
-- watches (budget-bearing: each watch charges its level when it opens; evidence against the certified level, alarm at 1/level) --
+---------------------+---------------------+---------+---------------+--------+--------+------------+------------------+
| watch               | certificate         |   level | certified     |   rows |   mean |   evidence | status           |
|---------------------+---------------------+---------+---------------+--------+--------+------------+------------------|
| WA-C6FFA247473EB287 | CD-A522E42BCBF09466 |    0.05 | risk <= 1.664 |    464 |  1.911 |      27.26 | alarm at row 464 |
+---------------------+---------------------+---------+---------------+--------+--------+------------+------------------+
```

The watch binds to the certificate and charges the book its level, 0.05, before any deployed row is seen, so the book stands at 0.10 of its 0.10. Each day's 24 rows go in with their start times; the first may not start before the certified rows end, and later rows may not go back in time. Every row moves the evidence, a mixture over the row at which a breach may have started and over a grid of bets of products prod (1 + lambda (u_r - m)), with u the loss scaled to [0, 1] and m the certified level on that scale. While every row's mean loss is at most the certified level, the evidence is a nonnegative supermartingale that starts at 1, so by Ville's inequality it ever reaches 1/delta = 20 with probability at most 0.05, however long the watch runs. Here the refit model's live risk first passed the certified level at lead 55, computed from the level the world hides. The evidence reached 27.3 at lead 464, the watch raised its alarm, and the CUSUM argmax points to a breach starting near lead 421. From lead 55 the live risk hovered around the certified level, above it at 65 percent of leads and by 0.08 on average; from lead 421 to the alarm it was above by 1.74 on average, and the argmax marks where that large breach began, not the first crossing. The last 16 rows of that day are stored but were not monitored, and the watch takes no more rows: a new stretch of deployment is a new certificate and a new watch. The watch monitors the predictions it is given against the certified level. Here they come from the refit model, so the alarm says when the report carried to it stopped holding.

### 5. Sixty worlds

```python
def study(rep, tau=tau):
    """One world: certify f_S on the latest h rows, deploy the refit f_n, and watch it twice, without a horizon and with a horizon of J rows."""
    level, panel = world(rep, tau)
    train, hold, live = panel.iloc[:m], panel.iloc[m:n_hist], panel.iloc[n_hist:]
    S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "drift", rows=f.index)) for f in (train, hold))
    f_S, f_n = float(train["y"].iloc[-w:].mean()), float(panel["y"].iloc[n_hist - w:n_hist].mean())
    book = ct.Ledger(tempfile.mkdtemp(dir="."), budget=0.15)
    ho = pv.seal(hold[["row"]], hold["y"], book, lineage=H, baseline=float(train["y"].mean()), baseline_lineage=S,
                 exam_share=0.0, loss_range=RANGE, clock=(hold["start_utc"], hold["end_utc"]), seed=rep)
    cert = ho.open(lambda Z: np.full(len(Z), f_S), lineage=S)
    U = cert.artifact["bound"]
    risk = clipped_risk(level[n_hist - 1:n_hist + J - 1] - f_n, tau ** 2 + sigma ** 2)
    out = {"held": bool(risk[0] <= U), "cross": int(np.argmax(risk > U)) + 1 if (risk > U).any() else None}
    for name, horizon in (("anytime", None), ("horizon", J)):
        wa = pv.watch(cert, book, horizon=horizon)
        for k in range(0, J, 168):
            week = live.iloc[k:k + 168]
            st = wa.update(week["y"], np.full(len(week), f_n), times=week["start_utc"])
            if st.status != "watching":
                break
        out[name] = st.alarm["row"] if st.alarm else None
    return out


rows = []
for label, t in (("drift, tau 0.05", tau), ("no drift", 0.0)):
    r = pd.DataFrame([study(rep, t) for rep in range(60)])
    for name in ("anytime", "horizon"):
        a = r[name]
        early = int((a.notna() & (r["cross"].isna() | (a < r["cross"]))).sum())
        delay = (a - r["cross"]).dropna()
        rows.append([label, "none" if name == "anytime" else J, int(r["held"].sum()), int(r["cross"].notna().sum()), r["cross"].median() if r["cross"].notna().any() else None,
                     int(a.notna().sum()), early, a.median() if a.notna().any() else None, delay.median() if len(delay) else None])
print(tabulate(rows, headers=["world, 60 each", "horizon", "bound held at lead 1", "risk exceeded the bound", "median lead it did", "alarms", "alarms before it did", "median alarm lead", "median delay"], tablefmt="psql", floatfmt=".0f"))
```

```text
+------------------+-----------+------------------------+---------------------------+----------------------+----------+------------------------+---------------------+----------------+
| world, 60 each   | horizon   |   bound held at lead 1 |   risk exceeded the bound |   median lead it did |   alarms |   alarms before it did |   median alarm lead |   median delay |
|------------------+-----------+------------------------+---------------------------+----------------------+----------+------------------------+---------------------+----------------|
| drift, tau 0.05  | none      |                     60 |                        60 |                  138 |       49 |                      0 |                 608 |            376 |
| drift, tau 0.05  | 1500      |                     60 |                        60 |                  138 |       53 |                      0 |                 560 |            271 |
| no drift         | none      |                     60 |                         0 |                      |        0 |                      0 |                     |                |
| no drift         | 1500      |                     60 |                         0 |                      |        0 |                      0 |                     |                |
+------------------+-----------+------------------------+---------------------------+----------------------+----------+------------------------+---------------------+----------------+
```

With drift, the refit model's live risk passed the certified level in all 60 worlds, at a median lead of 138. No watch alarmed before that lead in any world. That is the guarantee at work: an alarm before the live risk first passes the certified level is an error, and it happens with probability at most 0.05. Without a horizon the watch alarmed in 49 worlds within the 1,500 deployed rows, at a median of 376 rows after the crossing. With a horizon of 1,500 rows it alarmed in 53, at a median delay of 271. The horizon spreads the prior over the starts it covers, log T in place of 2 log s for a breach starting at row s, so a late breach is caught sooner, but the guarantee ends with the horizon. The delay is long because the breach grows slowly, by tau^2 = 0.0025 per row on average, and a breach of a few tenths in squared error, against a per-row standard deviation of the loss near 2, takes hundreds of rows to show. Without drift the refit model's live risk was within the certified level at lead 1 in all 60 worlds, and no watch raised an alarm.

The watch does not void the certificate: the bound remains a true statement about the law the holdout sampled. What it adds is a forward record with a guarantee that needs no assumption on how the world moves: no alarm while the certified level holds, except with probability delta. How soon an alarm follows a breach depends on the size of the breach and is measured here, not promised. [C18](c18-end-to-end.md) runs the holdout protocol end to end, and [Certificates & the book](../manual/certificates.md) gives the reference for `watch`.
