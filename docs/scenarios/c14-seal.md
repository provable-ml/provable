# C14 · Look at the holdout, and pay for it

*Synthetic world throughout, built by the first block below: 3,000 hourly rows, a target with a linear part and one interaction; the first 2,000 rows train and the last 1,000 are the holdout. The clock and the custody salt were fixed when this page was generated, so the holdout id, the certificate id, the model hash and the dates are specific to that run; every other number is reproduced by the code shown.*

A holdout earns its keep by staying unseen, and tuning wants to look at it. `pv.seal` makes the looking countable. The rows go into custody in the book, and salted hashes of their content, their row identities, the split and the protocol are committed before any answer exists. From then on the holdout is reached through its own methods only, every answer goes into a hash-chained log with its price in bits, and the certificate at the end is either priced by that log or taken on an exam slice that answered none of it.

The world: six candidate models, each fit on the 2,000 training rows only, and the lineage of both row sets.

```python
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import provable as pv
from provable import certificates as ct

rng = np.random.default_rng(14)
n = 3000
panel = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"x{i}" for i in range(8)])
panel["y"] = 0.8 * panel["x0"] - 0.5 * panel["x1"] + 0.6 * panel["x2"] * (panel["x3"] > 0) + rng.normal(size=n)
panel["start_utc"] = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
feats = [f"x{i}" for i in range(8)]
train, hold = panel.iloc[:2000], panel.iloc[2000:]


def fit(model, cols):
    model.fit(train[cols], train["y"])
    return lambda X: model.predict(X[cols])


candidates = [("ols x0", fit(LinearRegression(), ["x0"])),
              ("ols x0 x1", fit(LinearRegression(), ["x0", "x1"])),
              ("ols all", fit(LinearRegression(), feats)),
              ("gbm depth 2", fit(GradientBoostingRegressor(max_depth=2, random_state=0), feats)),
              ("gbm depth 3", fit(GradientBoostingRegressor(max_depth=3, random_state=0), feats)),
              ("gbm depth 5", fit(GradientBoostingRegressor(max_depth=5, random_state=0), feats))]
S_rows = ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=train.index))
H_rows = ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=hold.index))
```

```python
ho = pv.seal(hold[feats], hold["y"], "hold_book", lineage=H_rows,
             baseline=train["y"].mean(), baseline_lineage=S_rows, claim="gain")
print(ho)
for name, f in candidates:
    print(f"{name:12s}", ho.ask(f, lineage=S_rows, label=name))
```

```text
-- holdout HO-A926AA9A91DC073D --
+----------------------+------------------------------+
| holdout              | HO-A926AA9A91DC073D          |
| mode / route         | comparative / exam           |
| claim                | gain (squared loss)          |
| rows (tuning / exam) | 800 / 200                    |
| delta                | 0.05                         |
| code / eta           | kt / 1                       |
| asks (accepts)       | 0 (0)                        |
| bits spent           | 0.000                        |
| transcript bar z_L   | 1.6449                       |
| clean share s(L)     | 1.0000                       |
| retirement           | s(L) <= 0.05 (at 38.94 bits) |
| tuning slice         | open                         |
| exam slice           | sealed                       |
| certificates         | none                         |
+----------------------+------------------------------+
ols x0       ask 1 (log #1): ACCEPT | 1.000 bits; spent 1.00; z_L 1.960; s(L) 0.704; tuning open
ols x0 x1    ask 2 (log #2): ACCEPT | 0.415 bits; spent 1.42; z_L 2.080; s(L) 0.625; tuning open
ols all      ask 3 (log #3): ACCEPT | 0.263 bits; spent 1.68; z_L 2.154; s(L) 0.583; tuning open
gbm depth 2  ask 4 (log #4): ACCEPT | 0.193 bits; spent 1.87; z_L 2.207; s(L) 0.556; tuning open
gbm depth 3  ask 5 (log #5): REJECT | 3.322 bits; spent 5.19; z_L 2.996; s(L) 0.301; tuning open
gbm depth 5  ask 6 (log #6): REJECT | 2.000 bits; spent 7.19; z_L 3.396; s(L) 0.235; tuning open
```

The defaults are the protocol: level 0.05, the Krichevsky-Trofimov code, comparison at one paired standard error, an exam slice of 20 percent sealed apart, and retirement when the clean share s(L) falls to 0.05, which at this level is 38.94 bits. Each ask answers one question, whether this model beats the incumbent on the tuning slice by more than one paired standard error, and returns nothing else: no score, no difference, no p-value. An accepted model becomes the incumbent, which is the Ladder's rule. The answer is priced under the sealed code. The first answer costs one bit, accepts that follow accepts cost less, and the first reject after a run of accepts costs more (3.32 bits at ask 5). After 7.19 bits the bar a certificate on the tuning slice would need has moved from 1.645 to 3.396, the width a clean holdout gives with 23.5 percent of its rows: that fraction is s(L).

The gate runs before any answer. A model whose declared lineage touches a sealed row gets none:

```python
forty = ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=hold.index[:40]))
try:
    ho.ask(fit(LinearRegression(), feats), lineage=S_rows.union(forty))
except ValueError as e:
    print(e)
```

```text
ask refused: the model's lineage meets the sealed rows of HO-A926AA9A91DC073D: source 6fccec780a379a74: rows [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011] (+28 more)
```

Until retirement hands rows back, nothing else the holdout returns depends on the sealed values, and its refusals read only declarations, the public lineage and the public log, as this one does. A loss outside a declared range is clipped to it, a non-finite prediction falls back to the sealed baseline's for that row, arithmetic on sealed values ignores numpy's floating-point error settings, and the claim of a certificate names any such substitution, so a model built to provoke a refusal or an error gets an answer, priced like any other.

The certificate comes from `open`. With an exam slice sealed, it opens that slice once, at the clean bar; the transcript only decided which model to bring.

```python
best = candidates[3][1]
cert = ho.open(best, lineage=S_rows)
print(cert.claim)
try:
    ho.open(best, lineage=S_rows)
except ValueError as e:
    print(e)
print(ho.retire())
print(ho.audit())
print(pv.Ledger("hold_book").statement())
```

```text
performance: mean squared paired gain of model f2459d0b736d over the sealed baseline at least 0.63694 (FWER 0.05, CLT approximation; HO-A926AA9A91DC073D, exam slice, 7.19 bits spent)
open refused: the exam slice of HO-A926AA9A91DC073D was opened at #7 (CD-850EE205AAC003FE); a slice is opened once
retired HO-A926AA9A91DC073D: released 1000 rows (tuning, exam) after 7.19 bits; cause: declared
-- audit: HO-A926AA9A91DC073D --
+----------------+-----------+--------------------------------------------------------------------------------------+
| item           | verdict   | detail                                                                               |
|----------------+-----------+--------------------------------------------------------------------------------------|
| hash chain     | PASS      | 9 operations, head bc9f3025dbeb6f56                                                  |
| custody head   | PASS      | custody anchors #8; the log ends at #8                                               |
| dates          | PASS      | sealed 2026-05-07T09:00:01+00:00; every later operation dated after the seal         |
| holdout id     | PASS      | id = hash of the seal record                                                         |
| custody rows   | PASS      | x, y, split, baseline match the seal's salted commitments                            |
| row identities | PASS      | rows_sha 948916c521461102                                                            |
| seal state     | PASS      | the initial state the protocol implies                                               |
| #1 ask         | PASS      | accept at 1.0000 bits; spent 1.0000                                                  |
| #2 ask         | PASS      | accept at 0.4150 bits; spent 1.4150                                                  |
| #3 ask         | PASS      | accept at 0.2630 bits; spent 1.6781                                                  |
| #4 ask         | PASS      | accept at 0.1926 bits; spent 1.8707                                                  |
| #5 ask         | PASS      | reject at 3.3219 bits; spent 5.1926                                                  |
| #6 ask         | PASS      | reject at 2.0000 bits; spent 7.1926                                                  |
| #7 open        | PASS      | exam slice bound 0.63694 at bar 1.6449; certificate CD-850EE205AAC003FE              |
| #8 retire      | PASS      | declared                                                                             |
| book           | PASS      | 1 log certificate(s) active; no other active record names it or sits on custody rows |
+----------------+-----------+--------------------------------------------------------------------------------------+
audit PASS: 16 of 16 items recomputed and matched
-- ledger statement: hold_book --
+--------------------------------------------+------+
| budget (target)                            | 0.1  |
| charged (additive, active book)            | 0.05 |
| reserved (registered charges open)         | 0    |
| remaining                                  | 0.05 |
| active certificates                        | 1    |
| voided                                     | 0    |
| unknown-lineage items (not budget-bearing) | 0    |
+--------------------------------------------+------+
-- active certificates --
+---------------------+-------------+-------------+---------+--------------------------------------------------+
| id                  | kind        | guarantee   |   level | claim                                            |
|---------------------+-------------+-------------+---------+--------------------------------------------------|
| CD-850EE205AAC003FE | performance | FWER        |    0.05 | performance: mean squared paired gain of model f |
+---------------------+-------------+-------------+---------+--------------------------------------------------+
-- holdouts (bits = codelength of each holdout's own log; the transcript bar z_L is taken at delta 2^-bits; s(L) = (z_0/z_L)^2 is the clean share) --
+---------------------+------------------+------------------+---------------------------+--------+-------+--------+----------+--------+
| holdout             | mode/route       | tune/exam rows   | asks (accepts) | checks   |   bits |   z_L |   s(L) | tuning   | exam   |
|---------------------+------------------+------------------+---------------------------+--------+-------+--------+----------+--------|
| HO-A926AA9A91DC073D | comparative/exam | 800/200          | 6 (4)                     |   7.19 | 3.396 |  0.235 | retired  | opened |
+---------------------+------------------+------------------+---------------------------+--------+-------+--------+----------+--------+
```

The exam bound, a paired gain of at least 0.637 over the training mean at FWER 0.05 for the incumbent (gbm depth 2), holds however the six questions were chosen, because the exam rows answered none of them. The certificate is of kind `performance`, it charges the book once at the holdout's level, and it carries the tuning transcript as a peek: 7.19 bits on the 800 tuning rows, which [C17](c17-peeked.md) prices. Retirement releases rows from custody, here both slices because the exam is spent; before an exam is opened only the tuning slice leaves.

The audit is the replay anyone holding the book can run, under the book lock so that no write lands mid-replay. It checks the hash chain, the custody head that anchors the end of the log, the dates and the id first, and a log that fails them is not replayed at all. Then it checks the custody files against the seal's salted commitments, replays every operation through the state machine (every answer and its codelength from the stored predictions, every certificate rebuilt and matched to its record in the book, every logged state compared in full), and checks the book around the holdout: its certificates registered once and active, no other active record naming it, none registered on rows while they were in its custody, and none before the seal naming it or its rows. The book refuses a certificate or a refinement on rows in custody other than the holdout's own; the audit finds one written around the API. Passing the fitted models keyed by log number, `ho.audit(models={4: best})`, re-runs them once the replay has passed and matches the prediction hashes in the log; that answers questions about the rows, so it waits until every row has left custody, as here.

One reading for the next seal. At 7.19 bits the clean share is 0.235, above this seal's exam share of 0.2: sealed without an exam slice, the same transcript would have certified on all 1,000 rows more narrowly, which [C15](c15-transcript.md) does. `transcript.exam_wins(delta, bits, share)` is that comparison. The route is fixed at sealing, before the bits are known, so the comparison plans the next seal rather than amending this one.

Custody is the library's: the rows sit in the book directory under `holdouts/` as numpy and JSON files (no pickle), every row and prediction file is checked against its salted hash before it is parsed, and the API is the only door the library offers. A process that reads or writes those files directly is not stopped, and physical custody (a key held apart, a second person) is outside the library. What the log guarantees is that every answer the API gave is counted, whoever asked for it.
