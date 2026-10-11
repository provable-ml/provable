# C18 · From seal to reuse: the holdout end to end

*The usage specification of the holdout, then one study that runs every verb in order. Synthetic panel: 4,500 hourly rows, a binary outcome that depends on three of six features. The code blocks below, run in order, are the demo script, also shipped as [c18_holdout_demo.py](c18_holdout_demo.py). The clock and the custody salt were fixed when this page was generated, so the holdout and certificate ids, the model hashes and the dates are specific to that run; every other number is reproduced by the code shown.*

## The protocol

A holdout answers questions about rows nobody may look at, and it charges for every answer. The analyst's side of the arrangement is a fixed order of calls. Each call goes through the holdout's methods and is written to the book, so the analyst keeps only what the book cannot hold.

| step | call | before the call | what comes back | bits | the analyst keeps |
|---|---|---|---|---|---|
| plan | none | rows that no certificate of the book names | | | mode, exam share (the route), code, eta, delta, loss, claim |
| seal | `pv.seal(X, y, book, lineage=..., baseline=..., baseline_lineage=...)` | declared row identities; comparative mode needs a baseline and the rows it was fit on | a `Holdout`, its protocol committed to the book | none | the holdout id |
| tune, comparative | `ho.ask(model, lineage=..., label=...)` | tuning slice open; the model's lineage declared and clear of every row in custody | an `Answer`: ACCEPT or REJECT, nothing else | the answer's codelength under the sealed code; the total L sets the bar z_L and the clean share s(L) | the model, under its label |
| tune, numeric | `ho.check(model, guess, width, lineage=..., label=...)` | tuning slice open; width above `ho.min_width()` | an `Answer`: CONFIRMED (the guess) or FAILED (the value on a grid), with the width, or HALTED | answered at level delta 2^-L_i; the transcript's codelength grows | the model, under its label |
| certify | `ho.open(model, lineage=..., label=...)` | exam slice sealed, or, with none sealed, tuning slice open | a performance `Certificate`, registered in the book, one charge at delta | exam route: none, the clean level; transcript route: level delta 2^-L prices L | the certificate id |
| release | `ho.retire()` | none | a `Release`: the rows leaving custody and the bits they carry | none | the released rows, free for training and for guesses |
| read | `ho.transcript()` | none | the public log: log numbers, labels, answers, bits | | |
| verify | `ho.audit(models=...)` | for models: every row released | a `HoldoutAudit`, PASS or FAIL per item | | the verdict |
| reuse | `pv.reuse([cert], rows, ledger=book, level=q)` | none | FRESH, PEEKED, or NOT FRESH (refused), with inflation and pass level | PEEKED: inflation 2^L, pass level q 2^-L | the level of the next test on those rows |

Standing rules:

- The route is fixed at the seal. With an exam slice (`exam_share > 0`) the certificate is taken on it at the clean level delta; without one, on the tuning slice at level delta 2^-L, the bar z_L under the Gaussian tail. `transcript.exam_wins(delta, bits, share)` compares the two for the next seal under the Gaussian bar.
- The tail is fixed at the seal too. A loss with a range (a bounded loss such as Brier, or a declared `loss_range`) gets the exact betting bound; a loss without one gets the Gaussian bar, and the claim says "CLT approximation".
- One holdout per row, and the seal refuses rows that an active certificate of the book names.
- In comparative mode the tuning slice retires itself when s(L) falls to the sealed threshold (0.05 by default, 38.94 bits at delta 0.05); in numeric mode Guess and Check stops when a failure leaves no grid (HALTED); `retire()` ends either one sooner.
- A refusal reads only declarations, the public lineage and the public log, and a refused call costs nothing.
- Every call takes a label. `ho.transcript()` lists log numbers, labels, answers and bits, and `audit(models=...)` takes the models keyed by those log numbers.
- Outcome-derived lag columns are declared at the seal (`columns=`); after it, only those declarations are accepted.
- Models run on the sealed rows inside the holdout. What a model does with the rows it is given, the truth of declared lineage, and the closed channel are the analyst's undertaking.

What the analyst keeps outside the book is the book's directory, the holdout ids, and the models under the labels given at each call. Everything else (answers, bits, certificates, releases) is in the book.

## The study

Two blocks follow the 2,000 training rows: block A (1,000 rows) chooses a model, block B (the next 1,000) measures its error rate, and block C (the last 500) stays untouched until step 8.

### 1. The world and the candidates

Six candidate models, each fit on the training rows only. `keep` is the analyst's own record: every model under the label it will be submitted with.

```python
import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import provable as pv
from provable import certificates as ct
from provable.certificates import transcript as tr

os.chdir(tempfile.mkdtemp(prefix="holdout_demo_"))
rng = np.random.default_rng(20)
n = 4500
panel = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
signal = 0.9 * panel["x0"] - 0.6 * panel["x1"] + 0.8 * panel["x2"] * (panel["x3"] > 0)
panel["y"] = (signal + rng.normal(size=n) > 0).astype(float)
panel["start_utc"] = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
feats = [f"x{i}" for i in range(6)]
train, block_a, block_b, block_c = panel.iloc[:2000], panel.iloc[2000:3000], panel.iloc[3000:4000], panel.iloc[4000:]
rows = lambda frame: ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=frame.index))
S, A, B, C = rows(train), rows(block_a), rows(block_b), rows(block_c)


def fit(model, cols):
    model.fit(train[cols], train["y"])
    return lambda X: model.predict_proba(X[cols])[:, 1]


candidates = [("logit x0", fit(LogisticRegression(), ["x0"])),
              ("logit x0 x1", fit(LogisticRegression(), ["x0", "x1"])),
              ("logit all", fit(LogisticRegression(), feats)),
              ("gbm depth 2", fit(GradientBoostingClassifier(max_depth=2, random_state=0), feats)),
              ("gbm depth 3", fit(GradientBoostingClassifier(max_depth=3, random_state=0), feats)),
              ("gbm depth 5", fit(GradientBoostingClassifier(max_depth=5, random_state=0), feats))]
keep = dict(candidates)
```

### 2. Seal block A

```python
book = ct.Ledger("demo_book", budget=0.20)
ho_a = pv.seal(block_a[feats], block_a["y"], book, lineage=A, loss="brier",
               baseline=float(train["y"].mean()), baseline_lineage=S, label="block A: model choice")
print(ho_a)
```

```text
-- holdout HO-00E291C3F141ED9E --
+----------------------+-------------------------------------+
| holdout              | HO-00E291C3F141ED9E                 |
| mode / route         | comparative / exam                  |
| claim                | risk (brier loss, range [0.0, 1.0]) |
| rows (tuning / exam) | 800 / 200                           |
| delta                | 0.05                                |
| code / eta           | kt / 1                              |
| asks (accepts)       | 0 (0)                               |
| bits spent           | 0.000                               |
| transcript bar z_L   | 1.6449                              |
| clean share s(L)     | 1.0000                              |
| retirement           | s(L) <= 0.05 (at 38.94 bits)        |
| tuning slice         | open                                |
| exam slice           | sealed                              |
| certificates         | none                                |
+----------------------+-------------------------------------+
```

Comparative mode with the defaults: level 0.05, the Krichevsky-Trofimov code, comparison at one paired standard error, a 20 percent exam slice, and retirement at s(L) = 0.05, which is 38.94 bits at this level. The loss is the Brier score, so the range is [0, 1], and the claim is risk: the certificate will bound the chosen model's mean Brier loss from above. The training base rate is the baseline and the first incumbent. Nothing has been answered, so nothing has been spent. Record: the holdout id.

### 3. Tune on block A

```python
for name, model in candidates:
    print(f"{name:12s}", ho_a.ask(model, lineage=S, label=name))
try:
    ho_a.ask(keep["logit all"], lineage=S.union(rows(block_a.iloc[:25])), label="logit all, refit")
except ValueError as e:
    print(e)
print(ho_a.transcript())
```

```text
logit x0     ask 1 (log #1): ACCEPT | 1.000 bits; spent 1.00; z_L 1.960; s(L) 0.704; tuning open
logit x0 x1  ask 2 (log #2): ACCEPT | 0.415 bits; spent 1.42; z_L 2.080; s(L) 0.625; tuning open
logit all    ask 3 (log #3): ACCEPT | 0.263 bits; spent 1.68; z_L 2.154; s(L) 0.583; tuning open
gbm depth 2  ask 4 (log #4): ACCEPT | 0.193 bits; spent 1.87; z_L 2.207; s(L) 0.556; tuning open
gbm depth 3  ask 5 (log #5): REJECT | 3.322 bits; spent 5.19; z_L 2.996; s(L) 0.301; tuning open
gbm depth 5  ask 6 (log #6): REJECT | 2.000 bits; spent 7.19; z_L 3.396; s(L) 0.235; tuning open
ask refused: the model's lineage meets the sealed rows of HO-00E291C3F141ED9E: source d3c6190c61c41611: rows [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011] (+13 more)
-- transcript: HO-00E291C3F141ED9E --
+-----+----------+-----------------------+-----------------------------------------------+--------+-------------------+---------------+
|   # | action   | label                 | answer                                        |   bits |   transcript bits | certificate   |
|-----+----------+-----------------------+-----------------------------------------------+--------+-------------------+---------------|
|   0 | seal     | block A: model choice | comparative / exam: 800 tuning, 200 exam rows |        |             0     |               |
|   1 | ask      | logit x0              | ACCEPT                                        |  1     |             1     |               |
|   2 | ask      | logit x0 x1           | ACCEPT                                        |  0.415 |             1.415 |               |
|   3 | ask      | logit all             | ACCEPT                                        |  0.263 |             1.678 |               |
|   4 | ask      | gbm depth 2           | ACCEPT                                        |  0.193 |             1.871 |               |
|   5 | ask      | gbm depth 3           | REJECT                                        |  3.322 |             5.193 |               |
|   6 | ask      | gbm depth 5           | REJECT                                        |  2     |             7.193 |               |
+-----+----------+-----------------------+-----------------------------------------------+--------+-------------------+---------------+
```

Each ask answers one question, whether the model beats the incumbent on the 800 tuning rows by more than one paired standard error, and returns nothing else. Each answer is priced by its codelength under the sealed code: the first costs one bit, an accept after accepts costs less (0.415, 0.263, 0.193), and the first reject after four accepts costs more, 3.32 bits, then 2 bits for the next. The running total L = 7.19 bits moves the bar a certificate on the tuning slice would need from 1.645 to 3.396, so the clean share is s(L) = 0.235: those 800 rows now certify like 188 clean ones. The refused call declared a model fit on 25 sealed rows; it got no answer and cost nothing, and the log has no entry for it. Record: nothing new. The transcript is in the book and the models are in `keep`.

### 4. Certify on the exam slice

```python
L_a = ho_a.state["bits"]
print(f"{L_a:.2f} bits: z_L {tr.z_bar(0.05, L_a):.3f}, s(L) {tr.clean_share(0.05, L_a):.3f}; "
      f"an exam slice of 0.2 narrower than the transcript certificate: {tr.exam_wins(0.05, L_a, 0.2)}")
incumbent = next(r["label"] for r in ho_a.transcript().rows if r["seq"] == ho_a.state["incumbent"])
cert_a = ho_a.open(keep[incumbent], lineage=S, label=incumbent)
print(cert_a.claim)
print(f"{cert_a.id} carries the tuning transcript as a peek: {ct.peeks_of(cert_a)[0]['bits']:.2f} bits on the 800 tuning rows")
```

```text
7.19 bits: z_L 3.396, s(L) 0.235; an exam slice of 0.2 narrower than the transcript certificate: False
performance: mean brier loss of model ec9eb9153d2c at most 0.207749 (FWER 0.05, exact, betting bound; HO-00E291C3F141ED9E, exam slice, 7.19 bits spent, probabilities clipped to [0, 1])
CD-594E2974BAC64EA6 carries the tuning transcript as a peek: 7.19 bits on the 800 tuning rows
```

The route was fixed at the seal: the exam slice opens once at the clean level, with no bits priced, and the transcript only decided which model to bring, the incumbent at log #4. The first line plans the next seal rather than this one, at the Gaussian exchange rate: at 7.19 bits the clean share 0.235 is above the exam share 0.2, so under the Gaussian bar a holdout sealed without an exam slice would have certified on all 1,000 rows more narrowly, as [C15](c15-transcript.md) shows. The Brier loss lies in [0, 1], so the certificate takes the exact betting bound: the mean Brier loss is at most 0.208 at FWER 0.05. It charges the book once and carries the 7.19 bits on the tuning rows as a peek, which step 8 prices. Record: the certificate id.

### 5. Release block A

```python
release_a = ho_a.retire()
print(release_a)
print(ho_a.transcript())
```

```text
retired HO-00E291C3F141ED9E: released 1000 rows (tuning, exam) after 7.19 bits; cause: declared
-- transcript: HO-00E291C3F141ED9E --
+-----+----------+-----------------------+-----------------------------------------------+--------+-------------------+---------------------+
|   # | action   | label                 | answer                                        |   bits |   transcript bits | certificate         |
|-----+----------+-----------------------+-----------------------------------------------+--------+-------------------+---------------------|
|   0 | seal     | block A: model choice | comparative / exam: 800 tuning, 200 exam rows |        |             0     |                     |
|   1 | ask      | logit x0              | ACCEPT                                        |  1     |             1     |                     |
|   2 | ask      | logit x0 x1           | ACCEPT                                        |  0.415 |             1.415 |                     |
|   3 | ask      | logit all             | ACCEPT                                        |  0.263 |             1.678 |                     |
|   4 | ask      | gbm depth 2           | ACCEPT                                        |  0.193 |             1.871 |                     |
|   5 | ask      | gbm depth 3           | REJECT                                        |  3.322 |             5.193 |                     |
|   6 | ask      | gbm depth 5           | REJECT                                        |  2     |             7.193 |                     |
|   7 | open     | gbm depth 2           | exam slice opened: bound 0.207749             |  0     |             7.193 | CD-594E2974BAC64EA6 |
|   8 | retire   |                       | retired: declared                             |        |             7.193 |                     |
+-----+----------+-----------------------+-----------------------------------------------+--------+-------------------+---------------------+
```

Retirement ends the tuning slice. Both slices leave custody here because the exam is spent; before an exam is opened only the tuning rows leave. The open at #7 priced no bits (the clean level), and the retire at #8 changes nothing in the account. The released rows are free for training and for guesses, as in step 6, but they carry their bits into any later test that builds on the certificate.

### 6. Seal block B and ask for a number

```python
ho_b = pv.seal(block_b[feats], block_b["y"], book, lineage=B, loss="zero_one", mode="numeric",
               exam_share=0.25, label="block B: error rate")
print(ho_b)
keep["chosen"] = lambda X: (keep[incumbent](X) > 0.5).astype(float)
keep["runner-up"] = lambda X: (keep["logit all"](X) > 0.5).astype(float)
for name, offset in (("chosen", 0.0), ("chosen", 0.10), ("runner-up", 0.0)):
    guess = float((keep[name](release_a.X) != release_a.y).mean()) + offset
    width = 2 * ho_b.min_width()
    print(f"{name:9s} guess {guess:.4f} width {width:.4f} ->", ho_b.check(keep[name], guess, width, lineage=S, label=name))
print(ho_b.transcript())
```

```text
-- holdout HO-7C2BE3362769EE4C --
+-----------------------+----------------------------------------+
| holdout               | HO-7C2BE3362769EE4C                    |
| mode / route          | numeric / check                        |
| claim                 | risk (zero_one loss, range [0.0, 1.0]) |
| rows (tuning / exam)  | 750 / 250                              |
| delta                 | 0.05                                   |
| checks (failures)     | 0 (0)                                  |
| transcript bits       | 0.000                                  |
| next check priced at  | 1.436 bits (level 0.0185)              |
| min width, next check | 0.0558825                              |
| tuning slice          | open                                   |
| exam slice            | sealed                                 |
| certificates          | none                                   |
+-----------------------+----------------------------------------+
chosen    guess 0.2330 width 0.1118 -> check 1 (log #2): CONFIRMED 0.233 +/- 0.111765 = [0.121235, 0.344765] | level 0.0185 (1.44 bits); 0 failure(s); tuning open
chosen    guess 0.3330 width 0.1272 -> check 2 (log #3): FAILED 0.254466 +/- 0.127233 = [0.127233, 0.381699] | level 0.00462 (3.44 bits); 1 failure(s); tuning open
runner-up guess 0.2480 width 0.1773 -> check 3 (log #4): CONFIRMED 0.248 +/- 0.177345 = [0.0706554, 0.425345] | level 1.51e-05 (11.69 bits); 1 failure(s); tuning open
-- transcript: HO-7C2BE3362769EE4C --
+-----+----------+---------------------+--------------------------------------------+--------+-------------------+---------------------+
|   # | action   | label               | answer                                     |   bits |   transcript bits | certificate         |
|-----+----------+---------------------+--------------------------------------------+--------+-------------------+---------------------|
|   0 | seal     | block B: error rate | numeric / check: 750 tuning, 250 exam rows |        |             0     |                     |
|   1 | family   |                     | family certificate registered              |        |             0     | CD-A69FE3A2E8F8E910 |
|   2 | check    | chosen              | CONFIRMED 0.233 +/- 0.111765               |  1.436 |             3.436 |                     |
|   3 | check    | chosen              | FAILED 0.254466 +/- 0.127233               |  3.436 |            11.693 |                     |
|   4 | check    | runner-up           | CONFIRMED 0.248 +/- 0.177345               | 11.693 |            13.108 |                     |
+-----+----------+---------------------+--------------------------------------------+--------+-------------------+---------------------+
```

Numeric mode answers with a number: the error rate on block B of the chosen model, its probabilities now turned into labels at 0.5. Each guess comes from block A's released rows, which are the analyst's own data now, and each width is twice the next check's tolerance, `min_width()`, which the public transcript alone determines. The family certificate at #1 was registered before the first answer: every interval `check` releases on this holdout covers its population value, simultaneously, at FWER 0.05, for one charge. The bits column is L_i, and the check is answered at level 0.05 x 2^-L_i: 1.44 bits for the first, 3.44 for the second. The second guess was moved 0.10 on purpose. It failed and released the holdout's own value on a grid, and that value is information: the transcript's codelength rose from 3.44 to 11.69 bits, so the next check was priced at 11.69 bits and its interval had to be wider to confirm. Record: the released intervals, which the transcript keeps.

### 7. Certify on block B, release it, audit both holdouts

```python
cert_b = ho_b.open(keep["chosen"], lineage=S, label="chosen")
print(cert_b.claim)
release_b = ho_b.retire()
print(release_b)
for ho in (ho_a, ho_b):
    models = {r["seq"]: keep[r["label"]] for r in ho.transcript().rows if r["label"] in keep}
    print(ho.audit(models=models))
```

```text
performance: mean zero_one loss of model 79e72d5e1cf6 at most 0.289242 (FWER 0.05, exact, betting bound; HO-7C2BE3362769EE4C, exam slice, 13.11 bits spent)
retired HO-7C2BE3362769EE4C: released 1000 rows (tuning, exam) after 13.11 bits; cause: declared
-- audit: HO-00E291C3F141ED9E --
+----------------+-----------+--------------------------------------------------------------------------------------+
| item           | verdict   | detail                                                                               |
|----------------+-----------+--------------------------------------------------------------------------------------|
| hash chain     | PASS      | 9 operations, head 4654abc44350bbb7                                                  |
| custody head   | PASS      | custody anchors #8; the log ends at #8                                               |
| dates          | PASS      | sealed 2026-05-07T09:00:01+00:00; every later operation dated after the seal         |
| holdout id     | PASS      | id = hash of the seal record                                                         |
| custody rows   | PASS      | x, y, split, baseline match the seal's salted commitments                            |
| row identities | PASS      | rows_sha de6e60180bde724d                                                            |
| seal state     | PASS      | the initial state the protocol implies                                               |
| #1 ask         | PASS      | accept at 1.0000 bits; spent 1.0000                                                  |
| #2 ask         | PASS      | accept at 0.4150 bits; spent 1.4150                                                  |
| #3 ask         | PASS      | accept at 0.2630 bits; spent 1.6781                                                  |
| #4 ask         | PASS      | accept at 0.1926 bits; spent 1.8707                                                  |
| #5 ask         | PASS      | reject at 3.3219 bits; spent 5.1926                                                  |
| #6 ask         | PASS      | reject at 2.0000 bits; spent 7.1926                                                  |
| #7 open        | PASS      | exam slice bound 0.207749 at bar 0.0428; certificate CD-594E2974BAC64EA6             |
| #8 retire      | PASS      | declared                                                                             |
| book           | PASS      | 1 log certificate(s) active; no other active record names it or sits on custody rows |
| #1 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
| #2 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
| #3 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
| #4 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
| #5 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
| #6 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
| #7 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
+----------------+-----------+--------------------------------------------------------------------------------------+
audit PASS: 23 of 23 items recomputed and matched
-- audit: HO-7C2BE3362769EE4C --
+----------------+-----------+--------------------------------------------------------------------------------------+
| item           | verdict   | detail                                                                               |
|----------------+-----------+--------------------------------------------------------------------------------------|
| hash chain     | PASS      | 7 operations, head a9ddef76191aed1c                                                  |
| custody head   | PASS      | custody anchors #6; the log ends at #6                                               |
| dates          | PASS      | sealed 2026-05-07T09:00:11+00:00; every later operation dated after the seal         |
| holdout id     | PASS      | id = hash of the seal record                                                         |
| custody rows   | PASS      | x, y, split match the seal's salted commitments                                      |
| row identities | PASS      | rows_sha 7d74452111eb2add                                                            |
| seal state     | PASS      | the initial state the protocol implies                                               |
| #1 family      | PASS      | Guess and Check family certificate CD-A69FE3A2E8F8E910                               |
| #2 check       | PASS      | confirmed; level 0.01848                                                             |
| #3 check       | PASS      | failed; level 0.00462                                                                |
| #4 check       | PASS      | confirmed; level 1.51e-05                                                            |
| #5 open        | PASS      | exam slice bound 0.289242 at bar 0.0772; certificate CD-3ADC02A1D8FEBA4E             |
| #6 retire      | PASS      | declared                                                                             |
| book           | PASS      | 2 log certificate(s) active; no other active record names it or sits on custody rows |
| #2 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
| #3 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
| #4 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
| #5 model       | PASS      | re-run on the released rows, matched to the logged hash                              |
+----------------+-----------+--------------------------------------------------------------------------------------+
audit PASS: 18 of 18 items recomputed and matched
```

Block B's exam slice opens once at the clean level, with the exact betting bound since the zero-one loss lies in [0, 1], and carries the check transcript's 13.11 bits as a peek. Each audit replays its holdout from the log and the custody files and, with every row released, re-runs the models and matches their prediction hashes to the log. The audit takes models keyed by log number, and the transcript maps the labels kept in `keep` to those numbers.

### 8. Reuse: FRESH, PEEKED, NOT FRESH

```python
for name, rows_ in (("block C", C), ("A's tuning rows", ho_a.tuning_lineage()), ("A's exam rows", ho_a.exam_lineage())):
    try:
        v = pv.reuse([cert_a], rows_, ledger=book, level=0.05)
        print(f"{name:15s}: {v.verdict.label} | bits {v.bits:.2f} | inflation {v.inflation:.4g} | pass level {v.level:.4g}")
    except ct.FreshnessError as e:
        print(f"{name:15s}: REFUSED -- {str(e)[:80]} ...")
v = pv.reuse([cert_b], ho_b.tuning_lineage(), ledger=book, level=0.05)
print(f"B's tuning rows: {v.verdict.label} | bits {v.bits:.2f} | inflation {v.inflation:.4g} | pass level {v.level:.4g}")
print(f"A's 800 tuning rows after {L_a:.2f} bits certify like {800 * tr.clean_share(0.05, L_a):.0f} clean rows; "
      f"matching 800 clean rows takes {ct.rows_for_bits(800, L_a):,.0f}")
```

```text
block C        : FRESH | bits 0.00 | inflation 1 | pass level 0.05
A's tuning rows: PEEKED | bits 7.19 | inflation 146.3 | pass level 0.0003418
A's exam rows  : REFUSED -- refused: certificate lineage meets the consuming rows. [performance: mean brier  ...
B's tuning rows: PEEKED | bits 13.11 | inflation 8832 | pass level 5.661e-06
A's 800 tuning rows after 7.19 bits certify like 188 clean rows; matching 800 clean rows takes 3,410
```

Admission is the question a later analysis asks before it tests rows while building on a certificate. On block C, never touched, block A's certificate is FRESH. On block A's exam rows it is NOT FRESH and refused by default, since those are its own test rows. On block A's tuning rows it is PEEKED: no row of its test lineage is there, but its model was chosen with 7.19 bits of answers from those rows. A test on them that consumes the certificate is admitted at inflation 2^7.19 = 146.3 and must run at 0.05 x 2^-7.19 = 0.00034. Block B's certificate on block B's tuning rows carries 13.11 bits, so a test there runs at 5.7e-06: the failed check made those rows expensive. In rows, block A's 800 tuning rows now certify like 188 clean ones, and `rows_for_bits` gives the 3,410 rows that would match 800 clean ones.

### 9. The book and the record

```python
print(book.statement())
record = [["book", book.path, "every answer, certificate and release, in log order"],
          ["holdout A", ho_a.id, f"comparative, exam route; {L_a:.2f} bits; retired"],
          ["certificate A", cert_a.id, f"mean Brier loss at most {cert_a.artifact['bound']:.4f}, FWER 0.05; peek {L_a:.2f} bits"],
          ["holdout B", ho_b.id, f"numeric; transcript {ho_b.state['bits']:.2f} bits; retired"],
          ["family B", ho_b.state["family"], "every interval check released covers its value, FWER 0.05"],
          ["certificate B", cert_b.id, f"error rate at most {cert_b.artifact['bound']:.4f}, FWER 0.05; peek {ho_b.state['bits']:.2f} bits"],
          ["models", ", ".join(keep), "under the labels given at each call; audit takes them by log number"]]
print(tabulate(record, headers=["kept", "value", "what it is"], tablefmt="psql"))
```

```text
-- ledger statement: demo_book --
+--------------------------------------------+------+
| budget (target)                            | 0.2  |
| charged (additive, active book)            | 0.15 |
| reserved (registered charges open)         | 0    |
| remaining                                  | 0.05 |
| active certificates                        | 3    |
| voided                                     | 0    |
| unknown-lineage items (not budget-bearing) | 0    |
+--------------------------------------------+------+
-- active certificates --
+---------------------+-------------+-------------+---------+--------------------------------------------------+
| id                  | kind        | guarantee   |   level | claim                                            |
|---------------------+-------------+-------------+---------+--------------------------------------------------|
| CD-594E2974BAC64EA6 | performance | FWER        |    0.05 | performance: mean brier loss of model ec9eb9153d |
| CD-A69FE3A2E8F8E910 | performance | FWER        |    0.05 | performance: every answer check releases on HO-7 |
| CD-3ADC02A1D8FEBA4E | performance | FWER        |    0.05 | performance: mean zero_one loss of model 79e72d5 |
+---------------------+-------------+-------------+---------+--------------------------------------------------+
-- holdouts (bits = codelength of each holdout's own log; the transcript bar z_L is taken at delta 2^-bits; s(L) = (z_0/z_L)^2 is the clean share) --
+---------------------+------------------+------------------+---------------------------+--------+-------+--------+----------+--------+
| holdout             | mode/route       | tune/exam rows   | asks (accepts) | checks   |   bits | z_L   | s(L)   | tuning   | exam   |
|---------------------+------------------+------------------+---------------------------+--------+-------+--------+----------+--------|
| HO-00E291C3F141ED9E | comparative/exam | 800/200          | 6 (4)                     |   7.19 | 3.396 | 0.235  | retired  | opened |
| HO-7C2BE3362769EE4C | numeric/check    | 750/250          | 3 (1 failed)              |  13.11 | -     | -      | retired  | opened |
+---------------------+------------------+------------------+---------------------------+--------+-------+--------+----------+--------+
+---------------+--------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| kept          | value                                                                                      | what it is                                                          |
|---------------+--------------------------------------------------------------------------------------------+---------------------------------------------------------------------|
| book          | demo_book                                                                                  | every answer, certificate and release, in log order                 |
| holdout A     | HO-00E291C3F141ED9E                                                                        | comparative, exam route; 7.19 bits; retired                         |
| certificate A | CD-594E2974BAC64EA6                                                                        | mean Brier loss at most 0.2077, FWER 0.05; peek 7.19 bits           |
| holdout B     | HO-7C2BE3362769EE4C                                                                        | numeric; transcript 13.11 bits; retired                             |
| family B      | CD-A69FE3A2E8F8E910                                                                        | every interval check released covers its value, FWER 0.05           |
| certificate B | CD-3ADC02A1D8FEBA4E                                                                        | error rate at most 0.2892, FWER 0.05; peek 13.11 bits               |
| models        | logit x0, logit x0 x1, logit all, gbm depth 2, gbm depth 3, gbm depth 5, chosen, runner-up | under the labels given at each call; audit takes them by log number |
+---------------+--------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
```

The book holds three charges at 0.05 (block A's certificate, block B's family and block B's certificate) against the budget of 0.20 set when it was opened, and both holdouts with their bits. The second table is what the analyst wrote down, and all of it except the models can be read back from the book.

Each step has a page of its own: [C14](c14-seal.md) for the seal, the asks and the audit, [C15](c15-transcript.md) for the transcript certificate and the adversary that shows why the price is needed, [C16](c16-check.md) for Guess and Check, [C17](c17-peeked.md) for PEEKED, lint R7 and the rates register, and [Certificates & the book](../manual/certificates.md) for the reference. [C19](c19-changed-world.md) and [C20](c20-few-rows.md) set looking against not looking on worlds where looking gains, and [C21](c21-looking-hurts.md) on one where it hurts. [C22](c22-drift-watch.md) carries a certificate past a refit under drift and watches the deployed rows, and [C23](c23-transport-screen.md) screens a choice on the training rows before it reaches the holdout.
