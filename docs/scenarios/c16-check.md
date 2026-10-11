# C16 · Ask for a number, not a yes

*Synthetic world, built by the first block below: 4,000 rows, a binary target, logistic models fit on the first 2,000, each with its 5-fold cross-validated error on those rows as the guess. The clock and the custody salt were fixed when this page was generated, so the ids, the model hash and the dates are specific to that run; every other number is reproduced by the code shown.*

Sometimes the question is a number: what is this model's error rate? A holdout sealed with `mode="numeric"` answers `check` by Guess and Check (Rogers, Roth, Smith, Srebro, Thakkar and Woodworth, AISTATS 2020, Algorithm 1, with the paper's Hoeffding tolerance). The analyst brings a guess and a width. If the guess lies within width - tol of the holdout's value it is CONFIRMED and released with the width, and nothing beyond the guess is revealed. Otherwise the check FAILS and releases the holdout's value on a grid of step width - tol, with the width. A failure that leaves no grid HALTS the mechanism. Every released interval covers its population value, simultaneously, with probability 1 - delta. The statistic must have a declared range; the zero-one loss has one.

```python
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import provable as pv
from provable import certificates as ct

rng = np.random.default_rng(16)
n = 4000
panel = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
panel["y"] = (0.9 * panel["x0"] - 0.6 * panel["x1"] + 0.4 * panel["x2"] + rng.normal(size=n) > 0).astype(float)
panel["start_utc"] = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
feats = [f"x{i}" for i in range(6)]
train, hold = panel.iloc[:2000], panel.iloc[2000:]
S_rows = ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=train.index))
H_rows = ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=hold.index))


def fit(cols, C=1.0):
    m = LogisticRegression(C=C).fit(train[cols], train["y"])
    err = 1 - cross_val_score(LogisticRegression(C=C), train[cols], train["y"], cv=5).mean()
    return (lambda X: m.predict(X[cols])), float(err)
```

```python
ho = pv.seal(hold[feats], hold["y"], "check_book", lineage=H_rows, loss="zero_one", mode="numeric", exam_share=0.25)
print(f"min width for the first check: {ho.min_width():.4f}")
for cols, C, shift in ((["x0"], 1.0, 0.0), (["x0", "x1", "x2"], 1.0, 0.0), (["x0", "x1", "x2"], 1.0, 0.15), (feats, 1.0, 0.0), (feats, 0.01, 0.0)):
    f, guess = fit(cols, C)
    w = 2.0 * ho.min_width()
    print(f"{'+'.join(cols):18s} C={C:<5g} guess {guess + shift:.4f} width {w:.4f} ->", ho.check(f, guess + shift, w, lineage=S_rows))
f, guess = fit(["x0", "x1", "x2"])
print(ho.check(f, guess - 0.10, ho.min_width(), lineage=S_rows))
try:
    ho.check(f, guess, 0.05, lineage=S_rows)
except ValueError as e:
    print(e)
cert = ho.open(f, lineage=S_rows)
print(cert.claim)
print(pv.Ledger("check_book").statement())
```

```text
min width for the first check: 0.0395
x0                 C=1     guess 0.3145 width 0.0790 -> check 1 (log #2): CONFIRMED 0.3145 +/- 0.0790298 = [0.23547, 0.39353] | level 0.0185 (1.44 bits); 0 failure(s); tuning open
x0+x1+x2           C=1     guess 0.2375 width 0.0900 -> check 2 (log #3): CONFIRMED 0.2375 +/- 0.0899672 = [0.147533, 0.327467] | level 0.00462 (3.44 bits); 0 failure(s); tuning open
x0+x1+x2           C=1     guess 0.3875 width 0.0958 -> check 3 (log #4): FAILED 0.191576 +/- 0.095788 = [0.095788, 0.287364] | level 0.00205 (4.61 bits); 1 failure(s); tuning open
x0+x1+x2+x3+x4+x5  C=1     guess 0.2380 width 0.1318 -> check 4 (log #5): CONFIRMED 0.238 +/- 0.131822 = [0.106178, 0.369822] | level 4.37e-06 (13.48 bits); 1 failure(s); tuning open
x0+x1+x2+x3+x4+x5  C=0.01  guess 0.2370 width 0.1355 -> check 5 (log #6): CONFIRMED 0.237 +/- 0.135483 = [0.101517, 0.372483] | level 2.1e-06 (14.54 bits); 1 failure(s); tuning open
check 6 (log #7): HALTED, the width leaves no grid and the mechanism stops | level 1.17e-06; 2 failure(s); tuning halted
check refused: Guess and Check on HO-91334011A8E5D868 halted at #7; the mechanism stops after a failure leaves no grid
performance: mean zero_one loss of model 022207ab47fc at most 0.257972 (FWER 0.05, exact, betting bound; HO-91334011A8E5D868, exam slice, 16.10 bits spent)
-- ledger statement: check_book --
+--------------------------------------------+-----+
| budget (target)                            | 0.1 |
| charged (additive, active book)            | 0.1 |
| reserved (registered charges open)         | 0   |
| remaining                                  | 0   |
| active certificates                        | 2   |
| voided                                     | 0   |
| unknown-lineage items (not budget-bearing) | 0   |
+--------------------------------------------+-----+
-- active certificates --
+---------------------+-------------+-------------+---------+--------------------------------------------------+
| id                  | kind        | guarantee   |   level | claim                                            |
|---------------------+-------------+-------------+---------+--------------------------------------------------|
| CD-75A5422395AF1D84 | performance | FWER        |    0.05 | performance: every answer check releases on HO-9 |
| CD-6AAB87BDE394DD7A | performance | FWER        |    0.05 | performance: mean zero_one loss of model 022207a |
+---------------------+-------------+-------------+---------+--------------------------------------------------+
-- holdouts (bits = codelength of each holdout's own log; the transcript bar z_L is taken at delta 2^-bits; s(L) = (z_0/z_L)^2 is the clean share) --
+---------------------+---------------+------------------+---------------------------+--------+-------+--------+----------+--------+
| holdout             | mode/route    | tune/exam rows   | asks (accepts) | checks   |   bits | z_L   | s(L)   | tuning   | exam   |
|---------------------+---------------+------------------+---------------------------+--------+-------+--------+----------+--------|
| HO-91334011A8E5D868 | numeric/check | 1500/500         | 6 (2 failed)              |   16.1 | -     | -      | halted   | opened |
+---------------------+---------------+------------------+---------------------------+--------+-------+--------+----------+--------+
```

Honest guesses from cross-validation on the training rows confirm. The third guess was moved 0.15 on purpose; it failed, and the holdout released its own value, 0.1916, on a grid of step 0.0479, with the width. Each answer has its own level, beta_i = delta c_(i-1) c_f / nu with c_j = 6 / (pi^2 (j+1)^2) and nu the number of transcripts that reach query i: the C(i-1, f) places the f failures could sit, times the grid size of each failure. The level is delta 2^-L_i with L_i the bits printed beside it. A failure costs the most, because its grid value is information, and the widths that follow must grow (from 0.096 to 0.132). `min_width()` is the next check's tolerance, computed from the public transcript alone. A width at that tolerance cannot confirm, and when it fails it leaves no grid: check 6 halted and the mechanism stopped, as Algorithm 1 does.

The book holds two charges. The first check registered the family, one `performance` certificate whose claim is that every answer `check` releases covers its population value, simultaneously, at FWER 0.05; later checks ride it, and a check whose model was fit on rows the family does not yet name replaces it with a successor that does (the predecessor is voided, so the charge stays at 0.05). The exam slice, sealed at 25 percent, opens once at the clean level, as in C14, and carries the check transcript's bits as a peek on the tuning rows; the zero-one loss lies in [0, 1], so its certificate takes the exact betting bound. The grid count uses the exact number of values a failure can release, ceil(1/gamma) + 1, which is at least the paper's 1/gamma.
