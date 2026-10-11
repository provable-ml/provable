# C15 · No exam slice: certify on the rows you tuned on

*Continues [C14](c14-seal.md)'s world: its first block builds the panel, the six candidates and the lineages used here, and the same holdout rows are sealed without an exam slice. The clock and the custody salt were fixed when this page was generated, so the ids, the model hash and the dates are specific to that run; every other number is reproduced by the code shown.*

With `exam_share=0` the route is the transcript certificate. `open` takes the bound on the tuning slice itself, at the level the log has left, delta 2^-L, and the tuning slice closes with it.

```python
from provable.certificates import transcript as tr

ho = pv.seal(hold[feats], hold["y"], "kraft_book", lineage=H_rows, baseline=train["y"].mean(),
             baseline_lineage=S_rows, claim="gain", exam_share=0.0)
for name, f in candidates:
    ho.ask(f, lineage=S_rows)
cert = ho.open(candidates[3][1], lineage=S_rows)
fam = cert.guarantee.family
print(cert.claim)
print(f"bits {fam['bits']:.2f}; level at the bar {fam['level_at_bar']:.4g}; bar z_L {fam['bar']:.3f} (clean bar {tr.z_bar(0.05, 0):.3f})")
print(pv.lint(ho))
try:
    ho.ask(candidates[4][1], lineage=S_rows)
except ValueError as e:
    print(e)
```

```text
performance: mean squared paired gain of model 99e5a507e92f over the sealed baseline at least 0.723077 (FWER 0.05, CLT approximation; HO-D06C14EBDFE1FBF1, tuning slice, 7.19 bits spent)
bits 7.19; level at the bar 0.0003418; bar z_L 3.396 (clean bar 1.645)
-- lint findings: holdout HO-D06C14EBDFE1FBF1 --
+---------+-----------+-------------------------------------------+------------------------------------------------------------------+------------------------+
| check   | verdict   | price                                     | detail                                                           | source of record       |
|---------+-----------+-------------------------------------------+------------------------------------------------------------------+------------------------|
| R1      | PASS      | guarantee preserved                       | proposal rows disjoint from test rows; any measurable search on  | M2 Thm 3.1; U1 Lem 2.4 |
| R7      | PASS      | priced: the bar already carries 7.19 bits | test rows seen through 7.19 bits: HO-D06C14EBDFE1FBF1 (7.19 bits | RRSSTW20 Thm 3.2       |
+---------+-----------+-------------------------------------------+------------------------------------------------------------------+------------------------+
ask refused: the tuning slice of HO-D06C14EBDFE1FBF1 closed at #7, when its transcript certificate was issued
```

The six answers are the ones C14 got, 7.19 bits, and the bound now uses all 1,000 rows at z_L = 3.396: a paired gain of at least 0.723, against 0.637 from C14's 200-row exam. That is the clean share at work: s(L) = 0.235 exceeds the exam share 0.2, so the transcript certificate is the narrower one.

The bar is the Kraft union bound over transcripts. Suppose every history fixes codelengths that satisfy Kraft's inequality (the code is fixed at sealing), every choice of the analyst depends only on their own rows, independent randomness and past answers, and a model fixed in advance exceeds its bar at level gamma with probability at most gamma. Then the bound at level delta 2^-L, L the realized codelength, fails with probability at most delta for every strategy and every stopping rule. The argument is the transcript union bound of Guess and Check (Rogers, Roth, Smith, Srebro, Thakkar and Woodworth, AISTATS 2020) and of the Ladder (Blum and Hardt, ICML 2015), stated for any code fixed at sealing. Lint reads the holdout's run and prices it under R7: the test rows were seen through 7.19 bits, and the bar carries all of them.

Is the price needed? A deliberate adversary asks 20 yes/no questions, each the sign of one random direction's correlation with the holdout (eta = 0 turns the comparison into a sign), and submits the model that adds every direction with the sign it was told. On pure-noise rows the true gain of that model is negative. Two hundred such holdouts of 400 rows, each through `ask` and `open`:

```python
import tempfile


def adversary(rng, k=20, h=400):
    X = pd.DataFrame(rng.normal(size=(h, k)), columns=[f"c{j}" for j in range(k)])
    y = rng.normal(size=h)
    hn = pv.seal(X, y, tempfile.mkdtemp(), lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(),
                 claim="gain", eta=0.0, exam_share=0.0)
    cols, inc, signs = list(X.columns), np.zeros(k), []
    for j in range(k):
        w = inc.copy()
        w[j] += 1e-4
        ok = hn.ask(lambda Z, w=w: Z[cols].to_numpy() @ w, lineage=ct.Lineage()).accepted
        signs.append(1.0 if ok else -1.0)
        inc = w if ok else inc
    c = np.sqrt(2 / (np.pi * h))
    f = lambda Z: Z[cols].to_numpy() @ (c * np.asarray(signs))
    cert = hn.open(f, lineage=ct.Lineage())
    true_gain = -c * c * k
    d = y ** 2 - (y - f(X)) ** 2
    unpriced = d.mean() - tr.z_bar(0.05, 0) * d.std(ddof=1) / np.sqrt(h)
    return unpriced > true_gain, cert.artifact["bound"] > true_gain, hn.state["bits"]


rng = np.random.default_rng(15)
runs = [adversary(rng) for _ in range(200)]
print(f"200 null holdouts, 20 sign answers each ({np.mean([r[2] for r in runs]):.1f} bits on average):")
print(f"  bound at the clean bar z_0, ignoring the bits : wrong in {sum(r[0] for r in runs)}/200")
print(f"  bound at the transcript bar z_L              : wrong in {sum(r[1] for r in runs)}/200")
```

```text
200 null holdouts, 20 sign answers each (21.8 bits on average):
  bound at the clean bar z_0, ignoring the bits : wrong in 200/200
  bound at the transcript bar z_L              : wrong in 1/200
```

Twenty answers buy a model whose bound at the clean bar is wrong every time; the bound at the priced bar is wrong once in 200. The bar is the Gaussian form, the CLT approximation, and far in the tail it is an estimate. For a loss with a declared range, `tail="hoeffding"` makes the single-model premise exact: the width becomes span sqrt((ln(1/delta) + L ln 2) / (2n)), with span the range of the claim statistic, hi - lo for a risk claim and 2(hi - lo) for a gain claim, which is a difference of two losses in [lo, hi].
