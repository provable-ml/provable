# C17 · Price a result that saw the holdout

*Continues [C14](c14-seal.md)'s `hold_book`; the ids are those of the C14 run, and every other number is reproduced by the code shown.*

A certificate from a holdout records more than its test rows. Its model was chosen with answers from the tuning slice, so the certificate carries those rows as a peek: the rows, and the bits through which they were seen. Reuse against new rows now has three outcomes instead of two. The next block stands in for 500 hourly rows that arrive after the holdout.

```python
import numpy as np, pandas as pd
from tabulate import tabulate
import provable as pv
from provable import certificates as ct
from provable.certificates import transcript as tr

led = pv.Ledger("hold_book")
ho = pv.Holdout(led, led.holdout_ids()[0])
cert = led.active()[0]
print(f"{cert.id}: {ct.peeks_of(cert)[0]['bits']:.2f} bits on the tuning rows of {ho.id}")
rng = np.random.default_rng(17)
nxt = pd.DataFrame(rng.normal(size=(500, 8)), columns=[f"x{i}" for i in range(8)])
nxt["start_utc"] = pd.date_range("2026-06-01", periods=500, freq="h", tz="UTC")
nxt["end_utc"] = nxt["start_utc"] + pd.Timedelta(hours=1)
next_block = ct.Lineage.of(ct.slice_from_frame(nxt, "demo", rows=nxt.index))
for name, rows in (("next block", next_block), ("tuning rows", ho.tuning_lineage()), ("exam rows", ho.exam_lineage())):
    try:
        a = pv.reuse([cert], rows, ledger=led, level=0.05)
        print(f"{name:12s}: {a.verdict.label} | bits {a.bits:.2f} | inflation {a.inflation:.4g} | pass level {a.level:.4g}")
    except ct.FreshnessError as e:
        print(f"{name:12s}: REFUSED -- {str(e)[:100]} ...")
```

```text
CD-850EE205AAC003FE: 7.19 bits on the tuning rows of HO-A926AA9A91DC073D
next block  : FRESH | bits 0.00 | inflation 1 | pass level 0.05
tuning rows : PEEKED | bits 7.19 | inflation 146.3 | pass level 0.0003418
exam rows   : REFUSED -- refused: certificate lineage meets the consuming rows. [performance: mean squared paired gain of mod ...
```

On rows it never touched, the certificate is FRESH. On its own exam rows it is NOT FRESH, refused by default as in [C2](c2-reuse.md). On the tuning rows it is PEEKED: no row of its direct lineage is there, but its proposal saw them through 7.19 bits. A test on those rows that consumes this certificate is admitted at inflation 2^L = 146.3 and pass level 0.05 x 2^-7.19 = 0.00034, the Kraft union bound over transcripts read backwards. Peeks compose: one transcript per holdout at its largest bits, distinct holdouts adding, inherited by any certificate that consumes this one. `grade` on such rows tests its cells at the repaired level and charges the book the nominal one, which is what the test then holds; `certify` on rows a transcript saw refuses by name, since it does not price peeks.

Lint prices the same thing in a declared pipeline, under R7. A run that tuned through 40 answers on 800 rows and then reported its score on those rows at 0.05:

```python
findings = pv.lint({"name": "tuned through 40 answers, reported on the tuning rows", "instrumented": False, "q": 0.05,
                   "test_rows": {"source_id": "panel", "rows": list(range(2000, 2800))},
                   "proposal_rows": {"source_id": "panel", "rows": list(range(0, 2000))},
                   "peeks": [{"rows": {"source_id": "panel", "rows": list(range(2000, 2800))}, "bits": 40.3, "holdout": "tuning log"}]})
print(findings)
print(findings.repair("R7"))
```

```text
-- lint findings: tuned through 40 answers, reported on the tuning rows --
+---------+-----------+----------------------------------------------+------------------------------------------------------------------+------------------------+
| check   | verdict   | price                                        | detail                                                           | source of record       |
|---------+-----------+----------------------------------------------+------------------------------------------------------------------+------------------------|
| R1      | PASS      | guarantee preserved                          | proposal rows disjoint from test rows; any measurable search on  | M2 Thm 3.1; U1 Lem 2.4 |
| R7      | VIOLATION | realized level at most min(1, q 2^L) = 1.000 | test rows seen through 40.30 bits, 0.00 priced: tuning log (40.3 | RRSSTW20 Thm 3.2       |
+---------+-----------+----------------------------------------------+------------------------------------------------------------------+------------------------+
{'bits': 40.3, 'priced_bits': 0.0, 'inflation': 1353657597944.1948, 'rule': 'test at q 2^-L (the transcript union bound)', 'repaired_level': 3.6936962549418115e-14}
```

R1 passes, since no proposal row is a test row; R7 does not, since the test rows were seen through 40.3 bits that the reported level never priced. The realized level is at most min(1, q 2^L), here 1, and the exact repair tests at q 2^-L. A run whose bar priced the bits declares `priced_bits`, and R7 passes, as for the holdout's own transcript certificate in [C15](c15-transcript.md).

The rates register converts bits to rows. `rows_for_bits(n, L)` is the number of rows a holdout that has spent L bits needs to certify with the width of a clean holdout of n rows, n (z_L / z_0)^2 = n / s(L); read the other way, n rows after L bits certify like n s(L) clean ones.

```python
table = [[L, f"{tr.z_bar(0.05, L):.3f}", f"{tr.clean_share(0.05, L):.3f}", f"{ct.rows_for_bits(1000, L):,.0f}"] for L in (0, 1, 2, 3, 8.3, 23.1, 37.94, 38.94, 86.1)]
print(tabulate(table, headers=["bits L", "z_L", "s(L)", "rows_for_bits(1000, L)"], tablefmt="psql"))
print(ct.REGISTER["rows_for_bits"].line())
```

```text
+----------+--------+--------+--------------------------+
|   bits L |    z_L |   s(L) |   rows_for_bits(1000, L) |
|----------+--------+--------+--------------------------|
|     0    |  1.645 |  1     |                    1,000 |
|     1    |  1.96  |  0.704 |                    1,420 |
|     2    |  2.241 |  0.539 |                    1,857 |
|     3    |  2.498 |  0.434 |                    2,306 |
|     8.3  |  3.601 |  0.209 |                    4,792 |
|    23.1  |  5.713 |  0.083 |                   12,062 |
|    37.94 |  7.263 |  0.051 |                   19,496 |
|    38.94 |  7.356 |  0.05  |                   19,999 |
|    86.1  | 10.896 |  0.023 |                   43,880 |
+----------+--------+--------+--------------------------+
[rate] rows_for_bits: validity -> rows (identity at fixed effect size, one-sided Gaussian bar)
```

Each bit costs a few more rows than the one before: the first adds 420 to a clean 1,000, the second 437, the third 449, and the bit that reaches the retirement threshold 503. Measured against the rows already needed the price falls instead, from 42 percent for the first bit to 31 and then 24 percent, as the clean share goes from 1 to 0.704, 0.539 and 0.434. The retirement threshold, 38.94 bits, costs twenty times the rows; that is why the tuning slice retires there and the next block is sealed in its place.
