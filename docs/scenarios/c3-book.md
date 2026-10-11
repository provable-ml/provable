# C3 · What does my book guarantee?

*A fresh `batch_book`; [C1](c1-certify.md)'s single-entry book is the degenerate case.*

A book is an account. One rule always holds: the realized error level of the active book is at most the sum of its certificates' levels, the additive bound. A second, sharper discipline holds when you quote each batch's level from the book *before* running it: the online batch step-up, under which the union FDP sits at the account target at every step.

```python
led = ct.Ledger("batch_book", budget=0.10)
for i in (1, 2, 3):
    lv = led.next_level(1)
    rep = pv.certify(world(i)..., q=lv, ledger=led, dataset_tag=f"batch{i}")
print(led.statement().render()); print(led.combine_batches())
```

```text
batch 1: quoted level 0.0608 -> khat 1
batch 2: quoted level 0.0304 -> khat 2
batch 3: quoted level 0.1182 -> khat 7

-- ledger statement: batch_book --
+--------------------------------------------+---------+
| budget (target)                            |  0.1    |
| charged (additive, active book)            |  0.2094 |
| reserved (registered charges open)         |  0      |
| remaining                                  | -0.1094 |
| active certificates                        |  3      |
| voided                                     |  0      |
| unknown-lineage items (not budget-bearing) |  0      |
+--------------------------------------------+---------+
-- active certificates --
+---------------------+---------+-------------+---------+--------------------------------------------------+
| id                  | kind    | guarantee   |   level | claim                                            |
|---------------------+---------+-------------+---------+--------------------------------------------------|
| CD-4350F776DAF159CC | carrier | FDR         |  0.0608 | certified carrier: 1 of 28 core members at FDR q |
| CD-37A03FD546393922 | carrier | FDR         |  0.0304 | certified carrier: 2 of 34 core members at FDR q |
| CD-8E4CDCFB51CCA9A8 | carrier | FDR         |  0.1182 | certified carrier: 7 of 45 core members at FDR q |
+---------------------+---------+-------------+---------+--------------------------------------------------+

combine_batches: {'target': 0.1, 'batches': [{'n': 28, 'level_used': 0.060792710185402665, 'level_prescribed': 0.060792710185402665, 'R': 1, 'R_plus': 2}, {'n': 34, 'level_used': 0.030396355092701333, 'level_prescribed': 0.015645182768302155, 'R': 2, 'R_plus': 3}, {'n': 45, 'level_used': 0.1182080475827274, 'level_prescribed': 0.03152214602206064, 'R': 7, 'R_plus': 8}], 'fdp_hat': 0.1053, 'conforms': False, 'guarantee': 'levels deviate from the batch step-up schedule; governing bound is additive: union FDP <= 0.2094 (M2 Prop 6.1)', 'additive_level': 0.2094}
```

Read the statement first: three active certificates, charged 0.2094 against a 0.10 target, remaining negative. This book *deviated*: the demo quoted levels without declaring the coming batch's test count, so the levels used drift from the schedule's prescription. `combine_batches` is the audit: it recomputes the prescribed level per batch from the realized history, reports `conforms: False`, and states the governing bound, which falls back to the additive sum (union FDP at most 0.2094). Deviation is detected, never papered over; a conforming book (quote with the upcoming batch's test count, levels match the prescription) carries the union guarantee at target with the FDP estimate alongside. The suite's simulation locks are the conformance witness.

Two more account objects, on a clean book:

```text
reserved 0.02 over a 2-member vocabulary; per-claim bar = 0.02/2 = 0.01
claim in vocabulary, p=0.004 : {'reservation': 'RES-0001', 'member': 'b0>=0.4 [batch1]', 'p': 0.004, 'bar': 0.01, 'vocabulary_size': 2, 'valid': True, 'lineage_free': True, 'source': 'M2 Prop 6.3', 'reason': 'valid at the per-pair bar'}

-- ledger statement: reg_book --
+--------------------------------------------+------+
| budget (target)                            | 0.05 |
| charged (additive, active book)            | 0    |
| reserved (registered charges open)         | 0.02 |
| remaining                                  | 0.03 |
| active certificates                        | 0    |
| voided                                     | 0    |
| unknown-lineage items (not budget-bearing) | 0    |
+--------------------------------------------+------+
-- notes --
  registered follow-up validated on fresh 2026 rows
```

The reservation with a vocabulary is the registered charge: fix the member/row-set pairs in advance, and any claim at p at most alpha over the vocabulary size is valid under arbitrary reuse, order, and lineage, the one guarantee re-testing cannot break. A claim outside the vocabulary is refused, whatever its p. A `reserve` without a vocabulary is a plain hold. Notes are append-only provenance, rendered in the statement, inert to the account. Grades, the third statement section a working book accumulates, are [C5](c5-grade.md)'s subject.
