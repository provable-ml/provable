# C5 · Did it work? Grade it.

*Synthetic world throughout; grading spends data, so what to grade on is a registration decision.*

A certificate pre-commits a family, effects, and directions. One later realization decides nothing; N firings against the committed direction at a quoted level decide something. `pv.grade` is that judgment as a verb: admission first, so spent rows refuse and the level is quoted by the book, then each certified cell is tested one-sided in its committed direction on its own firings on the new rows, and the scorecard is written to the book.

```python
rep = pv.certify(X, y, q=0.10, ledger="grade_book", dataset_tag="sim")
gr = pv.grade(rep.certificate, X_new, y_new, ledger="grade_book", level=0.05)
print(gr.render())
```

```text
certified: khat 3; committed directions [1, -1, -1]

-- grade: CD-0F600E78988BAD21 --
admission: FRESH; graded level 0.05 over k=3 cells; one-sided bar 0.01667 per cell (bar = level/k)
+---------------------------------+-------------+-----------+-------------+--------------+---------+---------+-------------+-------------+-----------+
| atom                            |   direction |   n_fired |   fire_rate |   mean_fired |      se |   agree |   p_confirm |   p_reverse | verdict   |
|---------------------------------+-------------+-----------+-------------+--------------+---------+---------+-------------+-------------+-----------|
| g0>=0.442698                    |           1 |      1075 |      0.3359 |       0.7292 | 0.03479 |  0.747  |   7.703e-98 |           1 | CONFIRMED |
| (g1>=-0.150942)&(g2>=0.0699856) |          -1 |       833 |      0.2603 |      -0.5112 | 0.03904 |  0.6795 |   1.802e-39 |           1 | CONFIRMED |
| (g1>=0.133792)&(g2>=0.0699856)  |          -1 |       682 |      0.2131 |      -0.6011 | 0.04229 |  0.7141 |   3.783e-46 |           1 | CONFIRMED |
+---------------------------------+-------------+-----------+-------------+--------------+---------+---------+-------------+-------------+-----------+
scorecard: 3 CONFIRMED, 0 REVERSED, 0 NOT CONFIRMED. REVERSED is significant evidence against the original direction at the graded bar, not merely absence of confirmation.
book: entry appended to grade_book (see the grades section of the statement); this grade is budget-bearing.

grade on the certifying rows: REFUSED -- refused: certificate lineage meets the consuming rows. [certified carrier: 3 

-- ledger statement: grade_book --
+--------------------------------------------+-------+
| budget (target)                            |  0.1  |
| charged (additive, active book)            |  0.15 |
| reserved (registered charges open)         |  0    |
| remaining                                  | -0.05 |
| active certificates                        |  1    |
| voided                                     |  0    |
| unknown-lineage items (not budget-bearing) |  0    |
+--------------------------------------------+-------+
-- active certificates --
+---------------------+---------+-------------+---------+--------------------------------------------------+
| id                  | kind    | guarantee   |   level | claim                                            |
|---------------------+---------+-------------+---------+--------------------------------------------------|
| CD-0F600E78988BAD21 | carrier | FDR         |     0.1 | certified carrier: 3 of 46 core members at FDR q |
+---------------------+---------+-------------+---------+--------------------------------------------------+
-- grades (budget-bearing; each graded level charges the account unless it consumed a reservation) --
+---------------------+---------+---------+-----+-------------+------------+------------+---------------+
| certificate         |   level |     bar |   k |   confirmed |   reversed |   not conf | reservation   |
|---------------------+---------+---------+-----+-------------+------------+------------+---------------|
| CD-0F600E78988BAD21 |    0.05 | 0.01667 |   3 |           3 |          0 |          0 |               |
+---------------------+---------+---------+-----+-------------+------------+------------+---------------+
```

The graded object is deliberately the marginal when-fired mean, not the joint partial that certified the cell: the certificate committed a direction, and the grade asks whether fresh firings move that way. The bar is the graded level over the number of cells (or alpha over the vocabulary size under a registered reservation, the pre-registered form). Three verdicts: CONFIRMED, NOT CONFIRMED, and REVERSED, and the third is not the second: REVERSED means significant evidence *against* the original direction at the graded bar, a finding in its own right, and the render says so.

Grading is budget-bearing. The admission step charges the account (or consumes the named reservation), the entry lands in the grades section of the statement, exactly as printed above, and grading the certifying rows refuses before any write. That cost is the point: a scorecard you can generate for free is a scorecard you can generate until it flatters you. Choose the grading data the way you would choose a holdout, once, and the book keeps you honest about it.
