# C6 · A public dataset, end to end

*The C1 walkthrough on public data: the Titanic passenger table (891 rows, as distributed in the seaborn-data repository). Nothing is planted; whatever certifies is what the data supports at the declared rate.*

```python
import pandas as pd
import provable as pv

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
X = pd.DataFrame({
    "pclass": df["pclass"].astype(float),
    "sex_female": (df["sex"] == "female").astype(float),
    "age": df["age"].fillna(df["age"].median()),
    "sibsp": df["sibsp"].astype(float),
    "parch": df["parch"].astype(float),
    "fare": df["fare"].astype(float),
    "embarked_C": (df["embarked"] == "C").astype(float),
    "embarked_Q": (df["embarked"] == "Q").astype(float),
})
y = df["survived"].astype(float)
X_fit, y_fit, X_hold, y_hold = X.iloc[:700], y.iloc[:700], X.iloc[700:], y.iloc[700:]

rep = pv.certify(X_fit, y_fit, q=0.10, ledger="titanic_book", dataset_tag="titanic", seed=0)
print(rep.render())
```

```text
-- certify: standard report --
+----------------+------------------+
| manifest       |                  |
|----------------+------------------|
| x_sha          | 8ef9ef1a0915ef41 |
| y_sha          | 4f2d0fa4ead93e59 |
| dictionary_sha | 8ec4dc9de87d6f8e |
| config_sha     | 1e43e0092612370e |
| instrument_sha | 492b390f7ee5a143 |
| seed           | 0                |
| provable       | 0.10.0           |
+----------------+------------------+
+-----+-----+--------+----------+--------+-----+
|   n |   P |   core |   k cert |   khat |   q |
|-----+-----+--------+----------+--------+-----|
| 700 | 130 |     25 |        1 |      1 | 0.1 |
+-----+-----+--------+----------+--------+-----+
engine vs certified core (test R2): engine 0.4769 vs core 0.4143; lead +0.06258
-- certified claims (selective intervals at alpha_sel = khat*q/m = 0.004; FCR) --
+---------------+----------+---------+---------+-------------+-------------+------+-----------+
| atom          |   effect |   ci_lo |   ci_hi |   ci_lo_sel |   ci_hi_sel |    t |         p |
|---------------+----------+---------+---------+-------------+-------------+------+-----------|
| sex_female>=1 |   0.4447 |  0.2551 |  0.6343 |      0.1649 |      0.7245 | 4.62 | 6.301e-06 |
+---------------+----------+---------+---------+-------------+-------------+------+-----------+
-- atom card (est; marginal per-atom effects on S2 and held-out test, never the proposal rows; net across cells via rep.core_fit()) --
+---------------+-------------+-----------+-------------+----------+---------+------------+-----------+------------+--------------+
| atom          |   direction |   fire_s2 |   fire_test |   eff_s2 |   se_s2 |   eff_test |   se_test |   agree_s2 |   agree_test |
|---------------+-------------+-----------+-------------+----------+---------+------------+-----------+------------+--------------|
| sex_female>=1 |           1 |    0.3893 |      0.3771 |   0.5165 | 0.05479 |     0.6654 |   0.05943 |     0.7353 |        0.803 |
+---------------+-------------+-----------+-------------+----------+---------+------------+-----------+------------+--------------+
-- penumbra (Thm 6a band; null floor subtracted, est) --
+-----------------+------------------+--------------+-------------+--------------------------------------+
|   raw band mass |   m0 (est nulls) |   null floor |   corrected | grade                                |
|-----------------+------------------+--------------+-------------+--------------------------------------|
|          225.77 |               48 |        22.78 |      202.99 | est: Storey null count at lambda=0.5 |
+-----------------+------------------+--------------+-------------+--------------------------------------+
ledger: 1 carrier certificate appended to titanic_book (CD-49D803667B47D89F)
book after this run: charged 0.1000 of budget 0.1000, remaining 0.0000
```

One claim survives the gate: `sex_female>=1`, a joint partial of 0.44 on the survival probability, with its FCR-calibrated interval. The binary target runs as a linear probability model, so effects read as probability differences. The atom card shows the marginal side of the same cell: the when-fired effect is larger than the joint partial (0.52 on S2, 0.67 on held-out test) because the marginal comparison does not hold class, age, or fare fixed, and the agreement rate is the share of fired rows whose outcome sign matches the committed direction. No unit cost is declared, so the card renders as a pure evaluation surface. The engine holds a modest lead over the one-atom core; that gap is dark mass in [B4](b4-darkmass.md)'s sense, structure the engine sees that no certifiable cell in this dictionary carries at q = 0.10.

The last 191 rows were held out and are fresh to the book, so the committed direction can be judged:

```python
g = pv.grade(rep.certificate, X_hold, y_hold, ledger="titanic_book", level=0.05, dataset_tag="titanic")
print(g.render())
```

```text
-- grade: CD-49D803667B47D89F --
admission: FRESH; graded level 0.05 over k=1 cells; one-sided bar 0.05 per cell (bar = level/k)
+---------------+-------------+-----------+-------------+--------------+---------+---------+-------------+-------------+-----------+
| atom          |   direction |   n_fired |   fire_rate |   mean_fired |      se |   agree |   p_confirm |   p_reverse | verdict   |
|---------------+-------------+-----------+-------------+--------------+---------+---------+-------------+-------------+-----------|
| sex_female>=1 |           1 |        63 |      0.3298 |        0.746 | 0.05528 |   0.746 |   8.326e-42 |           1 | CONFIRMED |
+---------------+-------------+-----------+-------------+--------------+---------+---------+-------------+-------------+-----------+
scorecard: 1 CONFIRMED, 0 REVERSED, 0 NOT CONFIRMED. REVERSED is significant evidence against the original direction at the graded bar, not merely absence of confirmation.
book: entry appended to titanic_book (see the grades section of the statement); this grade is budget-bearing.
```

Admission runs first and the held-out rows admit FRESH; with one cell the one-sided bar is the full graded level. The verdict is CONFIRMED on 63 firings. The grade charges the account: the book was already at its declared budget after certification, so the statement now shows the additive total past target, exactly as [C3](c3-book.md) reads such a book. Grading was possible here only because rows were withheld before certification; on a fully spent panel the same call refuses, which is [C5](c5-grade.md)'s point about choosing grading data once.

Data: the Titanic passenger table as shipped in the seaborn-data repository.
