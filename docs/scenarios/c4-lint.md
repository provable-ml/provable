# C4 · Price what that pipeline did

No instrumentation required: describe any past analysis as a declared run record, and the linter prices it against the statements of record, check by check, each row carrying its source tag.

```python
r = ct.RunRecord(name="legacy screen-then-gate", instrumented=False, q=0.10, m=60, khat=4,
                 test_rows=L,
                 screen=ct.ScreenDecl(threshold=1.0, rows=L, label="|t|>=1 prescreen on the gate rows"),
                 retests=[ct.Retest(rows=L, level=0.05, on_certified=True, label="re-test on the certifying rows")],
                 union_claims=[ct.UnionClaim(cert_levels=[0.10, 0.05], claimed_level=0.10, label="joint claim at 0.10")])
f = pv.lint(r); print(f.render()); print(f.repair("R4"))
try:
    f.repair("R3")
except KeyError as e:
    print(f"repair('R3') -> {e}")
```

```text
-- lint findings: legacy screen-then-gate --
+---------+-----------+----------------------------------------------+-----------------------------------------------------------------+---------------------------+
| check   | verdict   | price                                        | detail                                                          | source of record          |
|---------+-----------+----------------------------------------------+-----------------------------------------------------------------+---------------------------|
| R1      | PASS      | guarantee preserved                          | proposal rows disjoint from test rows; any measurable search on | M2 Thm 3.1; U1 Lem 2.4    |
| R3      | VIOLATION | re-test at alpha on the certifying rows cert | re-test re-test on the certifying rows rows meet the certifying | M2 Thm 5.2; U1 Lem 6.1(c) |
| R4      | VIOLATION | realized level min(1, q/s) = 0.3151; inflati | screen |z|>1 then gate at q=0.1 on shared rows (s=0.3173)       | M2 Thm 5.3                |
| R5      | VIOLATION | valid statement is the additive level 0.1500 | joint claim at 0.10: 2 certificate sets claimed jointly at 0.1  | M2 Thm 6.5; Prop 6.3      |
+---------+-----------+----------------------------------------------+-----------------------------------------------------------------+---------------------------+
{'repaired_level': 0.03173105078629142, 'rule': 'gate at q*s, equivalently divide screened p-values by s'}
repair('R3') -> 'no repair recorded for R3'
```

The screen-then-gate row is the classic: a prescreen at |z| at or above 1 on the same rows the gate then certifies at q = 0.10 does not deliver 0.10; the realized level is q over the screen's survival mass s = 0.3173, an inflation of 3.15x, and the repair hands back the exact remedy as numbers: gate at q times s, equivalently divide the screened p-values by s. The re-test row prices the other habit, re-testing at alpha on the certifying rows: its price line states the valid object, confirmation levels rather than fresh evidence. The joint claim row applies the additive rule: two certificate sets claimed together at 0.10 are worth 0.15, and the valid statement is printed. Declared lineage a linter cannot see returns UNKNOWN, never PASS.

Each row's source tag names the exact theorem or lemma of record behind the price, from the two source papers on the [citation page](../cite.md), so the rule-to-source map is auditable line by line. Repair returns numbers, not advice; what you do with a priced pipeline is your call, and if the answer is "judge it properly on fresh data", that is [C5](c5-grade.md).
