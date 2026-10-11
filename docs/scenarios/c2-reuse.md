# C2 · Can I use this certificate again?

*Continues [C1](c1-certify.md)'s `demo_book`.*

A certificate is only worth issuing if reuse is priced. `pv.reuse` is the ledger's admit: hand it certificates and the lineage of the rows about to consume them, and it returns a verdict, the inflation of the pattern, and the pass level for the next test. Rows are declared as a lineage (source fingerprint, dataset tag, row ids, clock window); the [reference](../manual/certificates.md#lineage-and-freshness) has the fields.

```python
adm = pv.reuse([cert], rows_of(X2, "demo"), ledger="demo_book")     # fresh rows
pv.reuse([cert], rows_of(X, "demo"), ledger="demo_book")            # the certifying rows
adm3 = pv.reuse([cert], rows_of(X, "demo"), ledger="demo_book", on_stale="demote")
adm5 = pv.reuse([cert_1], rows_2, ledger="book_1")                  # clocked, cross-tag
```

```text
fresh rows          : FRESH | inflation 1 | pass level 0.1
spent rows          : REFUSED -- refused: certificate lineage meets the consuming rows. [certified carrier: 3 of 63 co
spent rows, demote  : NOT FRESH | demotion: Demotion(claim='certified carrier: 3 of 63 core members at FDR q=0.1 (khat
new tag, same clock : FRESH | named overlap: []
clocked, cross-tag  : FRESH WITH DEPENDENCE FLAG | admitted: True
                      tags ('dataset_1', 'dataset_2') window (datetime.datetime(2025, 6, 17, 0, 0, tzinfo=datetime.tim
```

Four outcomes, one per line. Fresh rows admit at inflation 1 with the pass level quoted by the book. The certifying rows refuse before anything is written: identity freshness is exact (row sets, not file names), so re-testing on spent data is a hard stop by default. Under `on_stale="demote"` the same reuse is admitted but the guarantee travels marked MISCALIBRATED with the overlap recorded: you may proceed, and the record says what you did. The clocked pair is the cross-dataset case: identity-disjoint rows on a *different* dataset tag whose clock windows overlap admit as FRESH WITH DEPENDENCE FLAG, with the tags and window named. The flag reports and does not block: contemporaneous panels can share shocks, and the record carries that fact forward without pretending to price it. The particular pair and window the flag prints first are capture-dependent (slice iteration order is not fixed), so they may vary between runs; the verdict does not.

The fifth verdict, UNKNOWN LINEAGE, belongs to history: results that predate instrumentation enter through the unknown-lineage sentinel, are listed in the statement, and never bear budget or pass silently. [C4](c4-lint.md) shows the declared-record intake that produces them.

What reuse never does: recompute anything about the claim. Admission prices the *pattern* of data contact; the claim's own evidence is frozen in the certificate. Judging the claim on the new rows is a different verb with a different price, [C5](c5-grade.md).
