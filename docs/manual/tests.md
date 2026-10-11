# Tests & guarantees

```
python tests/test_firewall.py     # verifier imports ⊆ stdlib+numpy, zero engine imports
python -m pytest tests/           # 242 tests: contracts, protocol, mutations, certificate layer
python tests/selftest.py          # T1-T7 acceptance ledger (17/17)
```

| Suite | Locks |
|---|---|
| `test_contracts.py` | the identities certificates stand on: transport == stock margin across model shapes (depth 1–6, incl. single-split and constant models), float32 boundary routing at exact thresholds, base_score paths, append-tree on/off semantics |
| `test_protocol.py` | refusal codes (R-05, coded R-07), policy canonicalization, order-free bundle hashing, any-single-field mutation breaks the certificate id, limits always present and rendered, verdict grammar closed set, sieve determinism per seed; v0.2 vocabulary: flags-off schema stability, deterministic low-side (≤) claim recovery with the extended dictionary |
| `test_verifier_mutations.py` | each of the verifier's seven checks individually killed by a targeted tamper, incl. a resurrected-interaction attack with an attacker-re-hashed bundle |
| `selftest.py` | end-to-end: external artifact → audit → enforce → stock reload → independent verify → mutation → discover → refusal |
| `test_twin_suite.py` | the certificate layer, one lock per stated property (173 locks): lineage and the book, certify and grade, the mediated holdout and its bits, the watch, the route; each lock prints the value it measured |

Deferred by design: bundle-schema snapshots and renderer golden files (reporting layer), and a lightgbm adapter. xgboost 2.0.3, 2.1.4, 3.0.5 and the current release run in CI, and lock TH1 holds the released signatures.

## Validation records

`validation/null_result.py` runs the simulations that set looking at a mediated holdout against not looking: one stationary study where cross-validation on the training rows already finds the model and looking gains nothing, its repetition with 400 training rows, the twenty-question adversary that shows why the price is needed, the two designs where looking gains (a changed law, too few training rows), and the coverage of the Gaussian bar on skewed losses. Every look goes through `seal`, `ask` and `open` under the Gaussian bar, named with `tail="gaussian"`; for the Brier loss the default is now the exact betting bound. Its output is `validation/null_result.txt`; the run prints no timings and is seeded throughout, so a run on the same environment reproduces the file. It takes about fifteen minutes.

`validation/guarantee.py` checks when a look is sure not to hurt and when it is sure to buy something. A gated look, which leaves the default model only for a candidate whose gain is certified at level delta/K, is set against the bound that no look safe at level delta can beat; its expected harm is computed; the first-order cost of an ungated comparison loop is set against the loop run through `ask`; the gated look is run in three designs (nothing left to find, too few training rows, a changed law); and the value of looking at the latest rows when the effect of a choice persists across time blocks is simulated against its closed form, for a stationary and a random-walk effect. Its output is `validation/guarantee.txt`, seeded and without timings like the other; it takes about seven minutes.

## Guarantees, estimates, and caveats

- **Controlled:** the FDR of the claims ledger, by BH(q) over a single selection-conditional S2 pass. **Exact:** every `#exact` quantity is an arithmetic fact of the artifact (and sample): structural zeros, masses, prices, off-rule bit-identity.
- **Float32 is the write-back resolution:** appended-tree coefficients are stored as float32 by the native format; certification tolerances (structural ≤ 1e-4, behavioral ≤ 1e-3) state this rather than hide it. Measured values are orders of magnitude below the gates.
- **Estimates are labeled:** holdout deltas carry CIs; the sieve's penumbra, toll bounds, spotlight, and M_C are plug-in quantities with disclosed proxies; ridge* is a hindsight benchmark for the frame-blind class, not a deployable model.
- **The limits panel is non-removable:** no proxy claim, no causal claim, no off-distribution claim, no fairness-of-outcomes claim: certified absence of named rule families in the file at the stated hash, and nothing more.
- **`refit_scope='train'`** re-uses S2 in the sieve's refit and carries a residual selection-optimism term (measured small); `'s2'` avoids it; both rows print.
- **Verifier independence is a checked property,** not a promise (see the firewall and mutation suites above).

## Not in this release

lightgbm adapter; sealed PDF export (the HTML certificate binds to the same bundle); behavioral probe for `involving` clauses (structural only); refit surgery mode (excise only); vertical report packs; NaN-aware transported audit (R-07 declared instead).
