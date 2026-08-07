# Tests & guarantees

```
python tests/test_firewall.py     # verifier imports ⊆ stdlib+numpy, zero engine imports
python -m pytest tests/           # 35 contract + protocol + verifier-mutation tests
python tests/selftest.py          # T1-T7 acceptance ledger (17/17)
```

| Suite | Locks |
|---|---|
| `test_contracts.py` | the identities certificates stand on: transport == stock margin across model shapes (depth 1–6, incl. single-split and constant models), float32 boundary routing at exact thresholds, base_score paths, append-tree on/off semantics |
| `test_protocol.py` | refusal codes (R-05, coded R-07), policy canonicalization, order-free bundle hashing, any-single-field mutation breaks the certificate id, limits always present and rendered, verdict grammar closed set, sieve determinism per seed; v0.2 vocabulary: flags-off schema stability, deterministic low-side (≤) claim recovery with the extended dictionary |
| `test_verifier_mutations.py` | each of the verifier's seven checks individually killed by a targeted tamper, incl. a resurrected-interaction attack with an attacker-re-hashed bundle |
| `selftest.py` | end-to-end: external artifact → audit → enforce → stock reload → independent verify → mutation → discover → refusal |

Deferred by design until the respective layers land: bundle-schema snapshots and renderer golden files (reporting layer), enforce-signature tests (refit mode), and the multi-version xgboost/lightgbm adapter matrix (CI env matrix).

## Guarantees, estimates, and caveats

- **Controlled:** the FDR of the claims ledger, by BH(q) over a single selection-conditional S2 pass. **Exact:** every `#exact` quantity is an arithmetic fact of the artifact (and sample): structural zeros, masses, prices, off-rule bit-identity.
- **Float32 is the write-back resolution:** appended-tree coefficients are stored as float32 by the native format; certification tolerances (structural ≤ 1e-4, behavioral ≤ 1e-3) state this rather than hide it. Measured values are orders of magnitude below the gates.
- **Estimates are labeled:** holdout deltas carry CIs; the sieve's penumbra, toll bounds, spotlight, and M_C are plug-in quantities with disclosed proxies; ridge* is a hindsight benchmark for the frame-blind class, not a deployable model.
- **The limits panel is non-removable:** no proxy claim, no causal claim, no off-distribution claim, no fairness-of-outcomes claim: certified absence of named rule families in the file at the stated hash, and nothing more.
- **`refit_scope='train'`** re-uses S2 in the sieve's refit and carries a residual selection-optimism term (measured small); `'s2'` avoids it; both rows print.
- **Verifier independence is a checked property,** not a promise (see the firewall and mutation suites above).

## Not in 0.9.0

lightgbm adapter; xgboost version matrix beyond 3.3.0; sealed PDF export (the HTML certificate binds to the same bundle); behavioral probe for `involving` clauses (structural only); refit surgery mode (excise only); `monitor`; vertical report packs; NaN-aware transported audit (R-07 declared instead).
