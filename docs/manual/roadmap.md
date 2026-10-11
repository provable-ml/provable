# Road to 1.0.0

- A `schema_version` field in the evidence bundle, honored by the verifier.
- `enforce` grows by keyword-only arguments from here; the positional surface is frozen at 0.9.0 (lock TH1).
- The deferred snapshot/golden tests (bundle-schema snapshots, renderer golden files). The enforce-signature lock (TH1) and the xgboost version matrix in CI have landed.
- Semantic versioning from 0.9.0: breaking changes to the evidence-bundle schema or the public API happen only at a major version; **1.0.0 marks the freeze of both.**

Planned beyond the freeze (no dates attached): lightgbm adapter, refit surgery mode, behavioral probe for `involving`, sealed PDF export, NaN-aware transported audit.
