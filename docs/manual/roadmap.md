# Road to 1.0.0

- A `schema_version` field in the evidence bundle, honored by the verifier.
- `enforce` grows by keyword-only arguments from here; the positional surface is frozen at 0.9.0.
- The deferred snapshot/golden tests (bundle-schema snapshots, renderer golden files, enforce-signature tests, adapter matrix).
- Semantic versioning from 0.9.0: breaking changes to the evidence-bundle schema or the public API happen only at a major version; **1.0.0 marks the freeze of both.**

Planned beyond the freeze (no dates attached): lightgbm adapter, xgboost version matrix, refit surgery mode, behavioral probe for `involving`, `monitor`, sealed PDF export, NaN-aware transported audit.
