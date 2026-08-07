# Data contracts

## Contract A · `audit` / `enforce`

`X` is numeric (ndarray or DataFrame values); feature names come from the model artifact. Predicates are evaluated in float32 with xgboost's own routing semantics (`x < c` → left), so threshold-boundary rows route identically in the spectrum and the stock engine.

`NaN` anywhere in `X` is a coded refusal (R-07): `audit` returns `DECLINED`, `enforce` raises: the transported audit is not defined for missing-value routing, and the tool says so rather than guessing.

## Contract B · `CertifiedDiscovery`

`X_df` is a DataFrame with mixed dtypes; `y` is 1-D numeric. Numeric columns get threshold atoms, non-numeric columns get one-hot atoms; `NaN` makes every predicate evaluate to `False` (the engine receives NaN natively).

If `y ⊆ {0, 1}` the run is a linear probability model: MSE/Brier primary, AUC also reported; predictions are not clipped to [0, 1].
