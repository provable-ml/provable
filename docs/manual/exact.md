# What "exact" means here (workflow A)

A fitted ensemble on continuous inputs is already a Boolean object on its own split thresholds. Each leaf box expands exactly:

```
1[lo <= x < hi] = 1[x >= lo] - 1[x >= hi]        (per feature; telescoped over the box)
```

so the whole artifact equals `base_score + Σ coef · AND_f 1[x_f >= t_f]`: a finite, leaf-readable, signed rule spectrum with no sampling and no approximation. Audit families, prices, surgery, and both certification probes operate on this identity.

| Stage | Mechanics | Where |
|---|---|---|
| load / parse | native JSON → leaf boxes (interval constraints per feature); base_score parsed to the empty atom | `adapters_xgb` |
| transport | boxes → signed threshold atoms; guarded by `max_atoms` (R-03 above it) | `calculus.Spectrum` |
| audit | family selection (`between`, `involving`) → exact data mass, share of model variance, per-threshold profile | `surface.audit` |
| price | excision price = mean squared removed component on the sample (`#exact`) -- the removal cost -- quoted **before** the edit | `calculus.data_mass` |
| enforce | one appended chain-tree per removed atom (−coef on the conjunction, 0 elsewhere); off-rule rows **bit-identical** by construction | `adapters_xgb.append_rules` |
| structural probe | re-transport the edited file; max residual forbidden coefficient | `surface.enforce` + verifier check 4 |
| behavioral probe | consecutive-cell second differences over the full (i, j) threshold grid at probe contexts; zero everywhere ⇔ no i×j interaction at the artifact's own resolution | `calculus.grid_probe` + verifier check 5 |
| certificate | canonical bundle, hash-as-id, verdict, limits, self-grading panel (predicted price vs realized holdout change) | `evidence` |

Measured on the shipped acceptance ledger (`tests/selftest.py`, 17/17, xgboost 3.3.0, 200-tree depth-4 external artifact, 4,676 atoms):

| Identity | Measured |
|---|---|
| transport spectrum == stock margin | 2.4e-06 max abs |
| transport == reference Möbius calculus (binary model) | 2.2e-16 max abs |
| off-rule rows after surgery, stock reload | 0.0 (bit-identical) |
| on-rule rows vs functional edit | 1.9e-06 max abs |
| structural residual after excising 299 atoms | 6.2e-09 |
| behavioral grid probe on the edited file | 3.2e-06 |
| excision price predicted vs realized (holdout) | 0.1131 vs 0.1196 (within 25% band, printed on the certificate) |
