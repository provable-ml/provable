# The sieve (workflow B): pipeline, parameters, vocabulary

```
raw X, y
  ├─ engine:      XGBoost on raw features (early-stopped on a carve of the training rows)
  └─ instrument:  atom dictionary ─► RS(ν,R) proposal on S1 ─► forward core ─► one OLS
                  pass on S2 + BH(q) ─► certified model ─► claim-free halo ─► two-tier
diagnostics:      penumbra · toll bounds · spotlight margin · completion · phase · ledger
report:           console tables + plotly figure + machine-readable dict + claims DataFrame
```

The gate is one selection-conditional OLS pass on the reserved half S2 with BH(q) over partial-t p-values; every quoted effect, CI, t, p comes from that single pass. The halo is claim-free (using the two-tier predictor never touches the ledger's guarantee). The dark-mass ledger splits Var(y) into legible (certified), dark-to-certification-but-visible-to-engine, and engine MSE (noise + dark to both). Phases: I (refusal competitive or k = 0), II-a (engine ahead, frame-blind margin M_C > 0), II-b (engine ahead, M_C ≤ 0), III-like (certified core ahead on test).

## Key parameters (defaults)

`fdr_q=0.10`, `rounds=4`, `entry_t=2.0`, `entry_cap=300`, `core_cap=120`, `core_tmin=1.0`, `n_cuts=8`, `cart_cuts=True`, `n_harvest=40`, `interactions=True` (shadow rule `conj_shadow=0.9`), `prune_support=0.005`, `prune_dup_r=0.98`, `halo='auto'`, `refit_scope='train'`, `test_frac=0.25`, `s2_frac=0.5`, `stability_reps=0`, `xgb_params=None` (600 trees, lr .05, depth 6, hist, early stop 50), `seed=0`. Full table with meanings in the class docstring.

Binary `y ⊆ {0, 1}` runs as a linear probability model (the paper's Adult convention): MSE/Brier is the primary metric, AUC is also reported; predictions are not clipped to [0, 1].

## v0.2 vocabulary extensions

All default-off; when off, the run (claims, report dict, printed report) is byte-identical to v0.1:

- **`le_cuts=False`** mirrors every `>=` cut as a `<=` twin, making low-side cells nameable and claimable (the duplicate prune switches from |r| to signed r so exact complements survive).
- **`band_conj=False`** admits same-column `(x>=a)&(x<=b)` pairs, making interior bands first-class atoms ([B2](../scenarios/b2-vocab.md) is the demonstration).
- **`n_path_harvest=0`** with `path_depth=3` harvests top-gain root-subpath conjunctions from the fitted engine's own trees (Yes branch → ≤; numeric features only; standard leaf grammar), diversified by `path_per_set=3` so dominant structures cannot monopolize the harvest.
- **`core_dedup_r=None`** applies the dictionary's duplicate principle at core admission (one representative per near-duplicate family), without which a densified vocabulary floods the core with shadows whose S2 partial t's split and BH kills the whole family ([B1's missing x23](../scenarios/b1-claims.md#where-is-x23) is that mechanism photographed in the wild).

## Gate basis

**`cert_basis_tol=1e-6`** (keyword-only, on by default). The partial-variance guard conditions the forward core on S1 only, so a member admitted there can be constant on S2, or an exact combination of the intercept and other members there: a `<=` twin whose tie rows all fell in S1, one-hots of a category whose remaining level fell in S1, a cut and a conjunction that coincide on S2. Such a member has no identified partial effect on S2. Before the single pass, walking the core in admission order, a member is dropped when its norm after projection on the intercept and the kept members falls below that share of its own centered norm. A dropped member is not tested. The BH family, the certificate's family record and the selective intervals count the kept members (`core_`). The drop is reported with the kept member nearest to it: `cert_dropped_` and `core_full_` on the object, a `gate_basis` entry in the report's certification block and a table in the printed report, each present only when a member was dropped. When the core has full column rank on S2 the run (claims, report dict, printed report) is byte-identical to 0.12.0. `cert_basis_tol=None` restores the unreduced pass.

## Reproducibility

All stochastic steps are seeded (`seed`; house conventions: split rng 4000+seed, engine 9000+seed, stability replicates 31000+b). Given a seed, repeated runs reproduce the same report and the same certificate id up to timestamps. Runtime reference (single container CPU): full acceptance ledger ≈ 8 s; independent verification of the demo dossier ≈ 5 s; sieve on Adult (45,222 rows, P = 235, B = 8) ≈ 10 s.
