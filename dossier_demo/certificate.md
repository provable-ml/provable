# provable · enforce  —  `PV-04350CAA899E86BD`

## Verdict: **PASS**

**Certified absent:** age × group\_A

This artifact contains no rules combining age × group_A, at any threshold the artifact can express. Verified structurally and behaviorally; independently re-checkable.

| Establishes | Does not establish |
|---|---|
| exact removal of the named rule families from the artifact, priced, with a still-native output file | proxies; fairness of outcomes; causal effects; off-distribution behavior |

## Findings

| clause | atoms | data mass `#exact` | share of model var | where |
|---|---|---|---|---|
| between(age, group\_A) | 299 | 0.113149 | 0.2770 | age>=46.166 (\|c\| 0.977), group\_A>=1 (\|c\| 6.96) |

## Action

| quantity | value |
|---|---|
| mode | excise |
| atoms removed / trees appended | 299 / 299 |
| structural residual `#exact` | 6.200e-09 |
| behavioral grid probe `#exact` | 3.219e-06 |
| stock reload diff on/off rule `#exact` | 1.882e-06 / 0.000e+00 |
| holdout MSE before → after `#est` | 0.0698153 → 0.187852 (Δ +0.118 ± 0.0152) |
| edited artifact | `model\_v2.json` sha256 `0815b0a1535103a9…` |

## The instrument grades itself

| quantity | predicted | observed | status |
|---|---|---|---|
| excision price: train-sample mass vs realized holdout mean squared change | 0.113149 | 0.119611 | within 25% |

## Where the instrument declined

none in this run

## What this does not establish (standing limits)

- Scope is the model artifact, not the world: certified absence of named rule families in the file at the stated hash.
- No proxy claim: other features may jointly reconstruct a protected attribute; this instrument does not test that here.
- No causal claim: prices are model-behavior quantities, not effects of interventions on the world.
- No off-distribution claim: behavior is characterized at the artifact's own thresholds and probed contexts.
- Missing-value routing of ORIGINAL trees is not modeled by the transported spectrum; appended trees route NaN to the fail leaf. Audit assumes NaN-free inputs (R-07 declared).

## Verify without trusting this document

```
python -m provable_verify evidence.json model_v2.json
```

`PV-04350CAA899E86BD` = `PV-` + sha256(canonical bundle)[:16] · `#exact` = arithmetic fact of the artifact · `#est` = estimate with interval · created 2026-07-30T06:12:36+00:00