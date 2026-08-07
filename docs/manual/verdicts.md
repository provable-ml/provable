# Verdicts, refusal codes, and the dossier

## Verdict grammar

| Verdict | Meaning |
|---|---|
| `PASS` | certified absence of every forbidden pattern, within stated (float32 write-back) tolerance |
| `FINDINGS` | violations exist: quantified, localized, priced |
| `DECLINED` | the instrument refuses, with a coded reason; never a crash, never a shrug |

The set is closed and machine-checked (`tests/test_protocol.py`).

## Refusal codes

| Code | Refusal |
|---|---|
| R-01 | nothing certifiable at the agreed FDR budget (k = 0) |
| R-02 | reserved certification split too small |
| R-03 | expansion above exact-enumeration cap |
| R-04 | pattern off the representable lattice |
| R-05 | feature not present in the model |
| R-06 | probe budget exceeded |
| R-07 | missing values present; transported audit not defined |
| R-08 | unsupported clause kind |

## The dossier

Every `audit`/`enforce` run with `out_dir` emits:

```
dossier/
├── evidence.json     canonical bundle -- the source of truth; certificate id =
│                     'PV-' + sha256(canonical bundle)[:16]
├── certificate.md    human rendering (GitHub-renderable markdown) bound to the same bundle
├── certificate.html  human rendering with the verdict seal, same bundle
└── (enforce)         model_v2.json; MANIFEST.sha256 and HOW_TO_VERIFY.txt in the demo
```

Marks used everywhere: `#exact` is an arithmetic fact of the artifact (structural zeros, masses, prices; no error bar exists); `#est` is a statistical estimate, always with its interval.

The limits panel ("what this does not establish": proxies, causal claims, off-distribution behavior, NaN routing) is furniture, not fine print: every bundle carries it and [verifier check 7](../scenarios/a4-verify.md) fails if it is stripped. The standing limits, verbatim from the certificate:

- Scope is the model artifact, not the world: certified absence of named rule families in the file at the stated hash.
- No proxy claim: other features may jointly reconstruct a protected attribute; this instrument does not test that here.
- No causal claim: prices are model-behavior quantities, not effects of interventions on the world.
- No off-distribution claim: behavior is characterized at the artifact's own thresholds and probed contexts.
- Missing-value routing of ORIGINAL trees is not modeled by the transported spectrum; appended trees route NaN to the fail leaf. Audit assumes NaN-free inputs (R-07 declared).
