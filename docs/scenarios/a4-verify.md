# A4 · Let someone else check, without trusting your code

*Continues the running example from [A1](a1-read.md); the dossier was produced in [A3](a3-cut.md).*

A certificate produced by the same code that performed the surgery proves little to a skeptic. So the checker is a separate distribution, `provable-verify` (numpy + stdlib only), that re-implements from the xgboost JSON spec **its own parser, its own predictor, and its own transport**, and shares zero code with the engine. Independence is machine-checked: `tests/test_firewall.py` fails CI on any verifier import outside {stdlib, numpy} or any import of the engine.

Hand a counterparty `evidence.json`, `model_v2.json`, and `pip install provable-verify`, and they run:

```console
$ python -m provable_verify dossier_enforce/evidence.json model_v2.json
```

```text
provable-verify v0.9.0 (independent; numpy+stdlib; shares no code with provable)
  [PASS] 1 bundle hash = certificate id
  [PASS] 2 model file hash matches bundle
  [PASS] 3 policy hash matches bundle
  [PASS] 4 structural probe (own transport)
  [PASS] 5 behavioral probe (own predictor)
  [PASS] 6 verdict supported by own probes
  [PASS] 7 limits panel present
  measured: structural=5.360e-09  probe=9.546e-09
VERIFICATION: PASS
```

## What a forged certificate looks like

Every field of the bundle is bound by the certificate id (`PV-` + sha256 of the canonical bundle). Shade a single number; say an attacker edits the audit finding's mass down to look smaller:

```python
import json
b = json.load(open('dossier_enforce/evidence.json'))
b['findings'][0]['data_mass'] = 0.0100   # attacker shades the finding down
json.dump(b, open('evidence_tampered.json', 'w'), indent=1, sort_keys=True)
```

```console
$ python -m provable_verify evidence_tampered.json model_v2.json
```

```text
provable-verify v0.9.0 (independent; numpy+stdlib; shares no code with provable)
  [FAIL] 1 bundle hash = certificate id
  [PASS] 2 model file hash matches bundle
  ...
VERIFICATION: FAIL
```

Re-hashing the tampered bundle to dodge check 1 doesn't help either: the verifier's *own* transport and predictor re-measure the model file (checks 4–6), so a resurrected forbidden interaction fails on physics, not bookkeeping: the shipped mutation battery kills each of the seven checks with a targeted tamper, including an attacker-re-hashed bundle ([Tests](../manual/tests.md)).

## Reading it

- Checks 1–3 bind the paperwork: bundle ↔ id, model file ↔ recorded hash, policy ↔ recorded hash.
- Checks 4–5 re-measure the artifact with independent machinery and re-derive the structural and behavioral residuals; note the verifier's *own* measurements printed on its report.
- Check 6 requires the verdict to be supported by those independent probes; check 7 refuses any certificate whose limits panel has been stripped. Honesty about scope is load-bearing, not decorative.

A real dossier of exactly this shape ships in the repository under `dossier_demo/`; verify it in one line right after install.

## What's guaranteed

That the certificate you were handed describes the model file you were handed, as measured by code its author does not control. Verifier independence is a checked property of the release, not a promise.

---

*What the papers add*: the certificate schema, the threat model, and the recheck protocol: Paper 1, [SSRN 7247278](https://doi.org/10.2139/ssrn.7247278) (archived: [Zenodo 21819690](https://doi.org/10.5281/zenodo.21819690)).
