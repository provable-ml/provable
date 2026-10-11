# provable

Exact audit, surgery, certified discovery, and the certificate layer for tree ensembles.

`provable` treats a fitted model as an auditable artifact. On xgboost models it reads an
**exact** itemization of every rule the model uses (the transported atom spectrum),
checks it against a declared policy, removes forbidden rule families by **appended-tree
surgery whose output is still a valid native model file**, and emits a tamper-evident
certificate dossier that a third party can re-check with an independent, numpy-only
verifier that shares no code with the engine. On raw data it runs the certified-discovery
sieve: an FDR-controlled claims ledger, a dark-mass ledger, and the right to refuse
(k = 0 is a verdict, not an error).

The certificate layer keeps a book of what was certified, on which rows, at what price. Its
mediated holdout seals rows into custody and answers questions about them one at a time,
each answer priced in bits by the holdout's own log, so a certificate taken after looking
still holds at its stated level. A watch follows deployed rows against a certified level,
and a route screens a choice on the training rows before it reaches the holdout.

- **Tests that ship with the code:** 242 pytest tests, among them the 173 locks of the certificate layer, the 17-item acceptance ledger (`tests/selftest.py`, T1-T7), and a verifier mutation battery.
- **Independent verification:** every audit and enforce certificate is re-checked by `provable-verify` - numpy + stdlib only, zero shared code with the engine, machine-enforced in CI. A holdout's certificates are recomputed from its book by `audit`.
- **Reproducibility:** the archived version 0.9.0 of this package (DOI 10.5281/zenodo.21819947) reproduces every number in the papers.
- **Version 0.12**, the first release on PyPI since 0.9.0 (0.10 and 0.11 were development milestones). The 0.9.0 verification surface is frozen inside it; it adds the certificate layer (Scenarios C1-C23 in the documentation). It becomes **1.0.0 when the evidence-bundle schema and the public API freeze** (see Road to 1.0.0).

```
            ┌─ WORKFLOW A: audit a model ────────────────────────────────────────┐
model.json ─► load ────► transported atom spectrum (exact, leaf-readable)        │
Policy ─────► audit ───► verdict + findings: masses, shares, threshold locations │
            │ enforce ─► appended-tree surgery ─► model_v2.json (still native)   │
            │            + dossier: evidence.json · certificate.md · MANIFEST   │
            │ verify ──► independent re-check (own parser, own predictor,        │
            │            own transport; zero shared code; numpy + stdlib only)   │
            └────────────────────────────────────────────────────────────────────┘
            ┌─ WORKFLOW B: interrogate data ─────────────────────────────────────┐
X, y ───────► discover (certified sieve) ─► FDR claims ledger · two-tier         │
            │ predictor · toll/penumbra diagnostics · dark-mass ledger ·         │
            │ phase verdict (I / II-a / II-b / III) · or refusal at k = 0        │
            └────────────────────────────────────────────────────────────────────┘
            ┌─ WORKFLOW C: certify on a holdout ─────────────────────────────────┐
holdout ────► seal ────► ask (each answer priced in bits) ─► open ─► certificate │
            │ watch ───► deployed rows against the certified level               │
            │ route ───► VAULT or FORWARD, screened on the training rows         │
            └────────────────────────────────────────────────────────────────────┘
```

Workflows A and B share one substrate — the **atom schema** (threshold predicates and
their conjunctions) — so a violation found in a model (A) can be interrogated in the
data (B): *is this pattern a property of your model, or of your world?* Workflow C needs
only a model's predictions, so it takes any model.

---

## One repo, two distributions

| Path | Distribution | What it is | Depends on |
|---|---|---|---|
| `src/provable/` | `provable` | the instrument: load / spectrum / audit / enforce / discover, and the certificate layer | numpy ≥ 1.26, pandas ≥ 2.2, scipy ≥ 1.13, scikit-learn ≥ 1.5, statsmodels ≥ 0.14, xgboost ≥ 2.0, tabulate, plotly, qrcode |
| `verify/src/provable_verify/` | `provable-verify` | the independent checker examiners run | **numpy only** |

The verifier re-implements from the xgboost JSON spec its own parser, its own
vectorized tree-walk predictor, and its own transport expansion. Independence is
machine-checked: `tests/test_firewall.py` fails CI on any verifier import outside
{stdlib, numpy} or any import of the engine.

The checker also ships standalone at `github.com/provable-ml/provable-verify`, a byte-identical mirror of `verify/`; changes land in this repository first and are mirrored there.

---

## Install

From PyPI:

```
pip install provable
pip install provable-verify
```

Working from a clone (editable installs of both distributions):

```
pip install -e .            # provable (engine): pulls numpy, pandas, scipy, scikit-learn,
pip install -e ./verify     # provable-verify (checker): numpy only     statsmodels, xgboost, tabulate, plotly, qrcode
```

After this, `import provable` works anywhere in the venv and the verifier runs as
`python -m provable_verify <evidence.json> <model.json>` from any directory (file
arguments are relative to your cwd -- the demo dossier lives in `dossier_demo/`).

Tested on Python 3.10 to 3.14: at the minimum versions (numpy 1.26.4, pandas 2.2.3, scipy
1.13.1, scikit-learn 1.5.2, statsmodels 0.14.4) with xgboost 2.0.3, 2.1.4 and 3.0.5, and at
current releases with xgboost 3.4. CI runs the same matrix. The verifier needs numpy only.

---

## Quickstart A: audit, enforce, verify

```python
import xgboost as xgb
from provable import audit, enforce, Policy, between

policy = Policy([between('age', 'group_A')], name='fair_lending_demo')

# Audit an EXTERNAL artifact you did not train
a = audit('model.json', policy, X_val, out_dir='dossier_audit')
a['verdict']['state']            # 'PASS' | 'FINDINGS' | 'DECLINED'
a['findings']                    # per clause: n_atoms, exact data mass, share of
                                 # model variance, threshold locations

# Remove the family; output is a still-valid native model
b = enforce('model.json', policy, X_train, X_holdout, 'model_v2.json',
            y_holdout=y_holdout, out_dir='dossier_enforce')

m2 = xgb.Booster(); m2.load_model('model_v2.json')   # loads in STOCK xgboost
```

```
$ python -m provable_verify dossier_enforce/evidence.json model_v2.json
  [PASS] 1 bundle hash = certificate id
  [PASS] 2 model file hash matches bundle
  [PASS] 3 policy hash matches bundle
  [PASS] 4 structural probe (own transport)
  [PASS] 5 behavioral probe (own predictor)
  [PASS] 6 verdict supported by own probes
  [PASS] 7 limits panel present
VERIFICATION: PASS
```

**Data contract (A).** `X` is numeric (ndarray or DataFrame values); feature names come
from the model artifact. Predicates are evaluated in float32 with xgboost's own routing
semantics (`x < c` → left), so threshold-boundary rows route identically in the spectrum
and the stock engine. `NaN` anywhere in `X` is a coded refusal (R-07): `audit` returns
DECLINED, `enforce` raises — the transported audit is not defined for missing-value
routing, and the tool says so rather than guessing.

## Quickstart B: certified discovery

```python
from provable import CertifiedDiscovery

cd = CertifiedDiscovery(stability_reps=8, seed=0)
report = cd.run(X_df, y)                    # X_df: DataFrame (mixed dtypes), y: 1-D numeric
cd.plot()                                   # 4-panel plotly figure
cd.claims_                                  # certified claims ledger (DataFrame)
p = cd.predict(X_new, model='two_tier')     # 'two_tier' | 'certified' | 'xgb'
```

**Data contract (B).** Numeric columns get threshold atoms, non-numeric columns get
one-hot atoms; `NaN` makes every predicate evaluate to `False` (the engine receives NaN
natively). If `y ⊆ {0, 1}` the run is a linear probability model: MSE/Brier primary,
AUC also reported; predictions are not clipped to [0, 1].

## Quickstart C: certify on a holdout

```python
import provable as pv
from provable import certificates as ct

S = ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=train.index))   # training rows
H = ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=hold.index))    # holdout rows
ho = pv.seal(hold[feats], hold["y"], "book", lineage=H,
             baseline=train["y"].mean(), baseline_lineage=S)
ho.ask(model, lineage=S)            # ACCEPT or REJECT, priced in bits
cert = ho.open(model, lineage=S)    # the certificate, on the exam slice
```

**Data contract (C).** `model` is any callable from a DataFrame of features to
predictions; `panel` carries the clock columns `start_utc` and `end_utc`, which
`slice_from_frame` reads. The loss is squared error by default. With a declared
`loss_range`, or a bounded loss such as Brier, the certificate is exact (the betting
bound); without a range it uses the Gaussian bar and says so in its claim. Scenarios C14
to C23 in the documentation walk the protocol.

---

## The dossier

Every `audit`/`enforce` run with `out_dir` emits:

```
dossier/
├── evidence.json     canonical bundle -- the source of truth; certificate id =
│                     'PV-' + sha256(canonical bundle)[:16]
├── certificate.md   human rendering (GitHub-renderable markdown) bound to the same bundle
└── (enforce)         model_v2.json, MANIFEST.sha256, HOW_TO_VERIFY.txt in the demo
```

Marks used everywhere: `#exact` — an arithmetic fact of the artifact (structural zeros,
masses, prices; no error bar exists). `#est` — a statistical estimate, always with its
interval. The limits panel ("what this does not establish": proxies, causal claims,
off-distribution behavior, NaN routing) is furniture, not fine print: every bundle
carries it and verifier check 7 fails if it is stripped.

## Verdict grammar

| Verdict | Meaning |
|---|---|
| `PASS` | certified absence of every forbidden pattern, within stated (float32 write-back) tolerance |
| `FINDINGS` | violations exist — quantified, localized, priced |
| `DECLINED` | the instrument refuses, with a coded reason — never a crash, never a shrug |

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

---

## What "exact" means here (workflow A)

A fitted ensemble on continuous inputs is already a Boolean object on its own split
thresholds. Each leaf box expands exactly:

```
1[lo <= x < hi] = 1[x >= lo] - 1[x >= hi]        (per feature; telescoped over the box)
```

so the whole artifact equals `base_score + Σ coef · AND_f 1[x_f >= t_f]` — a finite,
leaf-readable, signed rule spectrum with no sampling and no approximation. Audit
families, prices, surgery, and both certification probes operate on this identity.

| Stage | Mechanics | Where |
|---|---|---|
| load / parse | native JSON → leaf boxes (interval constraints per feature); base_score parsed to the empty atom | `adapters_xgb` |
| transport | boxes → signed threshold atoms; guarded by `max_atoms` (R-03 above it) | `calculus.Spectrum` |
| audit | family selection (`between`, `involving`) → exact data mass, share of model variance, per-threshold profile | `surface.audit` |
| price | excision price = mean squared removed component on the sample (`#exact`), quoted **before** the edit | `calculus.data_mass` |
| enforce | one appended chain-tree per removed atom (−coef on the conjunction, 0 elsewhere); off-rule rows **bit-identical** by construction | `adapters_xgb.append_rules` |
| structural probe | re-transport the edited file; max residual forbidden coefficient | `surface.enforce` + verifier check 4 |
| behavioral probe | consecutive-cell second differences over the full (i, j) threshold grid at probe contexts; zero everywhere ⇔ no i×j interaction at the artifact's own resolution | `calculus.grid_probe` + verifier check 5 |
| certificate | canonical bundle, hash-as-id, verdict, limits, self-grading panel (predicted price vs realized holdout change) | `evidence` |

Measured on the shipped acceptance ledger (`tests/selftest.py`, 17/17, xgboost 3.3.0,
200-tree depth-4 external artifact, 4,676 atoms):

| Identity | Measured |
|---|---|
| transport spectrum == stock margin | 2.4e-06 max abs |
| transport == reference Möbius calculus (binary model) | 2.2e-16 max abs |
| off-rule rows after surgery, stock reload | 0.0 (bit-identical) |
| on-rule rows vs functional edit | 1.9e-06 max abs |
| structural residual after excising 299 atoms | 6.2e-09 |
| behavioral grid probe on the edited file | 3.2e-06 |
| excision price predicted vs realized (holdout) | 0.1131 vs 0.1196 (within 25% band, printed on the certificate) |

## The sieve (workflow B)

```
raw X, y
  ├─ engine:      XGBoost on raw features (early-stopped on a carve of the training rows)
  └─ instrument:  atom dictionary ─► RS(ν,R) proposal on S1 ─► forward core ─► one OLS
                  pass on S2 + BH(q) ─► certified model ─► claim-free halo ─► two-tier
diagnostics:      penumbra · toll bounds · spotlight margin · completion · phase · ledger
report:           console tables + plotly figure + machine-readable dict + claims DataFrame
```

The gate is one selection-conditional OLS pass on the reserved half S2 with BH(q) over
partial-t p-values; every quoted effect, CI, t, p comes from that single pass. The halo
is claim-free (using the two-tier predictor never touches the ledger's guarantee). The
dark-mass ledger splits Var(y) into legible (certified), dark-to-certification-but-
visible-to-engine, and engine MSE (noise + dark to both). Phases: I (refusal
competitive or k = 0), II-a (engine ahead, frame-blind margin M_C > 0), II-b (engine
ahead, M_C ≤ 0), III-like (certified core ahead on test).

Key parameters (defaults): `fdr_q=0.10`, `rounds=4`, `entry_t=2.0`, `entry_cap=300`,
`core_cap=120`, `core_tmin=1.0`, `n_cuts=8`, `cart_cuts=True`, `n_harvest=40`,
`interactions=True` (shadow rule `conj_shadow=0.9`), `prune_support=0.005`,
`prune_dup_r=0.98`, `halo='auto'`, `refit_scope='train'`, `test_frac=0.25`,
`s2_frac=0.5`, `stability_reps=0`, `xgb_params=None` (600 trees, lr .05, depth 6,
hist, early stop 50), `seed=0`. Full table with meanings in the class docstring.

**v0.2 vocabulary extensions** (all default-off; when off, the run — claims, report
dict, printed report — is byte-identical to v0.1): `le_cuts=False` mirrors every ≥ cut
as a ≤ twin, making low-side cells nameable and claimable (the duplicate prune switches
from |r| to signed r so exact complements survive); `band_conj=False` admits same-column
`(x>=a)&(x<=b)` pairs, making interior bands first-class atoms; `n_path_harvest=0` with
`path_depth=3` harvests top-gain root-subpath conjunctions from the fitted engine's own
trees (Yes branch → ≤; numeric features only; standard leaf grammar), diversified by
`path_per_set=3` so dominant structures cannot monopolize the harvest; `core_dedup_r=None`
applies the dictionary's duplicate principle at core admission — one representative per
near-duplicate family — without which a densified vocabulary floods the core with shadows
whose S2 partial t's split and BH kills the whole family.

---

## Tests

```
python tests/test_firewall.py     # verifier imports ⊆ stdlib+numpy, zero engine imports
python -m pytest tests/           # 242 tests: contracts, protocol, mutations, certificate layer
python tests/selftest.py          # T1-T7 acceptance ledger (17/17)
```

| Suite | Locks |
|---|---|
| `test_contracts.py` | the identities certificates stand on: transport == stock margin across model shapes (depth 1–6, incl. single-split and constant models), float32 boundary routing at exact thresholds, base_score paths, append-tree on/off semantics |
| `test_protocol.py` | refusal codes (R-05, coded R-07), policy canonicalization, order-free bundle hashing, any-single-field mutation breaks the certificate id, limits always present and rendered, verdict grammar closed set, sieve determinism per seed; v0.2 vocabulary: flags-off schema stability, deterministic low-side (\u2264) claim recovery with the extended dictionary |
| `test_verifier_mutations.py` | each of the verifier's seven checks individually killed by a targeted tamper, incl. a resurrected-interaction attack with an attacker-re-hashed bundle |
| `selftest.py` | end-to-end: external artifact → audit → enforce → stock reload → independent verify → mutation → discover → refusal |
| `test_twin_suite.py` | the certificate layer, one lock per stated property (173 locks): lineage and the book, certify and grade, the mediated holdout and its bits, the watch, the route; each lock prints the value it measured |

Deferred by design: bundle-schema snapshots and renderer golden files (reporting layer),
and a lightgbm adapter. xgboost 2.0.3, 2.1.4, 3.0.5 and the current release run in CI, and
lock TH1 holds the released signatures.

## Road to 1.0.0

- A `schema_version` field in the evidence bundle, honored by the verifier.
- `enforce` grows by keyword-only arguments from here; the positional surface is frozen at 0.9.0 (lock TH1).
- The deferred snapshot/golden tests above.
- Semantic versioning from 0.9.0: breaking changes to the evidence-bundle schema or the public API happen only at a major version; 1.0.0 marks the freeze of both.

## Guarantees, estimates, and caveats

- **Controlled:** the FDR of the claims ledger, by BH(q) over a single
  selection-conditional S2 pass. **Exact:** every `#exact` quantity is an arithmetic
  fact of the artifact (and sample) — structural zeros, masses, prices, off-rule
  bit-identity.
- **Float32 is the write-back resolution:** appended-tree coefficients are stored as
  float32 by the native format; certification tolerances (structural ≤ 1e-4, behavioral
  ≤ 1e-3) state this rather than hide it. Measured values are orders of magnitude below
  the gates.
- **Estimates are labeled:** holdout deltas carry CIs; the sieve's penumbra, toll
  bounds, spotlight, and M_C are plug-in quantities with disclosed proxies; ridge* is a
  hindsight benchmark for the frame-blind class, not a deployable model.
- **The limits panel is non-removable:** no proxy claim, no causal claim, no
  off-distribution claim, no fairness-of-outcomes claim — certified absence of named
  rule families in the file at the stated hash, and nothing more.
- **`refit_scope='train'`** re-uses S2 in the sieve's refit and carries a residual
  selection-optimism term (measured small); `'s2'` avoids it; both rows print.
- **Verifier independence is a checked property,** not a promise (see Tests).

## Not in this release

lightgbm adapter; sealed PDF export (the HTML certificate binds to the same bundle);
behavioral probe for `involving` clauses (structural only); refit surgery mode (excise
only); vertical report packs; NaN-aware transported audit (R-07 declared instead).

## Reproducibility

All stochastic steps are seeded (`seed`; house conventions: split rng 4000+seed, engine
9000+seed, stability replicates 31000+b). Given a seed, repeated runs reproduce the
same report and the same certificate id up to timestamps. Runtime reference (single
container CPU): full acceptance ledger ≈ 8 s; independent verification of the demo
dossier ≈ 5 s; sieve on Adult (45,222 rows, P = 235, B = 8) ≈ 10 s.

## Files

```
src/provable/                  adapters_xgb · calculus · policy · evidence · surface · sieve ·
                               certificates/ (lineage, the book, the transcript kit) · holdout ·
                               monitor (watch) · transport (route) · facade · inquiry · loop ·
                               purchase/
verify/src/provable_verify/    the independent checker (single module)
tests/                         firewall · contracts · protocol · verifier mutations · selftest ·
                               the certificate layer's locks
docs/                          the site: manual, glossary, scenarios and their demo scripts
validation/                    simulation records behind the documentation
dossier_demo/                  a real enforce dossier: evidence.json, certificate.md,
                               model.json, model_v2.json, MANIFEST.sha256, HOW_TO_VERIFY.txt
```

## Citing

`CITATION.cff` ships in this repository. BibTeX:

```bibtex
@software{provable2026,
  author  = {Souihli, Oussama},
  title   = {provable: Exact audit, surgery, and certified discovery for tree ensembles},
  year    = {2026},
  version = {0.9.0},
  doi     = {10.5281/zenodo.21819947},
  url     = {https://provable.ml}
}

@misc{souihli2026calculus,
  author = {Souihli, Oussama},
  title  = {Every Interaction Has a Price: An Exact Calculus for Tree Ensembles},
  year   = {2026},
  doi    = {10.5281/zenodo.21819690}
}

@misc{souihli2026instrument,
  author = {Souihli, Oussama},
  title  = {Certified Discovery on Real Grams: an Instrument, its Limits, and its Laws},
  year   = {2026},
  doi    = {10.5281/zenodo.21819895}
}

@misc{souihli2026metrology,
  author = {Souihli, Oussama},
  title  = {Dark-Mass Metrology: Pricing What a Certified Model Does Not Know},
  year   = {2026},
  doi    = {10.5281/zenodo.21819925}
}
```

License: MIT.


The site under `docs/` builds with MkDocs, which is deliberately not a package dependency: `uv run --with mkdocs-material mkdocs serve` resolves the docs tooling at run time.
