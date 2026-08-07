# Pick your question

Every page in this documentation answers one question you might actually bring to a **trained model** or a **dataset**, with runnable code and the real output it produces.

No prior familiarity with multiple testing, FDR, or the papers is assumed: each scenario introduces its terms as it goes, and the [Glossary](glossary.md) collects them in plain language.

Two workflows share one substrate, the [atom](glossary.md#atoms) vocabulary (threshold predicates and their conjunctions), so a violation found in a model (A) can be interrogated in the data (B): *is this pattern a property of your model, or of your world?*

## Workflow A · audit a model

You already have a trained XGBoost model.

| The question | Where it's answered |
|---|---|
| What rules is the model actually using? | [A1 · Read](scenarios/a1-read.md) |
| Does it ever combine `age` with `group_A`? Can I prove it, either way? | [A2 · Audit](scenarios/a2-audit.md) |
| Can I remove exactly those rules, nothing else, without retraining, and know the accuracy cost *before* the edit? | [A3 · Cut](scenarios/a3-cut.md) |
| Can a third party verify the rules are really gone, without trusting my code? | [A4 · Verify](scenarios/a4-verify.md) |
| What can no audit of a trained model ever tell me? | The limits panel, on every certificate; see [A3](scenarios/a3-cut.md) and [Verdicts](manual/verdicts.md) |

## Workflow B · interrogate data

You have raw `X, y` and want findings you can stand behind.

| The question | Where it's answered |
|---|---|
| I screened hundreds of candidate features. Which effects are real? | [B1 · Claims](scenarios/b1-claims.md) |
| The effect lives on the *low* side, or in an *interior band*. Can the claim say so? | [B2 · Vocabulary](scenarios/b2-vocab.md) |
| The run came back empty. Is "nothing found" informative? | [B3 · Refusal](scenarios/b3-refusal.md) |
| The certified model trails XGBoost. Is that the price of honesty, or a fixable gap? | [B4 · Dark mass](scenarios/b4-darkmass.md) |
| When a claim certifies, is it the right feature, or a near-duplicate standing in? | Stability tiers, in [A→B · Bridge](scenarios/bridge.md) and [B1](scenarios/b1-claims.md) |

## The bridge

| The question | Where it's answered |
|---|---|
| The audit found a pattern in my model. Is it in the world, or just in the model? | [A→B · Bridge](scenarios/bridge.md) |

## Questions the papers answer

Some questions are answered by the companion preprints rather than by code in 0.9.0: proofs, calibration gauntlets, and laws the instrument obeys. The scenarios point to them where relevant; the full set:

| The question | Where |
|---|---|
| How do we know the claim-machine itself works? (qualified like lab equipment: known truths planted in real data, graded against predictions frozen in advance) | Paper 2 · [10.5281/zenodo.21819895](https://doi.org/10.5281/zenodo.21819895) |
| What does honesty cost in accuracy, and is the cost bounded above *and* below? | Paper 2 (the toll theorems); the plug-in bounds print in every sieve report |
| When does the honest model beat the black box outright, and in what data-size regime? | Paper 2 (the phase diagram); the phase verdict prints in every report |
| What did the procedure miss, and why: search effort, data shortage, or geometry? | Paper 3 · [10.5281/zenodo.21819925](https://doi.org/10.5281/zenodo.21819925) |
| What would it cost, in rows of data, to stop missing each one? Which misses can never be bought? | Paper 3 |
| How much true signal could hide from every scan? Is there a hard ceiling? | Paper 3 (the dark-mass formula); the ledger prints in every report |
| The exact identity behind the calculus, and its transport to continuous features | Paper 1 · [10.5281/zenodo.21819690](https://doi.org/10.5281/zenodo.21819690) |

Full citations on the [Cite](cite.md) page.

What `provable` is, the identity it rests on, and the first thing to run: [About](about.md).
