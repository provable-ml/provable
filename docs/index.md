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

## Workflow C · certify a claim

You want a claim others can check, the next column worth buying, and a holdout you can look at without fooling yourself.

| The question | Where it's answered |
|---|---|
| I ran a study; make the claim citable. | [C1 · Certify](scenarios/c1-certify.md) |
| Can I use this certificate again? | [C2 · Reuse](scenarios/c2-reuse.md) |
| What does my book of claims guarantee? | [C3 · Book](scenarios/c3-book.md) |
| Price what that old pipeline did. | [C4 · Lint](scenarios/c4-lint.md) |
| Did it work? | [C5 · Grade](scenarios/c5-grade.md) |
| Show me on a real public dataset. | [C6 · Titanic](scenarios/c6-titanic.md) |
| More rows won't help; what column would? | [C7 · Purchase](scenarios/c7-purchase.md) |
| What is my model still getting wrong, kept as a record? | [C8 · Shadow](scenarios/c8-shadow.md) |
| Interrogate the thing I cannot name. | [C9 · Characterize](scenarios/c9-characterize.md) |
| A vendor claims to have the column. Test the claim first. | [C10 · Screen](scenarios/c10-screen.md) |
| Build everything the archive can hold, and see how fast. | [C11 · Alternate](scenarios/c11-alternate.md) |
| Several priced columns, one requirement. In which order, and when to stop? | [C12 · Pursue](scenarios/c12-pursue.md) |
| Forty columns I own, five that matter. Find them, and see why stepwise cannot. | [C13 · Select](scenarios/c13-select.md) |
| I want to look at the holdout while I tune. What does it cost? | [C14 · Seal](scenarios/c14-seal.md) |
| No exam slice. Can I certify on the rows I tuned on? | [C15 · Transcript](scenarios/c15-transcript.md) |
| I need a number from the holdout, not a yes or no. | [C16 · Check](scenarios/c16-check.md) |
| That result saw the holdout. What is it still worth? | [C17 · Peeked](scenarios/c17-peeked.md) |
| How is the holdout meant to be used, from the seal to reuse? | [C18 · End to end](scenarios/c18-end-to-end.md) |
| The world changed after my training rows. Does looking at the holdout buy a better model? | [C19 · Changed world](scenarios/c19-changed-world.md) |
| Too few training rows to choose among my candidates. Does looking at the holdout help, and what does it cost? | [C20 · Few rows](scenarios/c20-few-rows.md) |
| I kept every feature that improved the holdout score. Is the model better? | [C21 · Looking hurts](scenarios/c21-looking-hurts.md) |
| My model is certified and deployed, and the world drifts. When does the certificate stop describing it? | [C22 · Drift watch](scenarios/c22-drift-watch.md) |
| This choice helps in some weeks and hurts in others. Can I tune it on the holdout? | [C23 · Transport screen](scenarios/c23-transport-screen.md) |

## Questions the papers answer

Some questions are answered by the companion preprints rather than by code: proofs, calibration gauntlets, and laws the instrument obeys. The scenarios point to them where relevant; the full set:

| The question | Where |
|---|---|
| How do we know the claim-machine itself works? (qualified like lab equipment: known truths planted in real data, graded against predictions frozen in advance) | Paper 2 · [SSRN 7247302](https://doi.org/10.2139/ssrn.7247302) (archived: [Zenodo 21819895](https://doi.org/10.5281/zenodo.21819895)) |
| What does honesty cost in accuracy, and is the cost bounded above *and* below? | Paper 2 (the toll theorems); the plug-in bounds print in every sieve report |
| When does the honest model beat the black box outright, and in what data-size regime? | Paper 2 (the phase diagram); the phase verdict prints in every report |
| What did the procedure miss, and why: search effort, data shortage, or geometry? | Paper 3 · [10.5281/zenodo.21819925](https://doi.org/10.5281/zenodo.21819925) |
| What would it cost, in rows of data, to stop missing each one? Which misses can never be bought? | Paper 3 |
| How much true signal could hide from every scan? Is there a hard ceiling? | Paper 3 (the dark-mass formula); the ledger prints in every report |
| The exact identity behind the calculus, and its transport to continuous features | Paper 1 · [SSRN 7247278](https://doi.org/10.2139/ssrn.7247278) (archived: [Zenodo 21819690](https://doi.org/10.5281/zenodo.21819690)) |

Full citations on the [Cite](cite.md) page.

What `provable` is, the identity it rests on, and the first thing to run: [About](about.md).
