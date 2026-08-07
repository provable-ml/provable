# Glossary

Every term the scenarios use, in the order you're likely to meet them: each in plain ML-practitioner language first, formal only where it earns its keep. The [manual](manual/exact.md) carries the full mechanics.

## Atoms

An atom is a threshold predicate (`age>=35`) or a conjunction of them (`(age>=35)&(group_A>=1)`). Both workflows speak this one vocabulary. A trained tree ensemble *is* a finite signed sum of atoms (that's the [transport](#transport) identity), and the sieve proposes and certifies claims *about* atoms. One substrate is why a finding in a model ([A2](scenarios/a2-audit.md)) and a finding in the data ([bridge](scenarios/bridge.md)) are comparable objects.

## Transport

The identity that makes workflow A exact. Each leaf of a tree is a box, and a box expands as `1[lo <= x < hi] = 1[x >= lo] - 1[x >= hi]` per feature, telescoped. Apply it to every leaf and the whole ensemble equals `base_score + Σ coef · AND 1[x >= t]`: the atom spectrum. No sampling, no surrogate: the spectrum and the model agree to floating point ([measured](manual/exact.md)).

## FDR and BH(q)

The multiple-testing problem in one sentence: you tested 300 things, so some look good by pure luck, and the more you test the more luck you harvest. The **false discovery rate** is the expected fraction of your *published list* that is flukes. The **Benjamini–Hochberg procedure at level q**, BH(q), is a p-value cutoff rule that controls it: BH(q = 0.10) stamps a list on which at most ~10% of claims are expected to be flukes. Not "each claim is 90% sure": a budget on the *list*. The sieve's whole gate is one BH(q) pass, which is why a 14-claim ledger can honestly contain a fluke or two at the bottom, and why [stability tiers](#stability-tiers) exist to point at them.

## Selection-conditional

The reason your backtest sheet lies: the same data that *chose* the top effects also *graded* them, so the grades are biased up. The sieve's cure is architectural: propose on one half (S1), grade **once** on the reserved half (S2). Every quoted effect, CI, t, and p comes from that single pass on data the selection never saw. S1 is allowed to overfit shamelessly ([B3](scenarios/b3-refusal.md) shows it doing so, harmlessly); S2 is spent exactly once.

## Exact vs est

Two marks appear on everything the instrument emits. `#exact` is an arithmetic fact of the artifact (and sample): structural zeros, masses, prices, bit-identity. No error bar exists because nothing was estimated. `#est` is a statistical estimate, always carrying its interval or its disclosed proxy. The discipline is that the marks never blur: a certificate will print `stock reload diff: 0.0 #exact` and `holdout delta: +0.18 ± 0.019 #est` on adjacent lines and mean both.

## Data mass

The exact associational price of a rule family on a sample: the mean of the removed component squared. Quoted from the spectrum *before* any edit ([A3](scenarios/a3-cut.md)), realized after. It is uncentered and families overlap, so `share_of_var` figures are sizes, not a partition: they can exceed 100% ([A2](scenarios/a2-audit.md) shows 163%, meaning the rest of the model partially cancels that family).

## Verdicts and refusal

Every operation returns one of a closed set (`PASS`, `FINDINGS`, `DECLINED`), and refusals are *coded*, R-01 through R-08 ([full table](manual/verdicts.md)). The stance: never a crash, never a shrug, never a decorated answer where no answer is warranted. `k = 0`, nothing certifiable at the agreed FDR budget, is R-01, and [B3](scenarios/b3-refusal.md) is the argument that it's a finding.

## Vocabulary and nameability

A ledger can only certify what its dictionary can *say*. The default dictionary is `>=` cuts, one-hots, and pairwise conjunctions; v0.2 flags add `<=` twins, interior bands, and engine-harvested paths ([manual](manual/sieve.md)). [B2](scenarios/b2-vocab.md) shows the difference between a truth *certified in fragments* and the same truth *named*. Nameability is separate from the guarantee: FDR control is identical under any vocabulary.

## Toll

What honesty costs: the accuracy you give up by only using effects that certify, versus fitting freely. Paper 2 bounds it above and below; the report prints plug-in versions of both (labeled proxies, `#est`). Its diagnostic use: if the engine's lead over the certified model fits under the toll cap, the gap is the lawful price of certification: pay it or collect more data. If the lead *exceeds* the cap ([B4](scenarios/b4-darkmass.md): 0.400 vs 0.095), the problem isn't honesty, it's [vocabulary](#vocabulary-and-nameability).

## Penumbra

The evidence band: atoms with real but sub-certification signal (1 < |t| ≤ the pool bar). Mass there is "refundable": it can convert to accuracy through the claim-free halo without ever becoming a claim. The report prints its mass and whether the refund was observed.

## Two-tier

The predictor that keeps guarantee and accuracy separate: certified core (claims, FDR-controlled) plus a claim-free halo fitted on the residual (ridge or soft-threshold, capacity set by a disclosed rule, or declined outright when the residual spectrum is consistent with noise). Using the two-tier predictor never touches the ledger's guarantee: the halo asserts nothing at atom level.

## Dark mass

Variance the certified frame cannot see, sized by what the engine *can*. The ledger splits Var(y) into **legible** (certified), **dark-to-certification-but-visible-to-engine** (the engine's lead: structure real enough to convert to accuracy, unnameable at this vocabulary), and **floor** (engine MSE: noise plus dark to both). [B4](scenarios/b4-darkmass.md) constructs a world that is mostly dark and reads the itemization. Paper 3 proves the ceiling: how much signal *could* hide from every scan.

## Phases

The report's one-line regime classification. **I**, refusal competitive or k = 0: nothing worth certifying, and nothing lost by saying so. **II-a**, engine ahead, but the frame-blind margin M_C > 0: certification beats any frame-blind method on these atoms. **II-b**, engine ahead and M_C ≤ 0. **III-like**, the certified core is ahead outright on test. The phase line always shows its work: engine lead vs toll cap, so the [two-diagnoses](scenarios/b4-darkmass.md) call is on the page.

## Stability tiers

Optional B split-replicates re-run the whole propose-and-certify pipeline; each primary claim then carries `f_A` (how often *this exact atom* re-certified) and `f_F` (how often *its near-duplicate family* did). **Invariant**: the identity replicates. **Span-stable**: the family replicates; the exact threshold or representative wobbles. **Fragile**: neither reliably does. Tiers rewrite each claim down to the precision the evidence supports: [bridge](scenarios/bridge.md) has a claim at `f_A = 0.000`, certified once and never again, which is exactly the ~10% the FDR budget priced in, caught and labeled.

## Certificate id

`PV-` + sha256 of the canonical evidence bundle. The id *is* the hash, so any single-field edit breaks it ([A4](scenarios/a4-verify.md) demonstrates), and order of fields cannot matter. The bundle also carries a non-removable limits panel (what the certificate does **not** establish); its absence is itself a verification failure.
