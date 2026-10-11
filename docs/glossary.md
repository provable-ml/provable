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

The removal cost. The exact associational price of a rule family on a sample: the mean of the removed component squared, which is what excising the family would cost the model on this data. Quoted from the spectrum *before* any edit ([A3](scenarios/a3-cut.md)), realized after. It is uncentered and families overlap, so `share_of_var` figures are sizes, not a partition: they can exceed 100% ([A2](scenarios/a2-audit.md) shows 163%, meaning the rest of the model partially cancels that family).

## Verdicts and refusal

Every operation returns one of a closed set (`PASS`, `FINDINGS`, `DECLINED`), and refusals are *coded*, R-01 through R-09 ([full table](manual/verdicts.md)). The stance: never a crash, never a shrug, never a decorated answer where no answer is warranted. `k = 0`, nothing certifiable at the agreed FDR budget, is R-01, and [B3](scenarios/b3-refusal.md) is the argument that it's a finding.

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

## Certificate

The citable record `certify` issues: the claim, its guarantee (kind, level, and the family it was earned over), the fingerprinted [lineage](#lineage) of the rows that proposed and graded it, the certified support, a Gram descriptor, and the full machine artifact, under a content-addressed `CD-` id. Consumed certificates' ids sit inside the hashed bytes, so provenance is a tamper-evident chain. Field-by-field reference: [Certificates & the book](manual/certificates.md).

## Certificate kinds

`carrier` names the certified family (one per certified run); `existence` certifies that something is there without naming members; `artifact` is an engine evidence bundle (audit or enforce) unified into the certificate type with an EXACT guarantee, no lineage, and its `PV-` id kept. A conditional kind is reserved for a later release.

## Lineage

Which rows, of which source, over which clock window. A slice fingerprints its source table by content, carries its row ids in-record, and stamps a clock window; a certificate's proposal lineage hereditarily includes everything each consumed certificate touched. Lineage is what makes freshness checkable instead of remembered.

## Freshness

Identity-level: two lineages are fresh for each other when no slice pair on the same source shares a row. Fresh reuse transports a guarantee unchanged; reuse on shared rows does not, and the [admission](#admission) machinery prices exactly that. Cross-source clock overlap raises a dependence flag that reports rather than blocks.

## Admission

The verdict on reusing certificates against new rows: FRESH, FRESH WITH DEPENDENCE FLAG, NOT FRESH (refused by default, or demoted to MISCALIBRATED with the overlap recorded), or [UNKNOWN LINEAGE](#unknown-lineage). An admission also carries the inflation the pattern incurs and the pass level for the next test. Walkthrough: C2.

## UNKNOWN LINEAGE

The verdict for results that predate instrumentation. They are listed in the book's statement, can be talked about, and never bear budget or pass silently: the honest treatment of history you can no longer audit.

## The book

A `Ledger`: an append-only operation log plus a budget state. Its statement quotes one number for the whole book under the additive rule, the sum of active certificate levels, which holds unconditionally; EXACT artifacts and UNKNOWN LINEAGE items are excluded. Walkthrough: C3.

## Online budget

The batch step-up discipline: quote each run's level from the book *before* the run (an imported update rule, cited in the [reference](manual/certificates.md#the-account)), and the union false-discovery proportion stays at the account target at every step. A book whose recorded levels deviate is detected and falls back to the additive rule.

## Registered charge

Fix a finite vocabulary of member and row-set pairs in advance and claim only at p at most alpha over the vocabulary size: familywise level alpha then survives arbitrary reuse, order, and lineage. The one guarantee that re-testing on spent rows cannot break, and the correct home for pre-registered follow-ups.

## Selective intervals

Confidence intervals recalibrated for having been *selected*: on the certified set the level is alpha_sel = khat times q over m (FCR), so the quoted pair `ci_lo_sel, ci_hi_sel` can be wider or narrower than the marginal 95% pair depending on khat, and both directions are correct. [C1](scenarios/c1-certify.md) shows the wide case.

## Corrected penumbra

The evidence-band mass of [B4](scenarios/b4-darkmass.md) minus its estimated null floor (a Storey null count times the exact per-null band mass). What remains is band mass beyond pure noise; with a data-adaptive dictionary part of it is selection mass from the dictionary build, and the report says so in the block's note field.

## Atom card

The est panel beside the claims table: per certified atom, firing rate, marginal when-fired effect with its se, and the agreement rate of sign(y) with the certified direction, on S2 and on held-out test, never on the proposal rows that selected the atoms. Its header is the contract: marginal per-atom effects, net across cells via [core fit](#core-fit). Never budget-bearing.

## Core fit

`rep.core_fit()`: the certified tier's fitted value as a column, the joint netting of every certified cell for a row. The certified tier is one OLS of the target on the certified atoms alone, so overlapping cells are netted by the fit, which per-atom marginals cannot do.

## Grade

`pv.grade`: the scorecard verb. Admission first (spent rows refuse; the level is quoted by the book), then each pre-committed cell is tested one-sided in its certified direction on its firings in the new rows: CONFIRMED, NOT CONFIRMED, or REVERSED, where REVERSED is significant evidence against the original direction, not mere absence of confirmation. Budget-bearing; the entry lands in the grades section of the statement.

## Shadow

`pv.shadow(rep, X, y, clock)`: the archive-shadow record of one certified target on a declared out-of-fit slice — per row, the clock, the outcome, the Report's own certified-class prediction, and the owned residual `e = y - f_D(x)`, whose empirical second moment estimates the answerable aggregate `Var(u) = Mhat + Mbar`. Out-of-fit is asserted by row identity (source fingerprint, and the fitted lineage's clock envelope under the same dataset tag); overlap refuses by name. `Shadow.join` aligns several targets' shadows into a frame on the coarsest shared clock. Free of acquisition: issues nothing, charges nothing, writes to no book.

## Aim block

The direction set a screened candidate is coupled to. With a columns-leg purchase order it is the order's state-whitened feasible-dark block `Psi_hat` with its dimension `J`, rebuilt deterministically from the Report the Shadow consumed and evaluated on the slice in the Report's own encoding; with a declared floor it is the shadow itself, standardized, at `J = 1`, and the Screening says so.

## Archive aim ceiling

`lam_arch` (Def 4.14): the reliability already constructible from declared archive atoms — the archive family's explained share of `||e||^2_n`, on the residual-variance scale, lower-boundable by exhibiting a built candidate. The `characterize` record carries it; the `screen` trichotomy compares the floor to it and to the upper envelope `lam_cap` (its square plus the walled remainder share).

## Floor

`lam_min`: the reliability a candidate column must clear for the declared purpose. Mandatory and single in `screen`: either the guaranteed reliability `lam_A` a columns-leg purchase order priced at its promise (with the reliability-law `lam*` printed beside it), or a declared value carrying its derivation, labelled DECLARED. The verdict band tests the floor one-sided; `m_req` is the row count at which the threshold test at `t* = lam_min^2 + dlam^2` meets the declared miss and clear budgets.

## Holdout

`pv.seal(X, y, ledger, ...)`: rows put in custody in the book before anyone looks. The seal commits salted hashes of their content, their row identities, split, baseline and protocol; afterwards the rows are reached only through the holdout's methods, `ask`, `check`, `open` and `retire`, and every answer goes into a hash-chained log in the book with its price in [bits](#bits-spent). Until `retire` hands released rows back, nothing else the holdout returns depends on the sealed values, and its refusals read only declarations, the public lineage and the public log. Custody is API-level: the library offers no other door, the book refuses a certificate on rows in custody that the holdout did not issue, and physical custody of the files is outside it. Walkthrough: [C14](scenarios/c14-seal.md).

## Bits spent

The codelength of a holdout's own log under the code fixed at sealing. Under the default Krichevsky-Trofimov code, k answers with r accepts cost about log2(ek/r) bits per accept, and all the rejects together about (1/2) log2(pi k). Bits are log2 of an inflation: a test on rows seen through L bits runs at up to 2^L times its nominal level, and the transcript certificate is taken at level delta 2^-L.

## Transcript certificate

The certificate a holdout without an exam slice issues on its tuning slice, at level delta 2^-L (under the Gaussian tail, the bar z_L = Phibar^-1(delta 2^-L)). Valid for every analyst strategy and every stopping rule by the Kraft union bound over transcripts; the slice closes when it is issued. [C15](scenarios/c15-transcript.md).

## Clean share

s(L) = (z_0 / z_L)^2: under the Gaussian bar, a holdout that has spent L bits certifies with the width of a clean holdout holding that fraction of its rows, and an exam slice of share s beats the transcript certificate exactly when s > s(L). The tuning slice retires itself when s(L) falls to the sealed threshold (0.05 by default, 38.94 bits at level 0.05).

## Exam slice

A share of the holdout sealed apart from the tuning slice and opened once by `open`, at the clean level: whatever the tuning answers were, the exam rows answered none of them. A second open refuses.

## Tail

How a performance certificate turns its rows into a bound, fixed at the seal. With a loss range, declared or natural, the default is the betting bound: exact at every level for independent losses in the range, wider than the Gaussian bar. Without a range it is the Gaussian bar, a CLT approximation, and the claim says so. `tail="gaussian"`, `"hoeffding"` and `"betting"` may be named.

## Guess and Check

The numeric answer mode, `check(model, guess, width)` on a holdout sealed with `mode="numeric"`: a guess within width - tol of the holdout's value is confirmed and released; otherwise the value is released on a grid, or the mechanism halts. Every released interval covers its population value simultaneously at 1 - delta (Rogers, Roth, Smith, Srebro, Thakkar, Woodworth, AISTATS 2020). [C16](scenarios/c16-check.md).

## PEEKED

The admission verdict for rows a certificate's proposal saw only through a counted transcript: admitted at inflation 2^L with pass level level 2^-L. It sits between FRESH WITH DEPENDENCE FLAG and UNKNOWN LINEAGE; direct contact is still NOT FRESH. Lint's R7 prices the same pattern in a run record. [C17](scenarios/c17-peeked.md).

## Performance certificate

The certificate kind a holdout issues: a bound on one model's mean loss (risk) or on its mean paired gain over the sealed baseline (gain), FWER at the holdout's level, with the transcript accounting in its family and the tuning transcript recorded as a peek.

