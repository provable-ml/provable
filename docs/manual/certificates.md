# Certificates & the book

The reference for the certificate layer: the objects [C1](../scenarios/c1-certify.md) through C4 use, and the mediated holdout of [C14](../scenarios/c14-seal.md) through [C17](../scenarios/c17-peeked.md), in tables. The scenario pages show them working; this page states what each field asserts.

## The certificate tuple

A certificate is one record: *(claim, guarantee, test lineage, proposal lineage, artifact, support, Gram context, timestamp, kind, consumed)*.

| field | what it asserts |
|---|---|
| `claim` | The sentence being certified, human-readable. |
| `guarantee` | The error-rate contract: a kind, a level, and the family it was earned over (see below). |
| `ell_test` | The [lineage](#lineage-and-freshness) of the rows the claim was *graded* on. |
| `ell_prop` | The lineage of the rows that *proposed* it, hereditarily including everything each consumed certificate touched. |
| `artifact` | The full machine record of the run (for an instrumented run, the engine report verbatim). |
| `support` | The certified atom set V, by name. |
| `gram_context` | A descriptor of the certified-support Gram on the certifying rows: the correlation block, rounded, hashed. Two certificates over the same names but different geometry differ here. |
| `timestamp` | Issue time, UTC. |
| `kind` | carrier, existence, artifact, or performance (below). A conditional kind is reserved; loaders preserve unknown kind strings. |
| `consumed` | The certificates this one used, by content id, inside the hashed content. |
| `ell_peek` | Present only when the proposal saw rows through a counted transcript: per holdout, the rows and the bits. Inherited through consumption at the largest bits per holdout; inside the hashed content; absent otherwise, so a certificate without peeks serializes exactly as before. |

## Kinds

| kind | what it is |
|---|---|
| `carrier` | Names a coordinate set carrying mass: the sieve's certified family. One carrier per certified run; the guarantee is family-level. |
| `existence` | Detection without a carrier: something is there, at a level, without naming which members. Refinement (the R6 check) licenses later carriers against it. |
| `artifact` | An exact evidence bundle of the engine's `audit` and `enforce` verbs, unified into the certificate type. Guarantee kind EXACT, no lineage, and the bundle's `PV-` id retained as its identity; `verify_id` continues to hold on the embedded bundle unchanged. |
| `performance` | Issued by a [holdout](#the-holdout): a bound on one model's mean loss, or on its mean paired gain over the sealed baseline, at FWER over the transcript family (route, code, bits, bar, the log head). One per opened slice; the Guess and Check family is one more. |

## Guarantees

| guarantee kind | level | meaning |
|---|---|---|
| `FDR` | q in (0, 1] | Expected false-discovery proportion at most q over the declared family. |
| `FWER` | alpha in (0, 1] | Probability of any false claim at most alpha (the [registered charge](#the-account) issues these). |
| `FCR` | alpha in (0, 1] | False coverage rate of the quoted intervals on the selected set. |
| `EXACT` | 0, always | An arithmetic fact of an artifact. No error budget exists, so EXACT certificates never charge the account. |

The `family` inside a guarantee must let a reader reconstruct the multiplicity the level was earned over: for a carrier, the core member ids, the full p-vector, m, khat, and q travel in it.

## Identity and the chain

A certificate's id is `CD-` plus the sha256 of its canonical bytes (sorted-key JSON, fixed number forms). Artifact-kind certificates keep the engine bundle's `PV-` id. The ids of consumed certificates sit *inside* the hashed content, so lineage is a tamper-evident chain: edit any ancestor and every descendant's id stops matching, which `verify_chain` checks recursively. Row sets are stored in-record, compressed, so a certificate is self-contained.

The schema is additive-only: a loader preserves fields it does not recognize and re-emits them on save. A record written by a later minor version round-trips through an earlier one.

## Lineage and freshness

A lineage is a set of slices; a slice is *(source fingerprint, dataset tag, row ids, clock window)*. The source fingerprint is a content hash of the source table, so identity is by data, not by file name. Date and time-span columns (and a datetime index) are hashed as int64 microseconds, or nanoseconds when they carry sub-microsecond parts, so a table has one fingerprint whatever pandas version or storage unit holds it; a slice built now also matches the ids earlier releases recorded for the same table under the units s, ms, us and ns, so an older book still recognises its rows. Row sets make identity-freshness exact: two lineages are fresh for each other when no slice pair on the same source shares a row. Clock windows carry real time when the panel has it; identity-disjoint slices on *different* tags with overlapping windows raise the dependence flag (below), which reports and does not block at lag zero.

Un-instrumented history enters through a sentinel: a lineage whose source is unknown. Nothing built on it is ever silently fresh.

## Admission verdicts

Reuse of certificates against new rows returns an Admission: a verdict, the inflation the pattern incurs, the pass level for the next test, the named overlap, and any demotion.

| verdict (enum) | printed label | meaning | budget-bearing |
|---|---|---|---|
| `FRESH` | FRESH | Every consumed certificate is fresh for the consuming rows; guarantees transport unchanged. | yes |
| `FRESH_DEP_FLAG` | FRESH WITH DEPENDENCE FLAG | Identity-disjoint, but clock windows overlap across dataset tags. Admitted; the overlap is named. Flag-only at lag zero. | yes |
| `PEEKED` | PEEKED | Identity-disjoint, but the consuming rows were seen by the certificate's proposal through counted transcripts of L bits in all (one transcript per holdout at its largest bits, distinct holdouts adding). Admitted at inflation 2^L with pass level level 2^-L. | yes |
| `NOT_FRESH` | NOT FRESH | Lineage meets the consuming rows. Refused by default; under explicit demotion, admitted with the guarantee marked MISCALIBRATED and the overlap recorded. | demoted |
| `UNKNOWN_LINEAGE` | UNKNOWN LINEAGE | The item predates instrumentation. Returned, listed in the statement, never a silent pass, never budget-bearing. | no |

## The account

The book (a `Ledger`) is a directory: an append-only `ledger.jsonl` of operations (register, void, reserve, note, grade, refine, route, and the holdout and watch operations below) and a `budget_state.json` holding the target. Its statement quotes one number for the whole book under the rule that always holds: the realized level of the active book is at most the *sum* of active certificate levels. EXACT certificates and UNKNOWN LINEAGE items are excluded from the sum.

Two further disciplines live on the account, both shown in [C3](../scenarios/c3-book.md):

| discipline | call | contract |
|---|---|---|
| Online budget | `next_level(n)`, `combine_batches()` | Batch levels quoted by the batch step-up rule (Zrnic, Jiang, Ramdas, Jordan, AISTATS 2020: Definition 1, Algorithm 1) keep the *union* FDP at the account target at every step, given per-batch fresh lineage and a monotone schedule. A book whose recorded levels deviate from the schedule is detected and falls back to the additive sum, which needs no conditions. |
| Registered charge | `reserve(alpha, vocabulary=V)`, `claim_registered(...)` | Fix a finite vocabulary of member/row-set pairs in advance; claim only at p at most alpha/\|V\|. The familywise level alpha then holds under arbitrary reuse, order, and lineage: the one guarantee that survives re-testing on spent rows. A `reserve` without a vocabulary is a plain budget hold, not this object. |

Notes (`note(...)`) are append-only provenance lines rendered in the statement; they never touch the account. Grade entries (the scorecard verb, [C5](../scenarios/c5-grade.md)) are budget-bearing: each graded level charges the account, or consumes the named reservation, and renders in the statement's grades section. A watch (below) is budget-bearing too: it charges its level when it opens, and renders in the statement's watches section. A route (below) reads training rows only and charges nothing; recorded with `ledger=`, it renders in the statement's routes section as the choice's transport status.

## The rates register

Quantities in this layer never cross word families; rates are the only cross-currency objects, and each quotes its exactness grade wherever it is used.

| rate | converts | grade |
|---|---|---|
| `row_equivalent(n, lam, tau_v)` | columns to rows | exact in the acquisition model |
| `rows_for_level(n, q_from, q_to, k, m)` | validity to rows | identity at fixed effect size and fixed rejection count |
| `rows_for_bits(n, bits, delta)` | validity to rows | identity at fixed effect size, one-sided Gaussian bar: n (z_L / z_0)^2 = n / s(L) |
| `anchor_to_promise(anchor, bridge)` | declared money to required mass | DECLARED: refuses without a user-supplied bridge; the register carries no theory for the conversion |

## The holdout

A holdout (`pv.seal`) is a set of rows put in custody in the book, reached afterwards only through its methods. Its operations are holdout ops in the book's log, hash-chained per holdout: each carries the hash of the one before it, the first is the seal, the holdout's id is the hash of the seal record, and a custody head records the last operation so that an edit or a truncation of the public log is caught.

| step | call | what it does |
|---|---|---|
| seal | `pv.seal(X, y, ledger, lineage=..., baseline=..., baseline_lineage=...)` | Commits, before any answer: salted hashes of X, y, the clock and the baseline's predictions, the row identities, the split seed and the exam slice, X's declared outcome columns (`columns=`), and the protocol (mode, route, code, eta, delta, exam share, retirement threshold, tail, loss and range, claim form, sealing type). Rows an active certificate of the book names (test or proposal rows, as far as its lineage is known), or rows already under seal in it, refuse, checked before the baseline runs and again under the book lock. |
| ask | `ho.ask(model, lineage=...)` | Comparative mode. ACCEPT iff the paired gain over the incumbent on the tuning slice exceeds eta paired standard errors (the Ladder's rule); an accepted model becomes the incumbent. The answer is priced under the sealed code and nothing else is returned. The answer that takes s(L) to the threshold retires the tuning slice in the same operation. |
| check | `ho.check(model, guess, width, lineage=...)` | Numeric mode, Guess and Check. CONFIRMED (the guess, with the width), FAILED (the value on a grid of step width - tol, with the width), or HALTED. The family certificate is registered before the first answer; a check whose model was fit on rows the family does not name registers a successor and voids the predecessor. `ho.min_width()` gives the next tolerance from the public transcript. |
| open | `ho.open(model, lineage=...)` | The performance certificate: on the exam slice once at the clean level delta, or, with no exam slice sealed, on the tuning slice at level delta 2^-L (the bar z_L under the Gaussian tail), after which the slice closes. One charge at delta. The operation is logged before the certificate is registered. |
| retire | `ho.retire()` | Ends the tuning slice and releases its rows (and the exam's, once opened) from custody. Released rows are free: a certificate on them registers like any other, and admission prices what the transcript saw (PEEKED, below). |
| transcript | `ho.transcript()` | The public log, one row per operation: the log number (the key `audit(models=...)` takes), the action, the label given with the call, the answer returned, the bits that answer was priced at, and the transcript's codelength after it. |
| audit | `ho.audit(models=...)` | Runs under the book lock. Checks the chain, the custody head, the dates and the id, and does not replay a log that fails them; then checks every commitment against the custody files, replays every operation through the state machine (answers, codelengths, certificates rebuilt and matched to the book, every logged state compared in full) and checks the book around the holdout: each of the log's certificates registered once, as a performance certificate, and active; no other active record naming the holdout (in any field but its peeks); none registered on rows while they were in its custody; none before the seal naming it or its rows. Models are re-run after the lock, only when the replay passed and every row has left custody. |

[C18](../scenarios/c18-end-to-end.md) runs these steps once, in order, with the bits at every step. [C19](../scenarios/c19-changed-world.md) and [C20](../scenarios/c20-few-rows.md) set looking against not looking: on a holdout drawn after the law changed, and on training rows too few to rank the candidates. [C21](../scenarios/c21-looking-hurts.md) does the same where looking hurts: a loop that keeps every feature the holdout score favours, with and without a price.

Every call that reads a model passes the lineage gate first: the model's lineage must be declared (`Lineage()` declares no rows), instrumented, and identity-disjoint from every row still in custody. Outcome-derived columns are declared with `ColumnDeclaration`, in two forms. A lag declaration is checked at the seal, row by row against the holdout's clock: it may not reach a sealed row's own outcome or an outcome delivered after that row's decision time, and one whose window reaches other sealed rows switches predictions to one row per call, the baseline's included, with the seal refusing a random exam split that would carry exam outcomes into tuning rows (`exam="last"` puts the exam slice after the tuning slice in time). After the seal the gate accepts exactly the lag declarations the seal checked and refuses any other by name, without reading the clock. A rows declaration, a column computed from named rows' outcomes, is checked at every call against the rows still in custody: UNKNOWN lineage, a row in custody, or an outcome delivered after the earliest decision time in custody refuses. Gapped columns and targets refuse at the seal, each by name with its count.

Until `retire()` hands released rows back, nothing the holdout returns depends on the sealed values beyond the priced answers and the issued certificates, and its refusals depend only on declarations, the public lineage (the seal's, and the rows and envelopes of the tuning and exam slices; under `exam="last"` the split follows the order of the sealed end times) and the public log; a column refusal names the column and the rule, never a row's window. A declared range clips the loss, brier clips probabilities to [0, 1], a non-finite prediction falls back to the sealed baseline's for that row (or, with no baseline, scores the top of the range), and a certificate's claim names each of these when it applies. Arithmetic on sealed values ignores numpy's floating-point error settings, so a model built to make it raise is answered and priced like any other.

What is theorem and what is design:

| | statement |
|---|---|
| theorem | The Kraft union bound over transcripts: a code fixed at sealing, a closed channel (every choice of the analyst a function of their own rows, independent randomness and past answers), and a single-model tail at level gamma give a certificate at level delta 2^-L that fails with probability at most delta for every strategy and stopping rule. Cited in substance: Guess and Check (Rogers, Roth, Smith, Srebro, Thakkar, Woodworth, AISTATS 2020, Theorem 3.2) and the Ladder (Blum and Hardt, ICML 2015). |
| theorem | An exam slice never queried certifies at the clean level; under the Gaussian bar, with a share s of the rows it is the narrower certificate exactly when s > s(L). |
| theorem | With a loss range the single-model tail is exact. The betting bound on the mean of n independent values in [lo, hi] at level gamma is the largest m whose capital (1/J) sum_j prod_i (1 + lambda_j (m - u_i)) stays below 1/gamma, with u the values scaled to [0, 1] and J bets lambda_j fixed by n. At the true mean the capital has expectation one, so by Markov's inequality the bound falls below the mean with probability at most gamma (Waudby-Smith and Ramdas, JRSS-B 2024). The Hoeffding bound is exact too, and wider. |
| theorem | Guess and Check, Algorithm 1 with the Hoeffding tolerance: every released interval covers its population value simultaneously at 1 - delta, for a statistic with a declared range. The transcript count uses the exact grid size, ceil(1/gamma) + 1. |
| design | The comparative rule and eta (default 1), the code (Krichevsky-Trofimov by default, or fixed-rate), the exam share (0.2), the retirement threshold on s(L) (0.05), the route fixed at sealing, and the tail. `tail="auto"` (the default) resolves at the seal to `betting` when the loss has a range (declared, or natural for a bounded loss such as Brier) and to `gaussian` when it has none, and the certificate's claim names it: "exact, betting bound" or "CLT approximation". The Gaussian bar is an estimate: in lock TW7, 100 exam rows of a heavy-tailed paired gain, it fails in 17 of 200 simulated holdouts at nominal 0.05, inside the lock's three-standard-error slack; on clipped squared errors it misses beyond its level, more so once bits are spent (lock TW21). `gaussian`, `hoeffding` and `betting` may be named; the last two need a range. |
| custody | API-level mediation plus salted hash commitments. The rows sit under `holdouts/<id>/` in the book directory as numpy and JSON files (no pickle). Every row and prediction file is checked against its salted hash before it is parsed; the custody metadata (salt and schema) and the custody head are plain JSON, and a changed salt or schema fails the same hashes. Book writes, with the checks that decide them, take an exclusive lock (a thread lock per handle, then an advisory file lock where the platform has one); models run outside it, so a model may write to the book. The seal refuses rows that an active certificate of the book names. The book refuses a certificate or a refinement on rows in custody other than the holdout's own (a performance certificate whose content hashes to an id its log names, registered once), and the audit finds one written around the API. A process that reads or writes the custody files directly is not stopped (the custody head carries no secret), and physical custody is outside the library. Models run on the sealed rows inside the custodian; what a model does with what it is given (its errors and side effects included), like the closed channel and the truth of column declarations, is the analyst's undertaking. |

PEEKED reaches the other verbs as well: `grade` on transcript-seen rows tests its cells at the repaired level and charges the book the nominal one, and `certify` refuses certifying rows that a book certificate, or a consumed one, saw through a transcript.

## The transport screen

A choice between two alternatives (a feature in or out, a hyperparameter, a regime filter) can be tuned through a holdout when its effect is the same in every time block. When its effect moves from block to block, an analyst who remembers what the holdout period was like can steer the choice with that memory, outside the holdout's log, and the certificate stops describing the blocks to come. `pv.route` screens the choice on the training rows only and routes it.

| step | call | what it does |
|---|---|---|
| screen | `pv.route(d, blocks, h=..., r=2, delta=0.05, window=0.3)` | d holds the choice's per-row paired loss differences on the training rows (the default's loss minus the alternative's), cross-fitted so that no row is scored by a model fitted on it, in time order; blocks gives each row's time block at the resolution at which history is remembered, each block one contiguous run of at least two rows, at least three blocks. h is the number of holdout rows that would answer the choice; `holdout=` reads it from a sealed holdout's tuning slice and, with `lineage=` the rows d came from, refuses rows still in its custody. r is the number of states per block that memory tells apart. Returns a `Routing`: the verdict with its reason, the effect per block, Cochran's Q, the spread of the effect (DerSimonian-Laird and its Q-profile upper bound), the memory bound, the clean bar of holdout noise and the fluctuation statistic. |
| record | `pv.route(..., ledger=book, label=...)` | Appends the verdict and its numbers to the book as a route op carrying hashes of d and the blocks. Charges nothing. |

What is theorem and what is design:

| | statement |
|---|---|
| theorem | Memory acts only through the spread of the effect. If the holdout's estimate of the effect is a weighted sum of the block effects plus noise independent of the states the blocks were in, and the block effects are independent across blocks and subgaussian about their mean with proxy tau_c, a choice steered by any information M about those states biases the certificate for the mean effect by at most tau_c \|w\|_2 sqrt(2 I(choice; M)) (the Donsker-Varadhan inequality); with equal weights over the holdout blocks and memory that tells r states apart in each, by at most tau_c sqrt(2 ln r), however long the holdout. With tau_c = 0 the estimate does not depend on memory. |
| cited | The Q-profile upper bound on tau_c^2 (Viechtbauer, Statistics in Medicine 2007) in the random-effects model of the block means with their standard errors, beside Cochran's Q and the DerSimonian-Laird estimate. The fluctuation statistic of Giacomini and Rossi (Journal of Applied Econometrics 2010), the largest moving-window sum of the differences over a long-run standard deviation (Newey-West), taken on the demeaned differences so that its null is a constant effect rather than equal performance at every point; its critical value is the 1 - delta quantile of the limiting Brownian-bridge statistic, by a seeded simulation (`transport.fluctuation_critical`). Lock TI2 checks the bound's coverage and TI3 the statistic's level. |
| design | VAULT when the fluctuation statistic stays below its critical value and tau_c^UCB sqrt(2 ln r) is at most z_0 se_H, se_H the standard deviation of d over sqrt(h): the worst memory bias stays below one clean bar of holdout noise. FORWARD otherwise: the choice is registered forward, its predictions issued before their outcomes, and followed by a watch. The rule uses the upper bound rather than a failure to reject, so a short history, whose bound is wide, routes forward, which is the conservative failure. The window (a share of the rows, 0.3 by default) and r (2 by default) are declarations. |

[C23](../scenarios/c23-transport-screen.md) screens a fine-grained choice and a regime-level one on 48 training weeks and runs both through the holdout anyway: the first one's certificates carry over to the long run, the second one's describe the holdout weeks.

## The watch

A performance certificate bounds a model's mean loss, or its mean paired gain over the sealed baseline, on the law the holdout sampled. After deployment the law can move. A watch (`pv.watch`) follows the deployed rows against the certified level. Its operations are watch ops in the book's log, hash-chained per watch like a holdout's, with a head file that anchors the last one; the watch's id is the hash of its open record, and the rows of every update are stored under `watches/<id>/` as numpy files whose hashes the log carries.

| step | call | what it does |
|---|---|---|
| open | `pv.watch(cert, ledger, delta=0.05, horizon=None, loss_range=None)` | Binds to a performance certificate registered and active in this book and charges delta to the account before any row is seen; refuses when the book's remaining budget is below delta. The monitored statistic is the certificate's loss (risk claim) or the model's loss minus the baseline's (gain claim), on the certificate's sealed loss range; an unbounded loss sealed without one needs `loss_range=(lo, hi)` here, with lo at most the loss's natural minimum, so that clipping only lowers losses. |
| update | `w.update(y, p, times=..., baseline=...)` | Appends forward rows, the outcomes and the predictions issued before them (with the baseline's predictions for a gain claim), and runs the detector over them in order. Times are given with every update or with none; with them, the first row may not start before the certified rows end, and rows may not go back in time. A non-finite prediction scores the top of the range. At the first row whose evidence reaches 1/delta the watch raises its alarm, with the estimated row at which the breach started, and stops; the rest of that update is stored but not monitored. |
| close | `w.close(cause)` | Ends a watch that is still watching. The charge stays: it was taken before any row was seen. |
| audit | `w.audit()` | Under the book lock: the hash chain, the id and the head; every update's row files against the hashes logged with them; the detector re-run over the stored rows, update by update, and every logged state compared in full. |

What is theorem and what is design:

| | statement |
|---|---|
| theorem | If, given the rows before it, every monitored row has a mean loss at most the certified level (a mean paired gain at least the certified bound), the watch raises an alarm with probability at most delta, at any row of an unbounded horizon, or of the declared horizon of T rows. The evidence is a mixture, over the row s at which a breach may start (prior weight 1/(s(s+1)), or 1/T under a horizon) and over a grid of bets lambda, of the betting processes prod (1 + lambda (u_r - m)) on the loss scaled to [0, 1], m the certified level on that scale. Under the hypothesis the mixture is a nonnegative supermartingale that starts at 1, and Ville's inequality bounds the chance that it ever reaches 1/delta. Cited: e-detectors (Shin, Ramdas and Rinaldo, arXiv:2203.03532), betting processes for bounded variables (Waudby-Smith and Ramdas, JRSS B 2024), Ville's inequality. Lock TZ2 recomputes the evidence by brute force, and TZ3 counts false alarms under the worst null, losses with mean exactly at the level. |
| design | The bet grid lambda = 2^-k / m for k = 1 to 8 (every factor at least 1/2) with uniform weight, the prior over start rows, the declared range, and the start estimate, the start row of the largest single-start process (the CUSUM argmax over the grid). A horizon prices a late breach at log T rather than 2 log s and ends the guarantee at T rows. How soon an alarm follows a breach is not a theorem here; [C22](../scenarios/c22-drift-watch.md) measures it. |
| meaning | An alarm says that, from some row on, the live mean loss of the monitored predictions exceeded the certified level. It does not void the certificate, which remains a statement about the law the holdout sampled. That the predictions are the certified model's (or the model the report is carried to, as after a refit) and were issued before their outcomes is the analyst's undertaking: the watch checks the clock and the loss, not the model. |

[C22](../scenarios/c22-drift-watch.md) certifies a model on the latest rows of a drifting world, refits it through the holdout, and watches the refit model's deployed rows: the report carried to the refit model holds on average up to lead (h + 1)/2, and over 60 worlds no watch alarmed before the live risk passed the certified level.

## Sources

Every check the linter emits (C4) carries a per-row tag naming the exact statement of record it enforces, from the certificate-algebra and floors papers on the [citation page](../cite.md); the imported batch rule additionally cites its source paper in full. R7, the priced peek, is tagged `RRSSTW20 Thm 3.2`: the transcript union bound of Guess and Check (Rogers, Roth, Smith, Srebro, Thakkar, Woodworth, Guaranteed Validity for Empirical Approaches to Adaptive Data Analysis, AISTATS 2020, PMLR 108), of which the Kraft form for any code fixed at sealing is a mild generalization. The tags print in the output so a reader can audit the rule-to-source map line by line.
