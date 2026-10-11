# Workflow map

Where every verb lives and what 0.12 ships. `[S]` is the shipped engine surface, frozen and additively extended; `[T]` is the certificate layer, built over 0.10 to 0.12 and first published in 0.12; `[P]` is partial. The version map says which milestone added what.

```text
==============================================================================
 PROVABLE 0.12 WORKFLOW MAP
 Status: [S] shipped engine  [T] certificate layer (0.10 to 0.12)  [P] partial
==============================================================================

CERTIFICATION (accuracy + validity)
  [T] certify(panel, y, q)            run instrument, issue certificate
      |    expect_manifest=... refuses on mismatch before any ledger write
      +--> [P] rep.refusal            empty-run diagnosis, coarse phase
      |                               string; buy-spec upgrade scheduled
      +--> [S] rep.engine             dark-mass accounting and the band
      |                               quantities print engine-side today
      +--> [T] rep.atom_card        est marginals per certified atom;
      |                               net across cells via rep.core_fit()
      +--> [T] rep.core_fit()         certified tier fitted value: the
      |                               netting object, own or fresh rows
      +--> [T] rep.claims             selective intervals, two-sided,
      |                               alpha_sel = khat*q/m (FCR)
      +--> [T] rep.engine_vs_core()   one-line finding row in the report
      +--> [T] rep.penumbra           corrected penumbra (Storey floor;
      |                               adaptive-dictionary note field)
      +--> [T] pv.reuse(...)          admission + pricing of reuse
      +--> [T] spec.as_column(fam)    bridge: model finding -> world
                                      feature, exact

ANATOMY (accuracy)
  [T] pv.examine(model, X)            one call: describe + families +
      |                               groups rollup + transport share
      +--> [T] ad.describe()          model summary, link-space handled
      +--> [T] spec.describe(X)       spectrum summary
      +--> [T] spec.families(X)       enumerate + rank by removal cost
      +--> [T] spec.describe_family(fam, X)
      |                               magnitude, removal cost, share,
      |                               joint value
      +--> [S] pv.audit(...)          compliance verdict, frozen
      +--> [S] pv.enforce(...)        surgical removal, frozen; additive
                                      probe memory budget

REUSE + BOOK (validity)
  [T] pv.reuse(certs, rows)           admission: FRESH / NOT FRESH /
                                      FRESH WITH DEPENDENCE FLAG
                                      (flag-only) / UNKNOWN LINEAGE
  [T] pv.lint(run | ledger)           R1-R6 post-hoc, source-tagged,
                                      numeric repair
  [T] led.statement()                 one error rate across the book;
                                      unknowns listed, never budget-
                                      bearing; notes rendered
  [T] led.next_level(n)               online budget quote (batch
                                      step-up, imported rule)
  [T] led.combine_batches()           schedule conformance + estimator;
                                      deviation detected, additive
                                      bound governs a deviating book
  [T] led.reserve(a, vocabulary=V)    registered charge: claims valid
      +--> led.claim_registered(...)  at a/|V| under arbitrary reuse
  [T] led.note(text)                  append-only provenance notes
  [T] pv.grade(cert, X, y, ledger)    scorecard verb: admission-first,
                                      one-sided per-cell verdicts,
                                      budget-bearing book entry

ACQUISITION (data)
  [T] rates register                  row_equivalent, rows_for_level,
                                      anchor_to_promise (DECLARED),
                                      each quoting its grade
  [T] purchase_order(rep)             rows vs columns pricing   (0.11)
  [T] licensed refinement on the book                           (0.11)

INQUIRY (the archive shadow; free of acquisition)          (0.11)
  [T] shadow(rep, X, y, clock)        owned residual on a declared
                                      out-of-fit slice; refusals by
                                      row identity; Shadow.join frames
  [T] characterize(sh, atoms)         basis interrogation, persistence
                                      ledger, the Twenty Questions
                                      charter filed (Table 4)
  [T] screen(sh, candidates, ...)     screening law: exact null,
                                      one-sided band, m_req, the
                                      build/screen/refuse trichotomy

LOOP (grow the shadow, then absorb it; free of acquisition)   (0.11)
  [T] alternate(sh, classes)          build half: two archive classes
                                      in alternation, Friedrichs angle,
                                      half-step ledger, EXIT chart
  [T] pursue(sh, candidates, req)     acquire half: forward stagewise
                                      priced pursuit, trichotomy each
                                      round, the staircase of drops
  [T] select(X, y, clock, ...)        the loop on owned columns for a
                                      logistic class: model-free value
                                      vs class gain, enrichment absorb

HOLDOUT (priced peeking through a mediated holdout)           (0.12)
  [T] seal(X, y, ledger, ...)         custody in the book; hashes,
      |                               identities, split, protocol
      |                               committed before any answer
      +--> [T] ho.ask(model)          comparative answer at eta paired
      |                               se, priced in bits (KT code)
      +--> [T] ho.check(model, g, w)  numeric answer, Guess and Check;
      |                               family certificate at 1st check
      +--> [T] ho.open(model)         exam slice once at the clean level,
      |                               or transcript certificate at
      |                               delta 2^-L; exact (betting bound)
      |                               with a loss range, CLT bar without
      +--> [T] ho.retire()            rows leave custody; also at
      |                               s(L) <= 0.05 on its own
      +--> [T] ho.transcript()        the public log: log numbers,
      |                               answers, bits, certificates
      +--> [T] ho.audit(models)       recompute chain, commitments,
                                      answers, bits and bounds
  [T] reuse verdict PEEKED            inflation 2^L, pass level q 2^-L
  [T] lint R7                         priced peek, repair q 2^-L
  [T] rows_for_bits(n, L)             validity -> rows, n / s(L)
  [T] watch(cert, ledger)             deployed rows against the certified
      |                               level; alarm at evidence 1/delta
      +--> [T] wa.update(y, p)        rows in time order, forward only
  [T] route(d, blocks, holdout=)      VAULT or FORWARD, read on the
                                      training rows before the holdout

MODEL VERIFICATION
  [S] verify CLI + evidence bundle    third-party verification, frozen;
                                      bundles are artifact-kind
                                      certificates, PV id retained

==============================================================================
 VERSION MAP (0.10 and 0.11 were development milestones; 0.12 is the first
 release of the certificate layer on PyPI)
  0.10.0  every [T] item not marked (0.11) or (0.12), on the frozen [S]
          engine
  0.11.0  purchase layer, licensed refinement, the inquiry layer
          (shadow / characterize / screen), the loop layer
          (alternate / pursue), acquisition vignettes
  0.12.0  the mediated holdout (seal / ask / check / open / retire /
          transcript / audit), the PEEKED verdict, lint R7, rows_for_bits, the
          performance certificate kind; conditional certificate kind; the
          watch and the route; tail auto (an exact betting bound when the
          loss has a range)
==============================================================================
```

The C-track scenario pages walk the certificate layer top to bottom: [C1](../scenarios/c1-certify.md) runs `certify` and reads the standard report; C2 prices reuse; C3 works the book's two budget disciplines; C4 lints a declared pipeline; [C8](../scenarios/c8-shadow.md)-[C10](../scenarios/c10-screen.md) walk the inquiry layer: the shadow, the charter, the screen; [C11](../scenarios/c11-alternate.md) and [C12](../scenarios/c12-pursue.md) walk the loop layer: the build half and the acquire half; [C13](../scenarios/c13-select.md) runs the loop over owned columns as feature selection; [C14](../scenarios/c14-seal.md)-[C17](../scenarios/c17-peeked.md) walk the mediated holdout: seal and ask with an exam slice, the transcript certificate, Guess and Check, and the price of a result that saw the holdout; [C18](../scenarios/c18-end-to-end.md) runs the whole protocol once, in order, with the bits at every step; [C19](../scenarios/c19-changed-world.md) and [C20](../scenarios/c20-few-rows.md) set looking against not looking, on a holdout drawn after the law changed and on training rows too few to rank the candidates, and [C21](../scenarios/c21-looking-hurts.md) on a loop that keeps every feature the holdout score favours, where looking hurts; [C22](../scenarios/c22-drift-watch.md) carries a certificate past a refit under drift and watches the deployed rows, and [C23](../scenarios/c23-transport-screen.md) screens a choice on the training rows before it reaches the holdout. The A track ([A1](../scenarios/a1-read.md) onward) covers anatomy on the shipped engine, the B track covers the sieve itself, and [Certificates & the book](certificates.md) is the field-by-field reference behind all of it.
