# A→B · Found in the model. Is it in the world?

*Continues the running example from [A1](a1-read.md). This page is where the two workflows meet.*

[A2](a2-audit.md) found two rule families in the artifact: `age × group_A` (mass 0.1755) and `util × tenure` (mass 0.0077). An audit stops at the file: presence in the model. The sieve asks the data directly: which patterns does the *world* certify? Because both workflows speak the same [atom](../glossary.md#atoms) vocabulary, the audit's finding and the sieve's claim are comparable objects.

```python
from provable import CertifiedDiscovery

cd = CertifiedDiscovery(stability_reps=8, seed=0, verbose=False)
report = cd.run(X, y)
print(f"k certified = {report['certification']['k']} of P = {report['certification']['pool']} atoms  (FDR q = 0.10, B = 8 stability replicates)\n")
cols = ['atom', 'src', 'effect', 'ci_lo', 'ci_hi', 't', 'p', 'tier', 'f_A']
print(cd.claims_[cols].to_string(index=False))
```

Output:

```text
k certified = 14 of P = 281 atoms  (FDR q = 0.10, B = 8 stability replicates)

                        atom     src    effect     ci_lo     ci_hi          t             p        tier   f_A
                  group_A>=1     cut -0.795351 -0.855906 -0.734795 -25.752140 2.214148e-133   invariant 1.000
   (group_A>=1)&(age>=34.75)    conj  0.818459  0.717240  0.919678  15.854118  1.319760e-54   invariant 1.000
             income>=22.6678     cut  0.070803  0.039262  0.102344   4.401382  1.109830e-05     fragile 0.750
                 util>=0.672     cut -0.087592 -0.126831 -0.048353  -4.376802  1.241800e-05 span-stable 0.250
              income>=45.495    cart  0.084775  0.042234  0.127316   3.907260  9.522767e-05 span-stable 0.500
                tenure>=4.85    cart  0.085338  0.042410  0.128265   3.897744  9.902917e-05 span-stable 0.375
             income>=60.5333     cut  0.093686  0.039469  0.147902   3.388056  7.121775e-04 span-stable 0.125
               income>=11.99 harvest  0.134308  0.052003  0.216614   3.199492  1.389787e-03     fragile 0.375
               tenure>=25.35    cart  0.096094  0.032078  0.160110   2.943156  3.271575e-03 span-stable 0.875
(util>=0.338)&(tenure>=15.1)    conj -0.139091 -0.239819 -0.038362  -2.707395  6.816423e-03     fragile 0.125
  (tenure>=13.5)&(age>=28.5)    conj -0.171180 -0.298288 -0.044071  -2.640499  8.317527e-03 span-stable 0.000
               income>=30.91     cut  0.038192  0.008993  0.067390   2.564600  1.037349e-02 span-stable 0.750
              income>=111.56 harvest  0.181017  0.036874  0.325160   2.462264  1.385721e-02 span-stable 0.500
               income>=88.68    cart  0.147716  0.029558  0.265873   2.451174  1.429080e-02 span-stable 0.375
```

## Reading it

**The audited pattern is in the world.** The top two claims are `group_A>=1` at **−0.795** and `(group_A>=1)&(age>=34.75)` at **+0.818**, both [invariant](../glossary.md#stability-tiers), certified in the primary split and in all 8 replicates (`f_A = 1.000`). That is the `>=`-vocabulary spelling of the planted `−0.8 · (age<35 & group_A)`: penalize the group, refund the over-35 members. The sieve recovered the world's own coefficients, at the world's own threshold, from raw data: the same cell boundary the audit's profile located in the file (`age>=34.8`). Model and world agree, and now you hold both halves: certified presence in the artifact, certified presence in the data.

**The incidental pattern is not, certifiably so at this budget.** The audit gave `util × tenure` 7.2% of model variance; here, no invariant util–tenure structure appears. One util&tenure conjunction did certify, at t = −2.7, `fragile`, appearing in **1 of 8** replicates. Which brings us to the honest part:

**The bottom of a q = 0.10 ledger is supposed to look like this.** [BH(q = 0.10)](../glossary.md#fdr-and-bhq) stamps a list on which at most ~10% of claims are expected to be flukes; with k = 14, that budget is ~1.4 claims, and they will live at the low-t end. `(tenure>=13.5)&(age>=28.5)` is the sharpest lesson on the page: certified in the primary split, **replicated in zero of eight** (`f_A = 0.000`). The FDR guarantee held (it prices a fluke rate, it does not forbid flukes), and the stability tiers are the instrument telling you *which rows to lean on*. Invariant rows are load-bearing; fragile rows are the risk you paid for. (The income staircase, five cuts, all positive, is the sieve spelling a small linear effect in steps: span-stable, identity fragile, family real.)

## What's guaranteed

The FDR of the ledger, at q = 0.10, from one [selection-conditional](../glossary.md#selection-conditional) OLS pass on the reserved half. Tiers are descriptive replication frequencies (`f_A` identity, `f_F` family), not additional probability claims. And the standing limit from A2 applies with full force in reverse: certified association in the data is still not causation, and absence from this ledger is absence *at this budget and vocabulary*, not proof of absence; [B1](b1-claims.md) meets that head-on.

---

*What the papers add*: tier→truth calibration as a cross-instrument pedigree (graded on planted ground truth), and near-duplicate families as the geometry behind fragile identities: Papers 2–3, [SSRN 7247302](https://doi.org/10.2139/ssrn.7247302) (archived: [Zenodo 21819895](https://doi.org/10.5281/zenodo.21819895)), [10.5281/zenodo.21819925](https://doi.org/10.5281/zenodo.21819925).
