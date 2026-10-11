# B2 · The truth your vocabulary cannot say

A claims ledger can only certify what its vocabulary can *name*. The default dictionary speaks in `>=` cuts, and plenty of real effects live elsewhere: in a **low tail**, or in an **interior band** ("moderate values of b are penalized"). The world below has exactly one fact: a −0.8 effect on the middle quintile of `b`. Watch the same data through two vocabularies.

```python
"""B2 -- the truth is an INTERIOR band. Same data, two vocabularies:
default (>= cuts only) vs v0.2 (band_conj + le_cuts, with core dedup)."""
import numpy as np
import pandas as pd
from provable import CertifiedDiscovery

rng = np.random.default_rng(9)
n = 4000
X = pd.DataFrame(rng.normal(0, 1, (n, 15)), columns=[f'x{j}' for j in range(15)])
X['a'], X['b'] = rng.normal(0, 1, n), rng.normal(0, 1, n)
lo, hi = X['b'].quantile(0.4), X['b'].quantile(0.6)
y = (-0.8 * ((X['b'] > lo) & (X['b'] <= hi)) + 0.3 * rng.normal(size=n)).values   # one fact

off = CertifiedDiscovery(seed=3, verbose=False)
r_off = off.run(X, y)
print(f"-- default vocabulary: k = {r_off['certification']['k']} --")
print(off.claims_[['atom', 'src', 'effect', 'ci_lo', 'ci_hi', 't', 'p']].to_string(index=False))

on = CertifiedDiscovery(seed=3, verbose=False, le_cuts=True, band_conj=True, core_dedup_r=0.95)
r_on = on.run(X, y)
print(f"\n-- extended vocabulary (le_cuts, band_conj, core_dedup_r=0.95): k = {r_on['certification']['k']} --")
print(on.claims_[['atom', 'src', 'effect', 'ci_lo', 'ci_hi', 't', 'p']].to_string(index=False))
```

Output:

```text
-- default vocabulary: k = 4 --
        atom src    effect     ci_lo     ci_hi          t            p
b>=-0.128257 cut -0.449145 -0.533550 -0.364739 -10.438178 1.185778e-24
  b>=0.16873 cut  0.407640  0.310175  0.505104   8.204205 5.026004e-16
 b>=0.437474 cut  0.329036  0.211472  0.446599   5.490084 4.728597e-08
b>=-0.429925 cut -0.301789 -0.421740 -0.181838  -4.935223 8.921167e-07

-- extended vocabulary (le_cuts, band_conj, core_dedup_r=0.95): k = 2 --
                        atom  src    effect     ci_lo     ci_hi          t            p
(b>=-0.429925)&(b<=0.437474) conj -0.345349 -0.395109 -0.295590 -13.613994 7.610252e-40
 (b>=-0.128257)&(b<=0.16873) conj -0.437050 -0.504371 -0.369729 -12.734558 2.559831e-35
```

## Reading it

**Default:** four true claims that spell one fact as a staircase: step down at −0.128, back up at +0.169, plus the shoulders at −0.43 and +0.44. Every claim is correct; each is a *difference boundary* of the band, and reconstructing "the middle quintile is penalized" from them is left to the reader. The claims are certified; the *statement you cared about* isn't in the ledger's language.

**Extended:** `band_conj=True` makes same-column `(b>=a)&(b<=b)` cells first-class atoms, and the ledger now says it directly: the inner band `(b>=-0.128)&(b<=0.169)`, the planted quintile at the planted boundaries, at **−0.437**, with a wider concentric band at −0.345. Two claims, readable as the world's own description. (The two bands are nested, so the S2 pass splits the −0.8 between them: −0.437 + (−0.345) ≈ −0.78 on the inner cell. Partial effects of overlapping cells add; the decomposition is honest, not a discrepancy.)

Two mechanics worth knowing before you flip flags on your own data:

- **Flags off is byte-identical to v0.1.** All v0.2 vocabulary extensions (`le_cuts`, `band_conj`, engine path harvest) default off; when off, claims, report dict, and printed report are unchanged. Certified history doesn't move under your feet.
- **A denser vocabulary needs the dedup.** `le_cuts` mirrors every `>=` cut as a `<=` twin (exact complements), and dense near-duplicates are how a family's S2 evidence gets split and killed ([B1's missing x23](b1-claims.md#where-is-x23) is that failure mode in the wild). `core_dedup_r` admits one representative per near-duplicate family at the core; the flags are designed to be used together.

**The diagnosis this page trains:** when the honest model underperforms, there are two different diseases: the lawful [toll](../glossary.md#toll) of certification, and a vocabulary that cannot span the signal. They have different fixes (more data vs. a richer dictionary), and the report's theory panel and [dark-mass ledger](b4-darkmass.md) tell you which one you have. Here the disease was vocabulary, and two flags cured it.

## What's guaranteed

Same guarantee, either vocabulary: FDR at q = 0.10 over one selection-conditional S2 pass. Extending the dictionary changes what is *nameable*, never what "certified" means.

---

*What the papers add*: the toll theorems (honesty's cost bounded above and below), and the two-diagnoses result: lawful price vs. vocabulary failure, and how the instrument distinguishes them: Paper 2, [SSRN 7247302](https://doi.org/10.2139/ssrn.7247302) (archived: [Zenodo 21819895](https://doi.org/10.5281/zenodo.21819895)).
