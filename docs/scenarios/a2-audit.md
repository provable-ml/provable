# A2 · Prove it never combines age with group, or find where it does

*Continues the running example from [A1](a1-read.md).*

A regulator, a model-risk officer, or your own policy says: this model may never use `age` **in combination with** `group_A`. Feature importance can't answer that: importance is per-feature, and the ban is on a *joint* rule. The audit resolves the ban into the exact set of [atoms](../glossary.md#atoms) whose support contains both features, and reports their presence, exact mass, and location.

```python
from provable import audit, Policy, between

policy = Policy([between('age', 'group_A'), between('util', 'tenure')], name='fair_lending_demo')
a = audit('model.json', policy, X_val, out_dir='dossier_audit')

print(f"verdict: {a['verdict']['state']}\n{a['verdict']['statement']}\n")
for f in a['findings']:
    print(f"{f['clause']}: atoms={f['n_atoms']}  data_mass={f['data_mass']:.4f}  "
          f"share_of_var={f['share_of_var']:.2%}  peak: {'; '.join(f['profile_top'])}")
```

Output:

```text
verdict: FINDINGS
2 clause(s) matched by rules present in the artifact; masses and locations quantified below. Removal is priced and available via enforce().

between(age, group_A): atoms=326  data_mass=0.1755  share_of_var=163.46%  peak: age>=34.8 (|c| 2.44); group_A>=1 (|c| 6.15)
between(tenure, util): atoms=1466  data_mass=0.0077  share_of_var=7.16%  peak: tenure>=5 (|c| 0.718); util>=0.194 (|c| 0.677)
```

## Reading it

The verdict grammar is a closed set (`PASS`, `FINDINGS`, `DECLINED`), and here it is **FINDINGS**: the artifact contains rules matching both forbidden clauses, quantified and located.

**`between(age, group_A)`**: 326 atoms jointly conditioning on both features, with exact removal price (`data_mass`) 0.1755 on this sample, peaking at the `age>=34.8` threshold. That is the planted pattern, found where it was planted.

**`between(tenure, util)`**: 1,466 atoms, mass 0.0077. The world we generated has *no* util×tenure interaction; boosted trees create incidental joint splits anyway. An audit reports what is **in the artifact**; the [bridge scenario](bridge.md) asks the sieve whether the world backs it up.

Two numbers deserve one sentence each:

- `data_mass` is [`#exact`](../glossary.md#exact-vs-est): the mean squared value of the family's own component on this sample, the associational price you would pay to excise it, quoted before any edit ([A3](a3-cut.md) collects it).
- `share_of_var` **can exceed 100%**, as it does here. Mass is the family component's own (uncentered) mean square; families overlap and covary with the rest of the model, so shares are sizes, not a partition of variance. 163% says: this family's component is larger than the whole prediction's variance; the rest of the model partially cancels it.

`audit` never trains, never touches the model, and works on an external artifact you did not build. With `out_dir=` it emits the same tamper-evident dossier as `enforce` (`evidence.json`, `certificate.md`, `certificate.html`), with the verdict, findings, and the non-removable limits panel ([Verdicts](../manual/verdicts.md)).

## What's guaranteed

Presence/absence and exact mass of the named rule families in this artifact on this sample: an arithmetic fact, independently re-checkable ([A4](a4-verify.md)). Not established, by construction and stated on every certificate: proxies, causal effects, fairness of outcomes, off-distribution behavior.

---

*What the papers add*: the family calculus, the policy algebra, and what no audit of a trained model can ever tell you (the wall: mediation, confounding, deployment shift, the model's own training bias): Paper 1, [10.5281/zenodo.21819690](https://doi.org/10.5281/zenodo.21819690).
