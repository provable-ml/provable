# C9 · Twenty questions for a missing page

*You cannot name the thing your panel is missing — the wall of the trilogy is exactly that its name is unreachable. But a missing page still casts a shadow, and a shadow can be interrogated: which rhythms it keeps, whether it persists, whether it is one thing across a frame or many. `characterize` plays the full Twenty Questions charter against the [C8](c8-shadow.md) shadow and files the answer sheet: every answerable row with its number and band, every walled row with the theorem that walls it. Interrogation is priced at zero on the certification ledger (Def 7.2) — asking costs nothing because no selection event is created. Everything here is on a synthetic panel.*

An **atom** is a declared, decision-time-attested question column (Def 7.1): a Fourier line at a named period, a weekday indicator, the shadow's own lag, another member's shadow, an archive quantity with its lag class stated. Each family is residualized against the licensed span — the intercept and the shadow's own `p_hat` — and orthonormalized within the family in the empirical inner product, with the conditioning norm of the change of basis quoted next to any band it scales (Thm 7.3(ii)). Deterministic frame families (Fourier, calendar) carry the *exact* hypothesis-free band of Prop 2.8, uniform over the family by a union bound; data families (own lags, cross-members, archive) carry the clustered plug-in band of Thm 8.5(iii). Every printed number says which it is.

```python
import numpy as np
import pandas as pd
import provable as pv

rng = np.random.default_rng(7)
n = 7500
X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
base = (X["f0"] >= 0.3).astype(float) * 0.80 + ((X["f1"] >= 0) & (X["f2"] >= 0)).astype(float) * 0.12
y = (rng.random(n) < np.clip(base + 0.03, 0, 1)).astype(float)
clock = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
fit = slice(0, 5000)
rep = pv.certify(X[fit], y[:5000], q=0.10, seed=0,
                 clock=(clock[fit], clock[fit] + pd.Timedelta(hours=1)))
sh = pv.shadow(rep, X[5002:].reset_index(drop=True), y[5002:], clock[5002:])

e = sh.rows["e"].to_numpy()
atoms = {
    "archive": {"book_flow": {"values": 0.3 * e + rng.normal(size=sh.m),
                              "lag_class": "post-decision"}},
    "state": {"f3_level": {"values": X["f3"].to_numpy()[5002:],
                           "quantiles": (0.25, 0.5, 0.75)}},
}
ch = pv.characterize(sh, atoms=atoms)
print(ch)
```

```text
-- Characterization (interrogation is priced at zero on the certification ledger, Def 7.2) --
+----------+---------------------+------+------------------+----------------------+--------------------------------------------------------------+
| member   | family              |   K' |   explained mass |   share of ||e||^2_n | band                                                         |
|----------+---------------------+------+------------------+----------------------+--------------------------------------------------------------|
| unit     | temporal (Fourier)  |    4 |      0.000216813 |               0.0024 | exact hypothesis-free (Prop 2.8; radius x conditioning 1.01) |
| unit     | temporal (own lags) |    4 |      4.61026e-05 |               0.0005 | plug-in (clustered, Thm 8.5(iii))                            |
| unit     | calendar            |    6 |      0.00025382  |               0.0029 | exact hypothesis-free (Prop 2.8; radius x conditioning 1.02) |
| unit     | archive             |    1 |      0.000658586 |               0.0074 | plug-in (clustered, Thm 8.5(iii))                            |
+----------+---------------------+------+------------------+----------------------+--------------------------------------------------------------+
joint share over all families = 0.0132 [exact ledger, Thm 7.3(i)]; remainder behind the wall = 0.9868 of ||e||^2_n [Thm 7.3(iii); the external part of it is walled, Thm 2.11]
+----------+---------------------+------------------+------------+-----------+---------+
| member   | family              | strongest atom   |   beta_hat | band      | grade   |
|----------+---------------------+------------------+------------+-----------+---------|
| unit     | temporal (Fourier)  | day:cos          |    -0.0112 | +/-0.0642 | exact   |
| unit     | temporal (own lags) | lag 2            |    -0.0065 | +/-0.0124 | plug-in |
| unit     | calendar            | weekday 4        |     0.0118 | +/-0.0673 | exact   |
| unit     | archive             | book_flow        |     0.0257 | +/-0.0116 | plug-in |
+----------+---------------------+------------------+------------+-----------+---------+
[a strongest-atom quote is a maximum over K selections; an honest widened quote adds sqrt(2 log K) noise scale, Thm 5.6(iv) / Thm 5.7]
+-------+-----------------+--------------------------------------------------------------------------------------------+
|   lag |   autocorr of e | band (uniform over the set; long-run plug-in at the printed Bartlett bandwidth, Thm 8.5)   |
|-------+-----------------+--------------------------------------------------------------------------------------------|
|     1 |         -0.0117 | [-0.0616, +0.0382] (b=9)                                                                   |
|     2 |         -0.017  | [-0.0736, +0.0395] (b=9)                                                                   |
|     7 |         -0.0095 | [-0.0670, +0.0481] (b=9)                                                                   |
|    14 |         -0      | [-0.0509, +0.0508] (b=9)                                                                   |
+-------+-----------------+--------------------------------------------------------------------------------------------+
persistence verdict: fast-decaying (half-life h* = 14 by band inversion, Thm 8.6(iv))
archive aim ceiling lam_arch = 0.0861 (+/-0.0543) on the residual-variance scale [est, Def 4.14]

-- the Twenty Questions charter, filed (Table 4; Thm 9.5) --
+--------------------------------+--------------------------+------------------------------------------------+--------------------+
| question                       | status                   | value and band                                 | law                |
|--------------------------------+--------------------------+------------------------------------------------+--------------------|
| Which temporal band does the   | answerable               | strongest line day:cos, beta -0.0112 +/-       | Cor 7.6            |
| shadow load                    |                          | 0.0642 [exact]                                 |                    |
| Does it vary as often as the   | answerable               | residual band-mass curve vs the outcome's own  | Cor 7.6            |
| outcome                        |                          | curve on the declared grid: sup gap 0.000      |                    |
|                                |                          | [est]                                          |                    |
| Frame-wide, group, or          | not applicable (single-  |                                                | Cor 7.7            |
| idiosyncratic                  | unit shadow; the frame   |                                                |                    |
|                                | clause)                  |                                                |                    |
| Does its factor loading rotate | answerable               | loading over 4 declared buckets of one state   | Cor 7.8            |
|                                |                          | column: sign flip False; concentration 1.43    |                    |
|                                |                          | [est, grid-priced]                             |                    |
| Fast-decaying or persistent    | answerable exactly for   | fast-decaying; autocorr at lag 1 = -0.012 in   | Thm 8.6(iv)        |
|                                | the shadow; to the       | [-0.062, +0.038]; h* = 14 [plug-in, uniform    |                    |
|                                | envelope for the score   | over the set]                                  |                    |
| Does past surprise feed it     | answerable at the        | own-lag family explained share 0.0005 at the   | Thm 8.3, Rem 8.4   |
|                                | archive tier; hypothesis | archive tier [plug-in]; the world tier stays a |                    |
|                                | at the world tier        | hypothesis                                     |                    |
| How much of it is buildable    | answerable; lower-       | lam_arch = 0.0861 (+/-0.0543) on the residual- | Def 4.14, Cor 4.15 |
| from the archive               | boundable by exhibition  | variance scale [est]                           |                    |
| Is the un-archived part noise  | walled; only the sum     |                                                | Thm 2.11           |
| or signal                      | answerable (the          |                                                |                    |
|                                | remainder line above)    |                                                |                    |
| What is its name               | walled at every rung;    |                                                | Thm 3.2            |
|                                | any test at exactly one  |                                                |                    |
|                                | half                     |                                                |                    |
| Its own dynamics beyond the    | walled before; measured  |                                                | Thm 9.4(ii)        |
| shadow pathway                 | upon acquisition         |                                                |                    |
| Can any purchasable column     | walled above the archive |                                                | Thm 3.4            |
| reach the specification        | ceiling; rows 7 and the  |                                                |                    |
|                                | Screening carry the      |                                                |                    |
|                                | answerable part          |                                                |                    |
| Is the external component      | walled in both           |                                                | Lem 7.4, Thm 7.5   |
| serially structured            | directions at innovative |                                                |                    |
|                                | lags; white by theorem   |                                                |                    |
|                                | otherwise                |                                                |                    |
+--------------------------------+--------------------------+------------------------------------------------+--------------------+
[rotation grid: the atom count multiplies across the product grid and the multiplicity radius scales with it, stated once here (Cor 7.8)]
```

The charter table is the page's spine — the posted Table 4, filed. Row by row, what the run above put in it:

1. **Which temporal band** — answerable (Cor 7.6). The strongest Fourier line prints with the exact hypothesis-free band; here nothing exceeds it, and the ledger says the same in mass terms.
2. **Does it vary as often as the outcome** — answerable (Cor 7.6): two *identified* normalized band-mass curves compared on the declared grid, the shadow's against the outcome's own, quoted as a sup gap. Comparing identified curves never opens the wall.
3. **Frame-wide, group, or idiosyncratic** — *not applicable* on a single-unit shadow; the frame clause of Cor 7.7 requires a multi-unit frame on a shared clock. The frame run below answers it.
4. **Does its factor loading rotate** — answerable (Cor 7.8) once a state column and its bucket quantiles are declared. The grid caveat is stated once: the atom count multiplies across the product grid and the multiplicity radius scales with it.
5. **Fast-decaying or persistent** — the persistence ledger: the shadow's autocorrelation at the declared lags, bands uniform over the set from the long-run plug-in at the printed Bartlett bandwidth (Thm 8.5), verdict by band inversion with the half-life `h*` (Thm 8.6(iv)). Exact for the shadow; for the unobserved score itself, answerable only to the envelope.
6. **Does past surprise feed it** — answerable at the archive tier (the own-lag family's explained share, Thm 8.3); whether the *world* feeds it stays a hypothesis (Rem 8.4).
7. **How much is buildable from the archive** — `lam_arch`, the archive aim ceiling (Def 4.14): the declared archive family's explained share on the residual-variance scale, lower-boundable by exhibition (Cor 4.15). This is the number [C10](c10-screen.md)'s trichotomy takes.
8. -- 12. **Walled, and filed as walled**: the noise/signal split of the un-archived part (Thm 2.11 — only the sum is answerable, and the ledger's remainder line *is* that sum), the name (Thm 3.2 — any identity test sits at exactly one half; the record carries no field for a candidate identity, so the absolute rule is enforced at the type level), dynamics beyond the shadow pathway (Thm 9.4(ii) — measured upon acquisition, not before), reachability of a specification above the ceiling (Thm 3.4 — rows 7 and the Screening carry the answerable part), and serial structure of the external component at innovative lags (Lem 7.4, Thm 7.5 — white by theorem otherwise, walled in both directions where it matters).

Two honesty clauses ride every read: the strongest-atom table is a maximum over `K` selections, so an honest widened quote adds `sqrt(2 log K)` noise scale (Thm 5.6(iv), Thm 5.7); and the per-family exact ledger identity — explained mass equals the fitted drop, to machine precision (Thm 7.3(i)) — is asserted on every run, not sampled.

## The frame: one charter per member

Join two certified targets' shadows on a shared clock and the frame path runs the shared block once — the residual Gram, its top eigenvalue against the independence reference, partial couplings with uniform bands, the scope shares — and then files one full charter per member, each member's cross-sectional family being the *other* members' shadows.

```python
rng = np.random.default_rng(3)
n = 2400
Xa = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
Xb = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
shared = rng.normal(size=n)
ya = (rng.random(n) < np.clip(0.15 + 0.5 * (Xa["f0"] > 0.4) + 0.3 * (shared > 0), 0, 1)).astype(float)
yb = (rng.random(n) < np.clip(0.15 + 0.5 * (Xb["f1"] > 0.4) + 0.3 * (shared > 0), 0, 1)).astype(float)
clock = pd.date_range("2026-02-01", periods=n, freq="h", tz="UTC")

fit = slice(0, 1400)
rep_a = pv.certify(Xa[fit], ya[:1400], q=0.10, seed=0)
rep_b = pv.certify(Xb[fit], yb[:1400], q=0.10, seed=1)
sh_a = pv.shadow(rep_a, Xa[1402:].reset_index(drop=True), ya[1402:], clock[1402:])
sh_b = pv.shadow(rep_b, Xb[1402:].reset_index(drop=True), yb[1402:], clock[1402:])

frame = pv.Shadow.join([sh_a, sh_b])
ch = pv.characterize(frame, lags=(1, 2), periods=("day",))
print(ch)
```

```text
-- Characterization (interrogation is priced at zero on the certification ledger, Def 7.2) --
+----------+---------------------+------+------------------+----------------------+--------------------------------------------------------------+
| member   | family              |   K' |   explained mass |   share of ||e||^2_n | band                                                         |
|----------+---------------------+------+------------------+----------------------+--------------------------------------------------------------|
| m0       | temporal (Fourier)  |    2 |      0.000429385 |               0.002  | exact hypothesis-free (Prop 2.8; radius x conditioning 1.00) |
| m0       | temporal (own lags) |    2 |      2.68856e-05 |               0.0001 | plug-in (clustered, Thm 8.5(iii))                            |
| m0       | calendar            |    6 |      0.001066    |               0.0049 | exact hypothesis-free (Prop 2.8; radius x conditioning 1.03) |
| m0       | cross-members       |    1 |      0.00154312  |               0.007  | plug-in (clustered, Thm 8.5(iii))                            |
| m1       | temporal (Fourier)  |    2 |      0.00050513  |               0.002  | exact hypothesis-free (Prop 2.8; radius x conditioning 1.00) |
| m1       | temporal (own lags) |    2 |      0.000281188 |               0.0011 | plug-in (clustered, Thm 8.5(iii))                            |
| m1       | calendar            |    6 |      0.00161484  |               0.0065 | exact hypothesis-free (Prop 2.8; radius x conditioning 1.03) |
| m1       | cross-members       |    1 |      0.00175486  |               0.007  | plug-in (clustered, Thm 8.5(iii))                            |
+----------+---------------------+------+------------------+----------------------+--------------------------------------------------------------+
joint share over all families = 0.0153 [exact ledger, Thm 7.3(i)]; remainder behind the wall = 0.9847 of ||e||^2_n [Thm 7.3(iii); the external part of it is walled, Thm 2.11]
+----------+---------------------+------------------+------------+-----------+---------+
| member   | family              | strongest atom   |   beta_hat | band      | grade   |
|----------+---------------------+------------------+------------+-----------+---------|
| m0       | temporal (Fourier)  | day:sin          |    -0.0194 | +/-0.0938 | exact   |
| m0       | temporal (own lags) | lag 1            |    -0.0047 | +/-0.0279 | plug-in |
| m0       | calendar            | weekday 1        |     0.027  | +/-0.1081 | exact   |
| m0       | cross-members       | member 1         |     0.0393 | +/-0.0289 | plug-in |
| m1       | temporal (Fourier)  | day:sin          |     0.0195 | +/-0.0938 | exact   |
| m1       | temporal (own lags) | lag 2            |     0.0162 | +/-0.0329 | plug-in |
| m1       | calendar            | weekday 2        |     0.0239 | +/-0.1081 | exact   |
| m1       | cross-members       | member 0         |     0.0419 | +/-0.0309 | plug-in |
+----------+---------------------+------------------+------------+-----------+---------+
[a strongest-atom quote is a maximum over K selections; an honest widened quote adds sqrt(2 log K) noise scale, Thm 5.6(iv) / Thm 5.7]
+-------+-----------------+--------------------------------------------------------------------------------------------+
|   lag |   autocorr of e | band (uniform over the set; long-run plug-in at the printed Bartlett bandwidth, Thm 8.5)   |
|-------+-----------------+--------------------------------------------------------------------------------------------|
|     1 |          0.0103 | [-0.0649, +0.0855] (b=7)                                                                   |
|     2 |         -0.0039 | [-0.0754, +0.0676] (b=7)                                                                   |
+-------+-----------------+--------------------------------------------------------------------------------------------+
persistence verdict: fast-decaying (half-life h* = 2 by band inversion, Thm 8.6(iv))
frame block: top residual-Gram eigenvalue 1.0839 vs independence reference 1.0915; verdict frame-wide [Cor 7.7]; shares frame-wide 0.08 / group 0.00 / idiosyncratic 0.92
archive aim ceiling lam_arch = 0.0000 (+/-0.0000) on the residual-variance scale [est, Def 4.14]

-- charter, member 0 (Table 4; Thm 9.5) --
+--------------------------------+--------------------------+------------------------------------------------+--------------------+
| question                       | status                   | value and band                                 | law                |
|--------------------------------+--------------------------+------------------------------------------------+--------------------|
| Which temporal band does the   | answerable               | strongest line day:sin, beta -0.0194 +/-       | Cor 7.6            |
| shadow load                    |                          | 0.0938 [exact]                                 |                    |
| Does it vary as often as the   | answerable               | two identified curves compared; outcome column | Cor 7.6            |
| outcome                        |                          | not carried on this frame                      |                    |
| Frame-wide, group, or          | answerable (multi-unit   | verdict frame-wide; shares frame-wide 0.08 /   | Cor 7.7            |
| idiosyncratic                  | frame)                   | group 0.00 / idiosyncratic 0.92; top eig 1.084 |                    |
|                                |                          | vs ref 1.092 [plug-in]                         |                    |
| Does its factor loading rotate | answerable when a state  |                                                | Cor 7.8            |
|                                | column is declared; none |                                                |                    |
|                                | this run                 |                                                |                    |
| Fast-decaying or persistent    | answerable exactly for   | fast-decaying; autocorr at lag 1 = +0.010 in   | Thm 8.6(iv)        |
|                                | the shadow; to the       | [-0.065, +0.086]; h* = 2 [plug-in, uniform     |                    |
|                                | envelope for the score   | over the set]                                  |                    |
| Does past surprise feed it     | answerable at the        | own-lag family explained share 0.0001 at the   | Thm 8.3, Rem 8.4   |
|                                | archive tier; hypothesis | archive tier [plug-in]; the world tier stays a |                    |
|                                | at the world tier        | hypothesis                                     |                    |
| How much of it is buildable    | answerable; lower-       | lam_arch = 0.0000 (+/-0.0000) on the residual- | Def 4.14, Cor 4.15 |
| from the archive               | boundable by exhibition  | variance scale [est]                           |                    |
| Is the un-archived part noise  | walled; only the sum     |                                                | Thm 2.11           |
| or signal                      | answerable (the          |                                                |                    |
|                                | remainder line above)    |                                                |                    |
| What is its name               | walled at every rung;    |                                                | Thm 3.2            |
|                                | any test at exactly one  |                                                |                    |
|                                | half                     |                                                |                    |
| Its own dynamics beyond the    | walled before; measured  |                                                | Thm 9.4(ii)        |
| shadow pathway                 | upon acquisition         |                                                |                    |
| Can any purchasable column     | walled above the archive |                                                | Thm 3.4            |
| reach the specification        | ceiling; rows 7 and the  |                                                |                    |
|                                | Screening carry the      |                                                |                    |
|                                | answerable part          |                                                |                    |
| Is the external component      | walled in both           |                                                | Lem 7.4, Thm 7.5   |
| serially structured            | directions at innovative |                                                |                    |
|                                | lags; white by theorem   |                                                |                    |
|                                | otherwise                |                                                |                    |
+--------------------------------+--------------------------+------------------------------------------------+--------------------+
-- charter, member 1 (Table 4; Thm 9.5) --
+--------------------------------+--------------------------+------------------------------------------------+--------------------+
| question                       | status                   | value and band                                 | law                |
|--------------------------------+--------------------------+------------------------------------------------+--------------------|
| Which temporal band does the   | answerable               | strongest line day:sin, beta +0.0195 +/-       | Cor 7.6            |
| shadow load                    |                          | 0.0938 [exact]                                 |                    |
| Does it vary as often as the   | answerable               | two identified curves compared; outcome column | Cor 7.6            |
| outcome                        |                          | not carried on this frame                      |                    |
| Frame-wide, group, or          | answerable (multi-unit   | verdict frame-wide; shares frame-wide 0.08 /   | Cor 7.7            |
| idiosyncratic                  | frame)                   | group 0.00 / idiosyncratic 0.92; top eig 1.084 |                    |
|                                |                          | vs ref 1.092 [plug-in]                         |                    |
| Does its factor loading rotate | answerable when a state  |                                                | Cor 7.8            |
|                                | column is declared; none |                                                |                    |
|                                | this run                 |                                                |                    |
| Fast-decaying or persistent    | answerable exactly for   | fast-decaying; autocorr at lag 1 = +0.018 in   | Thm 8.6(iv)        |
|                                | the shadow; to the       | [-0.070, +0.106]; h* = beyond the set [plug-   |                    |
|                                | envelope for the score   | in, uniform over the set]                      |                    |
| Does past surprise feed it     | answerable at the        | own-lag family explained share 0.0011 at the   | Thm 8.3, Rem 8.4   |
|                                | archive tier; hypothesis | archive tier [plug-in]; the world tier stays a |                    |
|                                | at the world tier        | hypothesis                                     |                    |
| How much of it is buildable    | answerable; lower-       | lam_arch = 0.0000 (+/-0.0000) on the residual- | Def 4.14, Cor 4.15 |
| from the archive               | boundable by exhibition  | variance scale [est]                           |                    |
| Is the un-archived part noise  | walled; only the sum     |                                                | Thm 2.11           |
| or signal                      | answerable (the          |                                                |                    |
|                                | remainder line above)    |                                                |                    |
| What is its name               | walled at every rung;    |                                                | Thm 3.2            |
|                                | any test at exactly one  |                                                |                    |
|                                | half                     |                                                |                    |
| Its own dynamics beyond the    | walled before; measured  |                                                | Thm 9.4(ii)        |
| shadow pathway                 | upon acquisition         |                                                |                    |
| Can any purchasable column     | walled above the archive |                                                | Thm 3.4            |
| reach the specification        | ceiling; rows 7 and the  |                                                |                    |
|                                | Screening carry the      |                                                |                    |
|                                | answerable part          |                                                |                    |
| Is the external component      | walled in both           |                                                | Lem 7.4, Thm 7.5   |
| serially structured            | directions at innovative |                                                |                    |
|                                | lags; white by theorem   |                                                |                    |
|                                | otherwise                |                                                |                    |
+--------------------------------+--------------------------+------------------------------------------------+--------------------+
[frame joint share is the member mean; per-member ledgers above]
```

The two targets were built over one shared driver, and the frame block reads it back: each member's cross-member atom exceeds its band, and the scope decomposition puts the co-moving mass on the frame-wide line — the answer row 3 could not give on a single unit. On a frame the outcome column is not carried per member, so row 2 states the clause instead of a gap, and each member's licensed span is the intercept.

## The leak guard

The target's own realized outcome is not an atom, and neither is anything that descends from it: an atom affinely identical to `y` on the aligned rows, or one whose declared lineage names the target, is refused before any number is computed.

```python
try:
    pv.characterize(sh, atoms={"archive": {"settle": {
        "values": 2.0 * sh.rows["y"].to_numpy() + 3.0, "lag_class": "post-decision"}}})
except ValueError as err:
    print(err)
```

```text
characterize refused: atom 'settle' is affinely identical to the target's realized outcome on the aligned rows (leak guard)
```

An archive atom with neither a lag class nor a clock offset is refused the same way: post-decision quantities must declare theirs, because the charter's archive tier is defined by *when* the question column becomes known, not by where it is stored.
