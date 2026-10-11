"""calculus.py -- the transported exact calculus on a loaded artifact.

Transport theorem in code: each leaf box PROD_f [lo_f, hi_f) expands exactly into signed
conjunctions of half-line indicators 1[x_f >= t] via
    1[lo <= x < hi] = 1[x >= lo] - 1[x >= hi]   (1[x >= -inf] = 1, 1[x >= +inf] = 0),
so the whole artifact equals base_score + SUM_atoms coef * AND_f 1[x_f >= t_f]. Atoms are
keyed by frozenset((feat, thr), ...); one threshold per feature per atom by construction.
Everything downstream (audit families, prices, surgery, probes) operates on this dict.
"""
import itertools
import numpy as np


class Spectrum:
    """Exact transported atom spectrum of one artifact, with family selection and prices."""

    def __init__(self, adapter, max_atoms=2_000_000):
        self.ad = adapter
        self.atoms = self._transport(adapter.leaf_boxes(), adapter.base_score, max_atoms)
        self.features_used = sorted({f for A in self.atoms for f, _ in A})

    # ------------------------------------------------------------- transport
    @staticmethod
    def _transport(boxes, base_score, max_atoms):
        est = sum(2 ** sum(1 for lo, hi in b.values() if np.isfinite(lo) and np.isfinite(hi)) for b, _ in boxes)
        if est > max_atoms: raise RuntimeError(f'transport expansion estimate {est} > max_atoms={max_atoms}')
        atoms = {frozenset(): float(base_score)}
        for box, w in boxes:
            factors = []
            for f, (lo, hi) in box.items():
                terms = []
                if np.isfinite(lo): terms.append(((f, float(lo)), +1.0))
                else: terms.append((None, +1.0))
                if np.isfinite(hi): terms.append(((f, float(hi)), -1.0))
                else: terms.append((None, 0.0))
                factors.append([t for t in terms if t[1] != 0.0])
            for combo in itertools.product(*factors) if factors else [()]:
                key = frozenset(c for c, _ in combo if c is not None)
                sgn = 1.0
                for _, s in combo: sgn *= s
                atoms[key] = atoms.get(key, 0.0) + sgn * w
        return {A: v for A, v in atoms.items() if v != 0.0 or not A}

    # ------------------------------------------------------------------ eval
    def evaluate(self, X, atoms=None):
        """Exact evaluation of an atom dict at rows (default: full spectrum). Predicates
        are evaluated in float32, matching xgboost routing (x < c in f32 -> left), so a
        row on a threshold routes identically here and in the stock engine."""
        Xf = np.asarray(X, dtype=np.float32)
        out = np.zeros(len(Xf))
        for A, c in (atoms if atoms is not None else self.atoms).items():
            if not A: out += c; continue
            m = np.ones(len(Xf), dtype=bool)
            for f, t in A: m &= Xf[:, f] >= np.float32(t)
            out += c * m
        return out

    # ---------------------------------------------------------------- family
    def support(self, A): return frozenset(f for f, _ in A)

    def family(self, feats=None, involving=None, min_order=2):
        """Select atoms: feats -> support contains ALL listed features; involving ->
        support contains the feature AND |support| >= min_order."""
        if feats is not None:
            fs = frozenset(feats)
            return {A: c for A, c in self.atoms.items() if fs <= self.support(A)}
        return {A: c for A, c in self.atoms.items() if involving in self.support(A) and len(self.support(A)) >= min_order}

    def top(self, k=20):
        return sorted(((A, c) for A, c in self.atoms.items() if A), key=lambda kv: -abs(kv[1]))[:k]

    # ---------------------------------------------------------------- prices
    def data_mass(self, fam, X):
        """Exact associational price of excising fam on this sample: mean of the removed
        component squared (an arithmetic fact of artifact + sample)."""
        v = self.evaluate(X, atoms={A: c for A, c in fam.items() if A})
        return float(np.mean(v ** 2)), v

    def profile(self, fam, feat):
        """Mass located along one feature's threshold axis: {thr: sum |coef|}."""
        out = {}
        for A, c in fam.items():
            for f, t in A:
                if f == feat: out[t] = out.get(t, 0.0) + abs(c)
        return dict(sorted(out.items()))

    # ------------------------------------------------------------- summaries
    def _fname(self, f):
        try: return str(self.ad.feature_names[f])
        except Exception: return f'f{f}'

    def _rows_for(self, X, fam, data='exact'):
        """Row mask for pricing under the declared data mode: 'exact'
        refuses NaN (R-07 semantics); 'complete-per-family' prices on the
        rows complete in the family's own features and tags DECLARED."""
        Xf = np.asarray(X, dtype=np.float64)
        feats = sorted({f for A in fam for f, _ in A})
        if data == 'exact':
            if np.isnan(Xf).any():
                raise ValueError("R-07: missing values present; exact prices are not defined for NaN routing (pass data='complete-per-family' for a DECLARED price on per-family complete rows)")
            return np.ones(len(Xf), bool)
        if data == 'complete-per-family':
            if not feats: return ~np.isnan(Xf).any(axis=1)
            return ~np.isnan(Xf[:, feats]).any(axis=1)
        raise ValueError("data must be 'exact' or 'complete-per-family'")

    def family_metrics(self, fam, X=None, data='exact'):
        """The four labeled quantities of one family. magnitude: sum of
        absolute coefficients, sample-free ceiling. removal_cost: mean
        squared value of the family's own component on the sample, the
        exact price of excising it, quoted before any edit. share: removal
        cost over Var of the margin on the same rows. joint_value: signed
        coefficient sum, the family's contribution where every predicate
        fires; not an importance."""
        body = {A: c for A, c in fam.items() if A}
        out = {'n_atoms': len(body), 'magnitude': float(sum(abs(c) for c in body.values())), 'joint_value': float(sum(body.values())), 'removal_cost': None, 'share': None, 'rows_used': None}
        if X is not None and body:
            mask = self._rows_for(X, body, data=data)
            Xs = np.asarray(X, dtype=np.float64)[mask]
            out['rows_used'] = int(mask.sum())
            if len(Xs):
                mass, _ = self.data_mass(body, np.nan_to_num(Xs) if data != 'exact' else Xs)
                out['removal_cost'] = float(mass)
                var = float(np.var(self.ad.predict_margin(np.nan_to_num(Xs) if data != 'exact' else Xs)))
                out['share'] = mass / var if var > 0 else 0.0
        return out

    def describe(self, X=None, top=10):
        """The spectrum in one block: atom counts by interaction order,
        features used, and the top families (by removal cost when a
        sample is given, by magnitude otherwise, labeled)."""
        orders = {}
        for A in self.atoms:
            orders[len(A)] = orders.get(len(A), 0) + 1
        return {'atoms': sum(1 for A in self.atoms if A), 'by_order': {k: orders[k] for k in sorted(orders) if k > 0}, 'features_used': len(self.features_used), 'intercept_margin': float(self.atoms.get(frozenset(), 0.0)), 'ranking': 'removal_cost' if X is not None else 'magnitude (sample-free ceiling)'}

    def families(self, X=None, by='removal_cost', top=20, min_order=1, data='exact'):
        """Every distinct support as one family row with the labeled
        quantities; single call, no user loops over atoms. by is
        'removal_cost' (needs X) or 'magnitude'."""
        import pandas as pd
        groups = {}
        for A, c in self.atoms.items():
            if not A or len(A) < min_order: continue
            groups.setdefault(self.support(A), {})[A] = c
        if by == 'removal_cost' and X is None:
            by = 'magnitude'
        rows = []
        for sup, fam in groups.items():
            m = self.family_metrics(fam, X=X, data=data)
            m['support'] = ' * '.join(sorted(self._fname(f) for f in sup))
            m['order'] = len(sup)
            rows.append(m)
        key = by if by in ('removal_cost', 'magnitude', 'joint_value') else 'magnitude'
        rows.sort(key=lambda r: -(r[key] if r.get(key) is not None else -np.inf))
        cols = ['support', 'order', 'n_atoms', 'magnitude', 'removal_cost', 'share', 'joint_value', 'rows_used']
        df = pd.DataFrame(rows[:int(top)])
        return df[[c for c in cols if c in df.columns]] if len(df) else df

    def describe_family(self, fam, X=None, data='exact'):
        m = self.family_metrics(fam, X=X, data=data)
        lines = [f"magnitude (sum abs coefficient, sample-free ceiling): {m['magnitude']:.6g}"]
        if m['removal_cost'] is not None:
            lines.append(f"removal cost (mean squared family component on this data; priced before any edit): {m['removal_cost']:.6g}")
            lines.append(f"share of margin variance: {m['share']:.4%} (rows used: {m['rows_used']})")
        lines.append(f"joint value (signed sum where every predicate fires; not an importance): {m['joint_value']:.6g}")
        return '\n'.join(lines)

    def as_column(self, fam):
        """The bridge: the family's own component as an exact column
        builder with the artifact's float32 routing semantics. Feed the
        returned callable's output to the instrument to ask whether the
        model's pattern is in the world."""
        body = {A: c for A, c in fam.items() if A}
        def build(X):
            return self.evaluate(X, atoms=body)
        build.n_atoms = len(body)
        return build

    # ----------------------------------------------------------------- probe
    @staticmethod
    def grid_probe(predict, thr_i, thr_j, i, j, contexts, tol_context=None):
        """Behavioral completeness probe for pair (i, j): consecutive-cell second
        differences of predict over the full (i, j) threshold grid at each context row.
        Zero everywhere <=> no i x j interaction at the artifact's own resolution.
        Returns max |dd|."""
        vi = Spectrum._probe_values(thr_i); vj = Spectrum._probe_values(thr_j)
        C = np.asarray(contexts, dtype=np.float64)
        n, mi, mj = len(C), len(vi), len(vj)
        if n == 0: return 0.0
        cells = max(mi * mj, 1)
        auto = max(1, int(256e6 / (8.0 * max(C.shape[1], 1)) / cells))
        chunk = tol_context if isinstance(tol_context, int) and tol_context > 0 else min(n, auto)
        best = 0.0
        for s in range(0, n, chunk):
            Cb = C[s:s + chunk]
            nb = len(Cb)
            G = np.repeat(Cb, mi * mj, axis=0)
            G[:, i] = np.tile(np.repeat(vi, mj), nb)
            G[:, j] = np.tile(vj, mi * nb)
            P = predict(G).reshape(nb, mi, mj)
            dd = P[:, 1:, 1:] - P[:, 1:, :-1] - P[:, :-1, 1:] + P[:, :-1, :-1]
            if dd.size: best = max(best, float(np.max(np.abs(dd))))
        return best

    @staticmethod
    def _probe_values(thr):
        thr = np.asarray(sorted(thr), dtype=np.float64)
        if len(thr) == 0: return np.array([0.0])
        return np.concatenate([[thr[0] - 1.0], (thr[:-1] + thr[1:]) / 2.0 if len(thr) > 1 else [], [thr[-1] + 1.0]]) if len(thr) > 1 else np.array([thr[0] - 1.0, thr[0] + 1.0])

    @staticmethod
    def grid_contexts(thresholds, num_feature, n, seed):
        """Data-free probe contexts: random corners of the artifact's own grid."""
        rng = np.random.default_rng(seed)
        C = np.zeros((n, num_feature))
        for f in range(num_feature):
            v = Spectrum._probe_values(thresholds.get(f, []))
            C[:, f] = rng.choice(v, size=n)
        return C
