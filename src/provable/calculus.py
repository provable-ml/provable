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
        G = np.repeat(C, mi * mj, axis=0)
        G[:, i] = np.tile(np.repeat(vi, mj), n)
        G[:, j] = np.tile(vj, mi * n)
        P = predict(G).reshape(n, mi, mj)
        dd = P[:, 1:, 1:] - P[:, 1:, :-1] - P[:, :-1, 1:] + P[:, :-1, :-1]
        return float(np.max(np.abs(dd))) if dd.size else 0.0

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
