"""FROZEN TEST ORACLE -- do not edit.

Vendored copy of the original Paper-1 driver (mobius_calculus.py). Used ONLY by the
acceptance ledger (T3) as an independent implementation of the Mobius read: the
transported spectrum of provable must agree with this file's full_spectrum() to
float64 on binary-input models. Not part of any distribution; lives under tests/.
"""
"""mobius_calculus.py -- the exact interaction calculus of `Every Interaction Has a Price'
(trilogy Paper 1), as a single practical driver shared by the conference papers A/B/C.

Implements, on fitted XGBoost tree ensembles over binary inputs:

  spectrum    exact leaf->Mobius extraction (Prop 1) from the saved model JSON, full
              enumeration under a size guard, B-infinity branch-and-bound top-k
              (Def 4 / Prop 6), Mobius->chi (orthonormal) transform under any product
              measure, exact reconstruction from either basis.
  surgery     functional Method C (Thm 7: delete named rules, zero collateral),
              junction-bundle deletion for a pair (Prop 9: parallel trends),
              allowed-corner interpolant for any lattice policy (Prop 46),
              projection edit per fiber (Thm 10(i)/47), marginalization (Prop 13),
              artifact certificates (Thm 49: max forbidden Mobius content = 0).
  pricing     one 2^s-corner pass per context row prices the entire policy lattice
              (Thm 47): pattern masses m(D) under the interventional corner measure nu
              (uniform arms) and a pooled product measure q-hat (empirical S-marginals),
              modular taxes, Pareto frontier, threshold selector with certified
              infeasibility (Thm 51(iii)); fork prices and the identification premium
              for a pair (Thm 29), exponential blindness ratio (Cor 32).
  causal      potential-prediction totals, mediated decomposition (Lem 33),
              total-effect editor (Thm 34), attainable envelope + leverage/vacuity
              (Thm 35), misspecification residual bound (Prop 36), three-layer ND
              environment battery (Thm 43 scope: verdicts `refuted' / `consistent').
  ledger      per-feature Shapley slice quantities (Thm 59(i)/(42)): marginal chi
              coefficient, interaction mass, E[phi_f^2]; Prop-11 closed-form residual
              dependence predicted before any edit (Cor 12).

Scope decisions, disclosed (tool-version ledger for the papers):
  * Edits are FUNCTIONAL: an edited model is base-booster predictions plus exact
    correction terms (closures). Booster-JSON write-back (appended trees) is not in
    this version; every guarantee checked here is the functional one, which is the
    basis-free statement all Paper-1 theorems are phrased in. Booster predictions are
    float32, so identities routed through the engine hold at ~1e-4 absolute; identities
    computed in spectrum arithmetic hold at float64.
  * base_score is pinned to 0.0 at fit time, so the margin is exactly the tree sum and
    I(emptyset) needs no offset.
  * Binary y runs as a linear probability model on the squared-error margin (house
    convention); accuracy/AUC reported by thresholding when asked.
  * `Pooled q-hat' associational corner measure = product of empirical S-marginals
    (clipped to [1e-6, 1-1e-6]); the interventional measure nu is uniform corners with
    contexts from the data. Real fibers are singletons, so the pooled product measure
    is the disclosed estimator of the associational price (Adult convention).
  * Full enumeration guarded by max_enum on sum_leaves 2^|Z|; beyond it use topk().
  * Seeds: engine 9000+seed, any split 4000+seed. Given a seed, runs are deterministic.

Conventions: statsmodels available for experiment scripts (not needed here), sklearn
AUC helper, tabulate('psql') console tables, plotly left to the per-paper figure
scripts. Single class; `python mobius_calculus.py' runs the seeded identity self-test
ledger that every later experiment number stands on.
"""
import json
import itertools
import heapq
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
import xgboost as xgb


class MobiusCalculus:
    """Exact interaction calculus on a fitted tree ensemble over binary inputs."""

    def __init__(self, seed=0, depth=5, trees=250, lr=0.1, reg_lambda=1.0, objective='reg:squarederror', max_enum=2_000_000):
        self.seed, self.depth, self.trees, self.lr, self.reg_lambda, self.objective, self.max_enum = seed, depth, trees, lr, reg_lambda, objective, max_enum
        self.bst_, self.cols_, self.leaves_, self.spec_ = None, None, None, None

    # ------------------------------------------------------------------ data
    def binarize(self, X_df, n_cuts=3, max_onehot=12):
        """Mixed-dtype DataFrame -> uint8 binary frame. Numerics: quantile cuts (col>=thr); already-{0,1} numerics pass through; categoricals/bools: one-hots by frequency (col==level), capped. NaN -> every predicate False. Names sanitized for xgboost."""
        def clean(s): return str(s).replace('<', '_lt_').replace('[', '(').replace(']', ')')
        out = {}
        for c in X_df.columns:
            v = X_df[c]
            if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v):
                vals = v.dropna().unique()
                if set(np.unique(vals)).issubset({0, 1}): out[clean(c)] = v.fillna(0).astype(np.uint8); continue
                qs = np.unique(np.nanquantile(v.astype(float), np.linspace(0, 1, n_cuts + 2)[1:-1]))
                for t in qs: out[clean(f"{c}>={t:.6g}")] = (v.astype(float) >= t).fillna(False).astype(np.uint8)
            else:
                lv = v.astype('object').fillna('__nan__').value_counts().index[:max_onehot]
                for l in lv: out[clean(f"{c}=={l}")] = (v.astype('object') == l).fillna(False).astype(np.uint8)
        return pd.DataFrame(out, index=X_df.index)

    # ------------------------------------------------------------------- fit
    def fit(self, Xb, y, groups=None, trees=None):
        """Fit the booster on a binary frame. groups: optional list of feature-index lists -> xgboost interaction_constraints (Thm 41). base_score pinned 0.0 so margin = tree sum."""
        self.cols_ = list(Xb.columns) if hasattr(Xb, 'columns') else [f"f{i}" for i in range(Xb.shape[1])]
        X = np.ascontiguousarray(np.asarray(Xb, dtype=np.float32))
        p = {'max_depth': self.depth, 'eta': self.lr, 'lambda': self.reg_lambda, 'objective': self.objective, 'tree_method': 'hist', 'base_score': 0.0, 'seed': 9000 + self.seed, 'nthread': 4}
        if groups is not None: p['interaction_constraints'] = str([[int(i) for i in g] for g in groups])
        self.bst_ = xgb.train(p, xgb.DMatrix(X, label=np.asarray(y, dtype=np.float64), feature_names=self.cols_), num_boost_round=(trees or self.trees))
        self.leaves_, self.spec_ = self._parse_leaves(), None
        return self

    def predict(self, X):
        """Margin (= exact tree sum, float32) of the fitted booster."""
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        return self.bst_.predict(xgb.DMatrix(X, feature_names=self.cols_), output_margin=True).astype(np.float64)

    # -------------------------------------------------------------- spectrum
    def _parse_leaves(self):
        """Model JSON -> list of (P frozenset forced-1, Z frozenset forced-0, leaf weight). Binary routing: x < split_condition goes left. Dead branches (feature re-split against its own path) pruned."""
        raw = json.loads(bytearray(self.bst_.save_raw(raw_format='json')))
        leaves = []
        for tr in raw['learner']['gradient_booster']['model']['trees']:
            L, R, SI, SC = tr['left_children'], tr['right_children'], tr['split_indices'], tr['split_conditions']
            stack = [(0, frozenset(), frozenset())]
            while stack:
                nid, P, Z = stack.pop()
                if L[nid] == -1: leaves.append((P, Z, float(SC[nid]))); continue
                f, c = int(SI[nid]), float(SC[nid])
                s0, s1 = (L[nid] if 0.0 < c else R[nid]), (L[nid] if 1.0 < c else R[nid])
                if s0 == s1: stack.append((s0, P, Z)); continue
                if f not in Z: stack.append((s1, P | {f}, Z))
                if f not in P: stack.append((s0, P, Z | {f}))
        return leaves

    def leaf_I(self, A):
        """Closed-form I(A) from the leaf formula (Prop 1), no enumeration."""
        A = frozenset(A)
        return float(sum(w * ((-1) ** len(A - P)) for P, Z, w in self.leaves_ if P <= A <= (P | Z)))

    def full_spectrum(self):
        """Enumerate the exact Mobius spectrum, dict frozenset->I (cached). Guarded by max_enum on the raw interval mass sum_leaves 2^|Z|."""
        if self.spec_ is not None: return self.spec_
        total = sum(2 ** len(Z) for _, Z, _ in self.leaves_)
        if total > self.max_enum: raise RuntimeError(f"enumeration size {total} > max_enum={self.max_enum}; use topk()")
        I = {}
        for P, Z, w in self.leaves_:
            Zl = sorted(Z)
            for r in range(len(Zl) + 1):
                for B in itertools.combinations(Zl, r):
                    A = P | frozenset(B); I[A] = I.get(A, 0.0) + w * ((-1) ** r)
        self.spec_ = {A: v for A, v in I.items() if v != 0.0}
        if frozenset() not in self.spec_: self.spec_[frozenset()] = 0.0
        return self.spec_

    def topk(self, k):
        """Top-k |I| by best-first branch-and-bound under the monotone admissible bound B_infinity (Sec 3.4); exact selection under the magnitude objective. Returns [(A, I)] sorted by |I| desc."""
        leafV = [(P, P | Z, abs(w)) for P, Z, w in self.leaves_]
        def bound(A): return sum(aw for P, V, aw in leafV if A <= V)
        feats = sorted(set().union(*[V for _, V, _ in leafV])) if leafV else []
        heap, best, thresh = [(-bound(frozenset()), 0, frozenset())], [], 0.0
        cnt = 1
        while heap:
            nb, _, A = heapq.heappop(heap)
            if -nb < thresh and len(best) >= k: break
            iA = self.leaf_I(A)
            if len(best) < k: heapq.heappush(best, (abs(iA), tuple(sorted(A)), iA))
            elif abs(iA) > best[0][0]: heapq.heapreplace(best, (abs(iA), tuple(sorted(A)), iA))
            thresh = best[0][0] if len(best) >= k else 0.0
            mx = max(A) if A else -1
            for f in feats:
                if f <= mx: continue
                A2 = A | {f}; b2 = bound(A2)
                if b2 >= thresh or len(best) < k: heapq.heappush(heap, (-b2, cnt, A2)); cnt += 1
        return sorted([(frozenset(t), v) for _, t, v in best], key=lambda x: -abs(x[1]))

    def chi_spectrum(self, p=None, keys=None, atom_floor=0.0):
        """Exact Mobius->chi transform under a product measure (marginals p: array over features, default 1/2): c(S) = sum_{B>=S} I(B) prod_{S} sqrt(pq) prod_{B\\S} p. Optional keys restricts accumulation to a given support set (plus the constant; used for bootstrap-t universes); optional atom_floor drops Mobius atoms below it (inexact; disclose if used). Weights via a vectorized kron over (p_i, sqrt(p_i q_i)) -- identical sums, reordered. Returns dict frozenset->c."""
        spec, n = self.full_spectrum(), len(self.cols_)
        p = np.full(n, 0.5) if p is None else np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        sq = np.sqrt(p * (1 - p))
        K = None if keys is None else set(keys)
        subs_cache, c = {}, {}
        for B, IB in spec.items():
            if atom_floor and abs(IB) < atom_floor: continue
            Bl = sorted(B); k = len(Bl)
            if k == 0:
                c[frozenset()] = c.get(frozenset(), 0.0) + IB; continue
            w = np.array([1.0])
            for i in Bl: w = np.concatenate([w * p[i], w * sq[i]])
            if k not in subs_cache: subs_cache[k] = [tuple(j for j in range(k) if (m >> j) & 1) for m in range(1 << k)]
            for m, pos in enumerate(subs_cache[k]):
                S = frozenset(Bl[j] for j in pos)
                if K is not None and S and S not in K: continue
                c[S] = c.get(S, 0.0) + IB * float(w[m])
        return c

    def mobius_predict(self, spec, X):
        """Evaluate a Mobius spectrum dict at rows (exact, float64)."""
        X = np.asarray(X, dtype=np.float64); out = np.zeros(len(X))
        for A, v in spec.items():
            if v == 0.0: continue
            out += v * (np.prod(X[:, sorted(A)], axis=1) if A else 1.0)
        return out

    def chi_predict(self, cspec, X, p=None):
        """Evaluate a chi spectrum dict at rows under product marginals p (default 1/2)."""
        X = np.asarray(X, dtype=np.float64); n = X.shape[1]
        p = np.full(n, 0.5) if p is None else np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        sq = np.sqrt(p * (1 - p)); out = np.zeros(len(X))
        for S, v in cspec.items():
            if v == 0.0: continue
            out += v * (np.prod((X[:, sorted(S)] - p[sorted(S)]) / sq[sorted(S)], axis=1) if S else 1.0)
        return out

    # ------------------------------------------------------ differences / audits
    def mixed_diff(self, predf, X, A):
        """(Delta_A predf)(x_{-A}) per row via one stacked call over the 2^|A| corners (Eq 3)."""
        A = sorted(A); k, n = len(A), len(np.asarray(X))
        X = np.asarray(X, dtype=np.float64)
        if k == 0: return predf(X)
        stack = np.repeat(X, 2 ** k, axis=0)
        sgn = np.empty(2 ** k)
        for m in range(2 ** k):
            bits = [(m >> j) & 1 for j in range(k)]
            for j, f in enumerate(A): stack[m::2 ** k, f] = bits[j]
            sgn[m] = (-1) ** (k - sum(bits))
        return (predf(stack).reshape(n, 2 ** k) * sgn).sum(axis=1)

    def dependence(self, predf, j, X):
        """Mean |Delta_j predf| over rows: the audit dependence on feature j."""
        return float(np.mean(np.abs(self.mixed_diff(predf, X, [j]))))

    def predicted_dependence_after(self, removed, j, X):
        """Cor 12: mean |Delta_j| of the model AFTER deleting `removed` rules, computed from the spectrum alone, before any edit: remaining {A ni j} shifted by A\\{j}."""
        spec = self.full_spectrum(); rem = {frozenset(a) for a in removed}
        dj = {}
        for A, v in spec.items():
            if j in A and A not in rem: dj[A - {j}] = dj.get(A - {j}, 0.0) + v
        return float(np.mean(np.abs(self.mobius_predict(dj, np.asarray(X, dtype=np.float64)))))

    # --------------------------------------------------------------- surgery
    def delete_rules(self, atoms):
        """Functional Method C (Thm 7): atoms = {frozenset: I}. Returns predict closure G = F - sum I_A x^A; off-support predictions bit-identical."""
        base = self.predict
        items = [(sorted(A), float(v)) for A, v in atoms.items()]
        def G(X):
            X = np.asarray(X, dtype=np.float64); out = base(X).copy()
            for Al, v in items: out -= v * (np.prod(X[:, Al], axis=1) if Al else 1.0)
            return out
        return G

    def junction_delete(self, i, j, predf=None):
        """Delete-only junction surgery for the pair (Eq 10): G = F - x_i x_j (Delta_ij F)(x_{-ij}), realized by 4 corner calls per row. Agrees with F off the joint corner; parallel-trends value on it (Prop 9)."""
        base = predf or self.predict
        def G(X):
            X = np.asarray(X, dtype=np.float64); d = self.mixed_diff(base, X, [i, j])
            return base(X) - X[:, i] * X[:, j] * d
        return G

    def marginalize(self, feats, pvals, predf=None):
        """Prop 13 iterated: substitute x_f := p_f for each f in feats; F' independent of feats, deploys as a 2^|feats| mixture of corner evaluations."""
        base = predf or self.predict; feats = sorted(feats); k = len(feats)
        pv = np.asarray([pvals[f] if isinstance(pvals, dict) else pvals[i] for i, f in enumerate(feats)], dtype=np.float64)
        def G(X):
            X = np.asarray(X, dtype=np.float64); n = len(X); out = np.zeros(n)
            stack = np.repeat(X, 2 ** k, axis=0); wts = np.empty(2 ** k)
            for m in range(2 ** k):
                w = 1.0
                for j, f in enumerate(feats):
                    b = (m >> j) & 1; stack[m::2 ** k, f] = b; w *= pv[j] if b else 1 - pv[j]
                wts[m] = w
            return (base(stack).reshape(n, 2 ** k) * wts).sum(axis=1)
        return G

    # ---------------------------------------------------------- lattice engine
    def corner_tensor(self, predf, S, X):
        """One pass (Thm 47): predictions at all 2^s S-corner configurations per context row. Returns (T (n,2^s), sigma_row (n,) own-config index). Corner c bit b = value of S[b]."""
        S = sorted(S); s = len(S); X = np.asarray(X, dtype=np.float64); n = len(X)
        stack = np.repeat(X, 2 ** s, axis=0)
        for c in range(2 ** s):
            for b, f in enumerate(S): stack[c::2 ** s, f] = (c >> b) & 1
        T = predf(stack).reshape(n, 2 ** s)
        sigma = np.zeros(n, dtype=int)
        for b, f in enumerate(S): sigma |= (X[:, f].astype(int) << b)
        return T, sigma

    def _mats(self, s, p=None):
        """Fiber transform matrices over 2^s corners: Mobius analysis Mmb (I = T @ Mmb), synthesis Smb (T = I @ Smb); orthonormal analysis Phi_w (c = T @ Phi_w) and synthesis Phi (T = c @ Phi.T) under product corner marginals p (default uniform)."""
        m = 2 ** s; p = np.full(s, 0.5) if p is None else np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        sq = np.sqrt(p * (1 - p))
        sub = np.zeros((m, m)); Phi = np.ones((m, m)); w = np.ones(m)
        for sig in range(m):
            for b in range(s): w[sig] *= p[b] if (sig >> b) & 1 else 1 - p[b]
            for D in range(m):
                if D & ~sig == 0: sub[sig, D] = 1.0
                for b in range(s):
                    if (D >> b) & 1: Phi[sig, D] *= (((sig >> b) & 1) - p[b]) / sq[b]
        Mmb = np.zeros((m, m))
        for D in range(m):
            for sig in range(m):
                if sig & ~D == 0: Mmb[sig, D] = (-1) ** (bin(D & ~sig).count('1'))
        return Mmb, sub, Phi * w[:, None], Phi

    def lattice_masses(self, predf, S, X):
        """Pattern masses over the protected set S from one corner pass. Returns dict: {'m_nu': {D:...}, 'm_q': {D:...}, 'var_nu', 'T', 'sigma', 'p_hat', 'S'}; D as frozenset of positions 0..s-1 in sorted(S). m(D) = mean_rows c_D^2 under the respective corner measure (Thm 47; fibers are rows)."""
        S = sorted(S); s = len(S); T, sigma = self.corner_tensor(predf, S, X)
        p_hat = np.clip(np.asarray(X, dtype=np.float64)[:, S].mean(axis=0), 1e-6, 1 - 1e-6)
        _, _, PwU, _ = self._mats(s, None); _, _, PwQ, _ = self._mats(s, p_hat)
        cU, cQ = T @ PwU, T @ PwQ
        m_nu = {self._bits(D, s): float(np.mean(cU[:, D] ** 2)) for D in range(1, 2 ** s)}
        m_q = {self._bits(D, s): float(np.mean(cQ[:, D] ** 2)) for D in range(1, 2 ** s)}
        var_nu = float(np.mean(cU[:, 1:] ** 2) * (2 ** s - 1) + np.var(cU[:, 0]))
        return {'m_nu': m_nu, 'm_q': m_q, 'var_nu': var_nu, 'T': T, 'sigma': sigma, 'p_hat': p_hat, 'S': S}

    @staticmethod
    def _bits(D, s): return frozenset(b for b in range(s) if (D >> b) & 1)

    @staticmethod
    def _mask(Dset): return sum(1 << b for b in Dset)

    def enumerate_policies(self, s):
        """All forbidden up-sets over nonempty patterns of [s] (Prop 45): 19 at s=3, 167 at s=4 (incl. empty policy). Returns list of frozensets of frozensets."""
        pats = list(range(1, 2 ** s)); sup = {pi: [pj for pj in pats if pj & pi == pi] for pi in pats}
        out = []
        for mask in range(2 ** len(pats)):
            fam = {pats[i] for i in range(len(pats)) if (mask >> i) & 1}
            if all(set(sup[pi]) <= fam for pi in fam): out.append(frozenset(self._bits(pi, s) for pi in fam))
        return out

    def policy_tax(self, masses, forbidden, which='m_nu'):
        """Modular tax (Thm 47): sum of pattern masses over the forbidden family."""
        return float(sum(masses[which][D] for D in forbidden))

    def policy_menu(self, masses, credits=None):
        """Price every policy: rows [forbidden, phi, tax_nu, tax_q, pareto]. phi default = #forbidden patterns of size>=2; Pareto = non-dominated on (max phi, min tax_nu)."""
        s = len(masses['S']); pols = self.enumerate_policies(s)
        rows = []
        for F in pols:
            phi = sum((credits or {}).get(D, 1.0 if len(D) >= 2 else 0.0) for D in F)
            rows.append({'forbidden': F, 'phi': phi, 'tax_nu': self.policy_tax(masses, F, 'm_nu'), 'tax_q': self.policy_tax(masses, F, 'm_q')})
        for r in rows: r['pareto'] = not any((o['phi'] >= r['phi'] and o['tax_nu'] < r['tax_nu']) or (o['phi'] > r['phi'] and o['tax_nu'] <= r['tax_nu']) for o in rows)
        return sorted(rows, key=lambda r: (r['tax_nu'], -r['phi']))

    def threshold_select(self, masses, tau, k_min=2):
        """Thm 51(iii): cheapest policy meeting `every remaining pattern with |D|>=k_min has m_nu/var<=tau' = up-closure of the violating set. Returns (forbidden, tax_nu, tax_q)."""
        s, V = len(masses['S']), masses['var_nu']
        viol = [D for D, v in masses['m_nu'].items() if len(D) >= k_min and v / V > tau]
        forb = frozenset(self._bits(pj, s) for D in viol for pj in range(1, 2 ** s) if pj & self._mask(D) == self._mask(D))
        return forb, self.policy_tax(masses, forb, 'm_nu'), self.policy_tax(masses, forb, 'm_q')

    def policy_edit(self, predf, S, forbidden, mode='delete', measure_p=None):
        """Compliant edit as a closure: per row, transform the 2^s corner fiber, zero the forbidden coefficients (Mobius for `delete' = allowed-corner interpolant Prop 46; orthonormal for `projection' = Thm 10(i)/47 under measure_p corners, default uniform), re-synthesize, read off the row's own configuration. Certificate via certify()."""
        S = sorted(S); s = len(S); fmask = [self._mask(D) for D in forbidden]
        Mmb, sub, Pw, Phi = self._mats(s, measure_p)
        def G(X):
            T, sigma = self.corner_tensor(predf, S, X)
            if mode == 'delete':
                I = T @ Mmb
                for D in fmask: I[:, D] = 0.0
                Te = I @ sub.T
            else:
                c = T @ Pw
                for D in fmask: c[:, D] = 0.0
                Te = c @ Phi.T
            return Te[np.arange(len(Te)), sigma]
        return G

    def certify(self, predf, S, forbidden, X):
        """Thm 49 artifact check: max over rows and forbidden patterns of |fiber Mobius content| of predf. Zero (to float) iff the certificate holds on these contexts."""
        S = sorted(S); s = len(S); T, _ = self.corner_tensor(predf, S, X)
        I = T @ self._mats(s)[0]
        return float(max(np.max(np.abs(I[:, self._mask(D)])) for D in forbidden)) if forbidden else 0.0

    # ------------------------------------------------------------ pair pricing
    def fork_prices(self, predf, i, j, X):
        """Thm 29 on data: Delta table (4 calls/row), interventional tax E[D^2]/16, pooled associational tax E[D^2]/K with K = sum 1/q_ab from empirical pair corner freqs, ratio (identification premium), plus audit masses W_both/W_single under nu (Lem 3, uniform pair arms)."""
        X = np.asarray(X, dtype=np.float64); d = self.mixed_diff(predf, X, [i, j])
        cnt = np.array([np.mean((X[:, i] == a) & (X[:, j] == b)) for a in (0, 1) for b in (0, 1)])
        K = float(np.sum(1.0 / np.clip(cnt, 1e-9, None)))
        di, dj = self.mixed_diff(predf, X, [i]), self.mixed_diff(predf, X, [j])
        w_both = float(np.mean(d ** 2) / 16.0)
        w_single = float(np.mean(di ** 2) / 4.0 + np.mean(dj ** 2) / 4.0 - 2 * w_both)
        return {'E_d2': float(np.mean(d ** 2)), 'tax_nu': float(np.mean(d ** 2) / 16.0), 'tax_q': float(np.mean(d ** 2) / K), 'K': K, 'premium': float(16.0 / K), 'W_both_nu': w_both, 'W_single_nu': w_single, 'mean_abs_d': float(np.mean(np.abs(d)))}

    @staticmethod
    def blindness_ratio(X, S):
        """Cor 32: associational/interventional discount prod 4 p q over the protected set, from empirical marginals."""
        p = np.clip(np.asarray(X, dtype=np.float64)[:, sorted(S)].mean(axis=0), 1e-9, 1 - 1e-9)
        return float(np.prod(4 * p * (1 - p)))

    # ------------------------------------------------------------- ND battery
    def nd_battery(self, predf, i, j, X, env, alpha=0.05, exclude=()):
        """Three-layer ND x (M) battery (Sec 8.4) for two environments (env: bool mask), family-wise level alpha via Bonferroni across the layers (alpha/3 each; L1 additionally Bonferroni across context features); `exclude` drops listed context columns from L1 (e.g. the environment-defining atom, which would otherwise trivially test itself). L1 omnibus: per-context-feature two-sample z on marginals. L2: KS on the Delta pushforward. L3: Welch t on Delta^2. Verdict: `refuted' if any layer rejects, else `consistent with ND' -- never `ND holds' (Prop 44)."""
        from scipy.stats import norm
        X = np.asarray(X, dtype=np.float64); env = np.asarray(env, dtype=bool)
        A, B = X[env], X[~env]; ctx = [f for f in range(X.shape[1]) if f not in (i, j) and f not in set(exclude)]
        aL = alpha / 3.0
        zs = []
        for f in ctx:
            m1, m2 = A[:, f].mean(), B[:, f].mean(); v = m1 * (1 - m1) / max(len(A), 1) + m2 * (1 - m2) / max(len(B), 1)
            zs.append((m1 - m2) / np.sqrt(max(v, 1e-12)))
        zs = np.abs(np.array(zs)); zcrit = float(norm.ppf(1 - aL / (2 * len(ctx))))
        dA, dB = self.mixed_diff(predf, A, [i, j]), self.mixed_diff(predf, B, [i, j])
        ks = ks_2samp(dA, dB); tt = ttest_ind(dA ** 2, dB ** 2, equal_var=False)
        l1 = int(np.sum(zs > zcrit)) > 0
        out = {'L1_maxz': float(zs.max()), 'L1_n_signif': int(np.sum(zs > zcrit)), 'L2_ks': float(ks.statistic), 'L2_p': float(ks.pvalue), 'L3_t': float(tt.statistic), 'L3_p': float(tt.pvalue), 'zcrit': zcrit, 'alpha_layer': aL}
        out['verdict'] = 'refuted (ND ^ M)' if (l1 or ks.pvalue < aL or tt.pvalue < aL) else 'consistent with ND'
        return out

    # ----------------------------------------------------------- mediator tools
    def _pair_k_eval(self, predf, i, j, k, a, b, kv, X):
        Xc = np.asarray(X, dtype=np.float64).copy(); Xc[:, i], Xc[:, j], Xc[:, k] = a, b, kv
        return predf(Xc)

    def controlled_contrast(self, predf, i, j, k, kv, X):
        """Delta_ctrl(kappa, w) = sum_ab s_ab F(a,b,kappa,w) (Eq 30)."""
        out = 0.0
        for a in (0, 1):
            for b in (0, 1): out = out + ((-1) ** ((1 - a) + (1 - b))) * self._pair_k_eval(predf, i, j, k, a, b, kv, X)
        return out

    def mediator_D(self, predf, i, j, k, X):
        """D_ab(w) = F(a,b,1,w) - F(a,b,0,w) for the four arms (Eq 30). Returns dict (a,b)->array."""
        return {(a, b): self._pair_k_eval(predf, i, j, k, a, b, 1, X) - self._pair_k_eval(predf, i, j, k, a, b, 0, X) for a in (0, 1) for b in (0, 1)}

    def total_interaction(self, predf, i, j, k, response, X):
        """Lem 33: T_tot(w) = Delta_ctrl(k00,w) + sum_{ab!=00} s_ab (k_ab - k00) D_ab(w), deterministic response dict {(a,b):0/1}."""
        k00 = response[(0, 0)]; D = self.mediator_D(predf, i, j, k, X)
        T = self.controlled_contrast(predf, i, j, k, k00, X)
        for (a, b), Dab in D.items():
            if (a, b) != (0, 0): T = T + ((-1) ** ((1 - a) + (1 - b))) * (response[(a, b)] - k00) * Dab
        return T

    def potential_total(self, predf, i, j, k, response, X):
        """Unit-level total interaction via potential predictions R(a,b,w)=F(a,b,k*(a,b),w): T = R11-R10-R01+R00 (Eq 23 under the response)."""
        R = {(a, b): self._pair_k_eval(predf, i, j, k, a, b, response[(a, b)], X) for a in (0, 1) for b in (0, 1)}
        return R[(1, 1)] - R[(1, 0)] - R[(0, 1)] + R[(0, 0)]

    def total_effect_editor(self, predf, i, j, k, response):
        """Thm 34: minimal editor for a known deterministic response -- subtract T_tot(w) on the slice {x_i=x_j=1, x_k=k*(1,1)}, nothing anywhere else."""
        def G(X):
            X = np.asarray(X, dtype=np.float64)
            mask = (X[:, i] == 1) & (X[:, j] == 1) & (X[:, k] == response[(1, 1)])
            return predf(X) - self.total_interaction(predf, i, j, k, response, X) * mask
        return G

    def envelope(self, predf, i, j, k, k00, X):
        """Thm 35: attainable total-interaction interval when only k(0,0)=k00 is known. Returns (lo, hi, leverage = sum_{ab!=00}|D_ab|)."""
        D = self.mediator_D(predf, i, j, k, X); base = self.controlled_contrast(predf, i, j, k, k00, X)
        lo, hi, lev = base.copy(), base.copy(), 0.0
        for (a, b), Dab in D.items():
            if (a, b) == (0, 0): continue
            c = ((-1) ** ((1 - a) + (1 - b))) * (1 - 2 * k00) * Dab
            lo, hi, lev = lo + np.minimum(0.0, c), hi + np.maximum(0.0, c), lev + np.abs(Dab)
        return lo, hi, lev

    def misspec_bound(self, predf_edited, i, j, k, khat, ktrue, X):
        """Prop 36: per-row bound sum_ab 1{khat!=ktrue} |D^G_ab| on the residual total interaction of an editor built with a guessed response."""
        D = self.mediator_D(predf_edited, i, j, k, X); out = 0.0
        for ab, Dab in D.items():
            if khat[ab] != ktrue[ab]: out = out + np.abs(Dab)
        return out

    # ---------------------------------------------------------------- ledger
    def shapley_slice(self, cspec, f):
        """Thm 59(i)/(42) quantities for feature f from a chi spectrum: marginal c_{f}, interaction mass sum_{A ni f,|A|>=2} c^2, E[phi_f^2] = sum c^2/|A|^2."""
        marg = cspec.get(frozenset([f]), 0.0)
        imass = sum(v * v for A, v in cspec.items() if f in A and len(A) >= 2)
        e2 = sum(v * v / len(A) ** 2 for A, v in cspec.items() if f in A)
        return {'marginal': float(marg), 'interaction_mass': float(imass), 'E_phi2': float(e2)}

    def shapley_values(self, cspec, X, p=None):
        """Interventional Shapley functions phi_f(x) = sum_{A ni f} c_A phi_A(x)/|A| for all features, from the chi spectrum. Returns (n, n_features)."""
        X = np.asarray(X, dtype=np.float64); n, nf = X.shape
        p = np.full(nf, 0.5) if p is None else np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        sq = np.sqrt(p * (1 - p)); out = np.zeros((n, nf))
        for A, v in cspec.items():
            if not A or v == 0.0: continue
            Al = sorted(A); phiA = np.prod((X[:, Al] - p[Al]) / sq[Al], axis=1)
            for f in Al: out[:, f] += v * phiA / len(A)
        return out

    # --------------------------------------------------------------- metrics
    @staticmethod
    def acc_auc(margin, y, thr=0.5):
        """LPM convention: accuracy at threshold + AUC of the margin."""
        y = np.asarray(y)
        return float(np.mean((margin >= thr) == (y >= 0.5))), float(roc_auc_score(y, margin))


# ---------------------------------------------------------------- self-test
if __name__ == '__main__':
    t0 = __import__('time').time(); rng = np.random.default_rng(4000)
    n, nf = 4000, 12
    p_true = np.array([.5, .35, .6, .45, .5, .3, .55, .4, .5, .25, .5, .7])
    X = (rng.random((n, nf)) < p_true).astype(np.float64)
    y = 1.2 * X[:, 0] + 0.9 * X[:, 1] - 0.8 * X[:, 2] + 1.5 * X[:, 3] * X[:, 4] + 1.1 * X[:, 0] * X[:, 5] * X[:, 6] + 0.3 * rng.standard_normal(n)
    mc = MobiusCalculus(seed=0, depth=5, trees=250, lr=0.1).fit(pd.DataFrame(X, columns=[f"x{i}" for i in range(nf)]), y)
    F = mc.predict; rows = []
    def add(name, metric, val, tol): rows.append([len(rows) + 1, name, metric, f"{val:.3e}", f"{tol:.0e}", 'PASS' if val <= tol else 'FAIL'])
    # T1 leaf formula vs mixed difference at the all-zeros anchor (booster float32 route)
    zero = np.zeros((1, nf)); errs = []
    for _ in range(30):
        A = frozenset(rng.choice(nf, size=rng.integers(1, 4), replace=False).tolist())
        errs.append(abs(mc.leaf_I(A) - mc.mixed_diff(F, zero, sorted(A))[0]))
    add('T1 leaf formula = Delta_A F(0)', 'max err (30 random A)', max(errs), 1e-3)
    # T2 full-spectrum reconstruction of the booster margin
    spec = mc.full_spectrum(); Xs = X[:200]
    add('T2 spectrum reconstructs F', 'max |sum - margin|', float(np.max(np.abs(mc.mobius_predict(spec, Xs) - F(Xs)))), 1e-3)
    # T3 chi transform round trip (exact, float64 vs float32 margin)
    csp = mc.chi_spectrum(p=p_true)
    add('T3 chi spectrum reconstructs F', 'max err', float(np.max(np.abs(mc.chi_predict(csp, Xs, p=p_true) - F(Xs)))), 1e-3)
    # T4 Method C: delete top atom; target zeroed, others untouched, off-support bit-identical
    top = mc.topk(15); Astar, Istar = top[0]
    G4 = mc.delete_rules({Astar: Istar})
    e_target = abs(mc.mixed_diff(G4, zero, sorted(Astar))[0])
    others = [a for a, _ in top[1:6]]
    e_others = max(abs(mc.mixed_diff(G4, zero, sorted(a))[0] - mc.leaf_I(a)) for a in others)
    off = np.prod(Xs[:, sorted(Astar)], axis=1) == 0
    e_off = float(np.max(np.abs(G4(Xs)[off] - F(Xs)[off]))) if off.any() else 0.0
    add('T4a Method C zeroes the target', '|I_G(A*)|', e_target, 1e-3)
    add('T4b collateral on 5 nearest atoms', 'max shift', e_others, 1e-3)
    add('T4c off-support predictions', 'max |dF| (exact-0 expected)', e_off, 0.0)
    # T5 junction surgery: audit zero, off-corner agreement, parallel trends
    G5 = mc.junction_delete(3, 4)
    add('T5a junction audit after surgery', 'max |Delta_34 G|', float(np.max(np.abs(mc.mixed_diff(G5, Xs, [3, 4])))), 1e-3)
    offc = (Xs[:, 3] * Xs[:, 4]) == 0
    add('T5b agreement off joint corner', 'max |G-F|', float(np.max(np.abs(G5(Xs)[offc] - F(Xs)[offc]))), 1e-9)
    onc = ~offc; Xc = Xs[onc]
    X10, X01, X00 = Xc.copy(), Xc.copy(), Xc.copy(); X10[:, 4] = 0; X01[:, 3] = 0; X00[:, 3] = 0; X00[:, 4] = 0
    add('T5c parallel-trends corner value', 'max err', float(np.max(np.abs(G5(Xc) - (F(X10) + F(X01) - F(X00))))), 1e-3)
    # T6 lattice: masses, modularity, projection loss identity, policy count
    Sset = [0, 5, 6]; masses = mc.lattice_masses(F, Sset, X[:1500])
    pols = mc.enumerate_policies(3); add('T6a policy count s=3 (=19)', '|count-19|', float(abs(len(pols) - 19)), 0.0)
    Ffam = frozenset([frozenset([0, 1]), frozenset([0, 1, 2])])
    Gp = mc.policy_edit(F, Sset, Ffam, mode='projection')
    T, _ = mc.corner_tensor(F, Sset, X[:1500]); Tp, _ = mc.corner_tensor(Gp, Sset, X[:1500])
    loss_nu = float(np.mean(np.mean((T - Tp) ** 2, axis=1)))
    add('T6b projection nu-loss = modular tax', '|loss - sum m(D)|', abs(loss_nu - mc.policy_tax(masses, Ffam, 'm_nu')), 1e-4)
    # T7 delete-mode certificate + off-arm agreement (Prop 46)
    Gd = mc.policy_edit(F, Sset, Ffam, mode='delete')
    add('T7a certificate after delete edit', 'max forbidden Mobius content', mc.certify(Gd, Sset, Ffam, X[:400]), 1e-9)
    add('T7b certificate after projection edit', 'max forbidden Mobius content', mc.certify(Gp, Sset, Ffam, X[:400]), 1e-4)
    sig_allowed = (X[:400, 0].astype(int) * X[:400, 5].astype(int)) == 0
    add('T7c interpolant agrees on allowed arms', 'max |G-F|', float(np.max(np.abs(Gd(X[:400])[sig_allowed] - F(X[:400])[sig_allowed]))), 1e-9)
    # T8 marginalization: independence + coefficient identity I'(B)=I(B)+p I(B+{j})
    jm = 2; pj = float(X[:, jm].mean()); G8 = mc.marginalize([jm], [pj])
    add('T8a marginalized model independent of j', 'max |Delta_j F\'|', float(np.max(np.abs(mc.mixed_diff(G8, Xs[:100], [jm])))), 1e-3)
    errs8 = []
    for _ in range(5):
        B = frozenset(int(v) for v in rng.choice([f for f in range(nf) if f != jm], size=2, replace=False))
        errs8.append(abs(mc.mixed_diff(G8, zero, sorted(B))[0] - (mc.leaf_I(B) + pj * mc.leaf_I(B | {jm}))))
    add('T8b Prop 13 coefficient identity', 'max err (5 random B)', max(errs8), 1e-3)
    # T9 Cor 12: predicted residual dependence before edit = measured after edit
    prot = [a for a, v in spec.items() if 0 in a and abs(v) >= sorted((abs(v) for A2, v in spec.items() if 0 in A2), reverse=True)[min(19, sum(1 for A2 in spec if 0 in A2) - 1)]]
    G9 = mc.delete_rules({a: spec[a] for a in prot})
    pred_dep = mc.predicted_dependence_after(prot, 0, Xs)
    meas_dep = mc.dependence(G9, 0, Xs)
    add('T9 Cor 12 predicted = measured dependence', '|pred - meas|', abs(pred_dep - meas_dep), 1e-3)
    # T10 fork prices + blindness ratio sanity
    fp = mc.fork_prices(F, 3, 4, X[:1500])
    add('T10 fork ratio = 16/K', '|tax_q/tax_nu - 16/K|', abs(fp['tax_q'] / max(fp['tax_nu'], 1e-12) - fp['premium']), 1e-9)
    # T11 total-effect editor on an analytic mediated world (Thm 34)
    def Fmed(Z): return 2.0 * Z[:, 0] * Z[:, 1] + 1.0 * Z[:, 2] + 0.9 * Z[:, 2] * Z[:, 3]
    resp = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
    Xm = (np.random.default_rng(7).random((300, 4)) < 0.5).astype(np.float64); Xm[:, 2] = Xm[:, 0] * Xm[:, 1]
    T_before = mc.potential_total(Fmed, 0, 1, 2, resp, Xm)
    Ged = mc.total_effect_editor(Fmed, 0, 1, 2, resp)
    T_after = mc.potential_total(Ged, 0, 1, 2, resp, Xm)
    add('T11 editor zeroes total interaction', f'max |T_after| (mean T_before {float(np.mean(T_before)):+.2f})', float(np.max(np.abs(T_after))), 1e-9)
    # T12 envelope sharpness vs brute force over all 8 unknown responses (Thm 35)
    lo, hi, lev = mc.envelope(Fmed, 0, 1, 2, 0, Xm[:100])
    allT = []
    for bits in itertools.product((0, 1), repeat=3):
        r = {(0, 0): 0, (0, 1): bits[0], (1, 0): bits[1], (1, 1): bits[2]}
        allT.append(mc.total_interaction(Fmed, 0, 1, 2, r, Xm[:100]))
    allT = np.stack(allT)
    add('T12 envelope = brute-force extremes', 'max endpoint err', float(max(np.max(np.abs(allT.min(0) - lo)), np.max(np.abs(allT.max(0) - hi)))), 1e-9)
    # T13 Shapley: efficiency sum_f phi_f = F - c_emptyset (uniform-basis identity)
    csp_u = mc.chi_spectrum(p=None); phi = mc.shapley_values(csp_u, Xs[:60], p=None)
    add('T13 Shapley efficiency', 'max |sum phi - (F - c0)|', float(np.max(np.abs(phi.sum(axis=1) - (F(Xs[:60]) - csp_u[frozenset()])))), 1e-3)
    # T14 top-k branch and bound = exact top-k of the enumerated spectrum
    exact = sorted(spec.items(), key=lambda kv: -abs(kv[1]))[:15]
    add('T14 B&B topk = enumerated topk', 'max |I| mismatch', float(max(abs(abs(a[1]) - abs(b[1])) for a, b in zip(top, exact))), 1e-9)
    # T15 ND battery: silent on a null split, fires on a shifted context
    envA = rng.random(1500) < 0.5
    bat_null = mc.nd_battery(F, 3, 4, X[:1500], envA)
    Xshift = X[:1500].copy(); mshift = ~envA; Xshift[mshift, 7] = (rng.random(int(mshift.sum())) < min(p_true[7] + 0.3, 0.95)).astype(float)
    bat_alt = mc.nd_battery(F, 3, 4, Xshift, envA)
    add('T15a battery silent on null split', '1{refuted}', float(bat_null['verdict'].startswith('refuted')), 0.0)
    add('T15b battery fires on shifted context', '1{consistent}', float(not bat_alt['verdict'].startswith('refuted')), 0.0)
    # report
    print(f"mobius_calculus self-test | xgb {xgb.__version__} np {np.__version__} | n={n} nf={nf} depth=5 trees=250 | atoms={len(spec)} | {__import__('time').time()-t0:.1f}s")
    print(tabulate(rows, headers=['#', 'test', 'metric', 'value', 'tol', 'status'], tablefmt='psql'))
    print(f"top-5 |I|: " + ", ".join(f"{tuple(sorted(a))}:{v:+.3f}" for a, v in top[:5]))
    print(f"fork: E[d2]={fp['E_d2']:.4f} tax_nu={fp['tax_nu']:.4f} tax_q={fp['tax_q']:.4f} premium={fp['premium']:.3f} | blindness(S015)={MobiusCalculus.blindness_ratio(X, [0,1,5]):.3f}")
    print(f"battery null: {bat_null} ")
    print(f"battery shifted: maxz={bat_alt['L1_maxz']:.1f} n_signif={bat_alt['L1_n_signif']} verdict={bat_alt['verdict']}")
    npass = sum(1 for r in rows if r[-1] == 'PASS')
    print(f"{npass}/{len(rows)} PASS")
