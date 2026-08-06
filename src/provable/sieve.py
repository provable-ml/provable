"""certified_discovery.py -- the certified-discovery sieve vs. a black-box baseline, on your data.

Implements the propose-and-certify instrument of `Certified Discovery on Real Grams:
an Instrument, its Limits, and its Laws' (2026) as a single practical tool:

  pipeline    atom dictionary (quantile cuts, CART-on-residual cuts, one-hots,
              engine-harvested cuts, pairwise conjunctions; support / near-duplicate
              pruning) -> RS(nu, R) residual-sighted proposal with absorption rounds
              and sigma_r studentization (residual-sd scoring; condition C4/D3)
              -> greedy forward core to the using bar (|t| >= 1) with a
              partial-variance conditioning guard -> ONE OLS pass on S2, BH(q) over
              partial-t p-values (the gate; Definition 1) -> pure certified model =
              OLS refit on the certified support -> optional claim-free halo, ridge
              (capacity by the Gram-spectrum dof prescription, Thm 6d, capped at the
              estimated harvestable dividend mass) or marginal soft-threshold
              (Thm 6c, with a disclosed 1-dof scalar Gram calibration that guards
              against marginal double-counting on correlated residuals), selected
              from the residual spectrum; 'auto' declines the halo entirely when the
              residual dividend mass is consistent with pure noise (null-calibrated
              gate) -> two-tier predictor (core + halo; FDR guarantee untouched).
  baselines   XGBoost on raw features (the engine face, early-stopped on a carve of
              the training rows), hindsight-tuned ridge* on the atom frame, OLS-all
              on the atom frame (when N > P), and refusal (intercept).
  theory      empirical penumbra mass in the evidence band (Thm 6a), stationarity /
              concealment certification-toll bounds with labeled plug-in proxies
              (Thm 3), the spotlight functional and the exact frame-blind margin
              F(M,P) - sum rho (Lem 9/11, Thm 12), completion condition and the N3
              boundary (Thm 5/7), phase classification I / II-a / II-b / III
              (Thm 7, Cor 19), a dark-mass ledger, and a predicted-vs-observed table.
  stability   optional B split-replicates (rng 31000+b) -> per-claim identity
              frequency f_A, family frequency f_F (single-linkage |r| >= family_r),
              tiers invariant / span-stable / fragile. Tier->truth calibration is a
              cross-instrument pedigree, not recomputed here.

Claims are quoted from the single S2 pass over the forward core (selection-conditional);
the halo asserts nothing at atom level. Binary y runs as a linear probability model
(the paper's Adult convention): MSE/Brier is the primary metric, AUC is also reported.
NaN in raw features evaluates every predicate to False. Splits and every stochastic
step are seeded (house conventions: split rng 4000+seed, replicates 31000+b, engine
9000+seed); given a seed, the run is deterministic.

Conventions: statsmodels for the certification OLS (p-values), sklearn for AUC,
plotly (single figure, make_subplots) for the report figure, tabulate('psql') for
console tables.
"""
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
from xgboost import XGBRegressor, XGBClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CertifiedDiscovery:
    """One object = one run of the instrument plus its baselines and diagnostics on a dataset."""

    def __init__(self, fdr_q=0.10, rounds=4, damping=0.5, entry_t=2.0, entry_cap=300, core_cap=120, core_tmin=1.0, union_ridge=1.0, guard_v=1e-3, n_cuts=8, forced_cuts=None, cart_cuts=True, cart_depth=3, cart_leaf_frac=0.015, max_onehot=12, n_harvest=40, interactions=True, n_parents=12, n_children=30, conj_shadow=0.9, prune_support=0.005, prune_dup_r=0.98, halo='auto', halo_u=2.0, halo_dof_frac=0.02, refit_scope='train', test_frac=0.25, s2_frac=0.5, stability_reps=0, family_r=0.6, invariant_freq=0.9, xgb_params=None, seed=0, verbose=True, le_cuts=False, band_conj=False, n_path_harvest=0, path_depth=3, path_per_set=3, core_dedup_r=None):
        self.fdr_q, self.rounds, self.damping = fdr_q, rounds, damping
        self.entry_t, self.entry_cap, self.core_cap, self.core_tmin = entry_t, entry_cap, core_cap, core_tmin
        self.union_ridge, self.guard_v = union_ridge, guard_v
        self.n_cuts, self.cart_cuts, self.cart_depth, self.cart_leaf_frac = n_cuts, cart_cuts, cart_depth, cart_leaf_frac
        self.forced_cuts = forced_cuts or {}
        self.max_onehot, self.n_harvest, self.interactions = max_onehot, n_harvest, interactions
        self.n_parents, self.n_children, self.conj_shadow = n_parents, n_children, conj_shadow
        self.prune_support, self.prune_dup_r = prune_support, prune_dup_r
        self.halo, self.halo_u, self.halo_dof_frac = halo, halo_u, halo_dof_frac
        self.refit_scope = refit_scope
        self.test_frac, self.s2_frac = test_frac, s2_frac
        self.stability_reps, self.family_r, self.invariant_freq = stability_reps, family_r, invariant_freq
        self.xgb_params, self.seed, self.verbose = xgb_params, seed, verbose
        # v2 vocabulary (all default-off; behavior is byte-identical to v1 when off):
        #   le_cuts        mirror every >= cut (quantile/guideline/CART/harvest) as a <= twin.
        #                  Twins are exact complements (r = -1), so the duplicate prune switches
        #                  from |r| to signed r when this is on -- otherwise every twin would be
        #                  silently deleted. Twins are linearly redundant to the core given an
        #                  intercept; their value is as NAMEABLE, BRIDGEABLE cells (claims and
        #                  in-cell sleeves), and as parts for bands and cross-column mixes.
        #   band_conj      allow same-column (x>=a)&(x<=b) pairs (a<b) in the conjunction stage,
        #                  making interior "moderate" bands first-class atoms.
        #   n_path_harvest harvest top-gain root-to-subpath conjunctions (depth 2..path_depth)
        #                  from the fitted engine's trees, numeric features only; the Yes branch
        #                  (f < split) is emitted as <= (boundary-point and missing-value routing
        #                  differ from the engine on a measure-zero / NaN set -- NaN rows are
        #                  simply outside the cell). Names use the standard leaf grammar
        #                  "(c>=t)&(c2<=t2)" so downstream parsers need no special casing.
        #                  Selection is gain-ranked but DIVERSIFIED: at most path_per_set
        #                  paths per distinct column-set, so dominant structures cannot
        #                  monopolize the harvest and starve minority interactions.
        #   core_dedup_r   the dictionary's duplicate principle applied at CORE admission:
        #                  a candidate whose |corr| with an already-selected core member
        #                  exceeds this is skipped (one representative per near-duplicate
        #                  family). Without it, a densified vocabulary floods the core with
        #                  shadows whose S2 partial t's split and BH kills the whole family.
        self.le_cuts, self.band_conj, self.n_path_harvest, self.path_depth, self.path_per_set, self.core_dedup_r = le_cuts, band_conj, n_path_harvest, path_depth, path_per_set, core_dedup_r

    # ------------------------------------------------------------------ statics
    @staticmethod
    def _bh(p, q):
        p = np.asarray(p, float); m = len(p)
        if m == 0: return np.zeros(0, bool), 0
        o = np.argsort(p); ps = p[o]
        hits = np.nonzero(ps <= q * np.arange(1, m + 1) / m)[0]
        if len(hits) == 0: return np.zeros(m, bool), 0
        k = int(hits[-1]) + 1; sel = np.zeros(m, bool); sel[o[:k]] = True
        return sel, k

    @staticmethod
    def _omega(a):
        a = np.asarray(a, float)
        return norm.sf(a) + a * norm.pdf(a)

    def _rho(self, nu, t):
        nu = np.abs(np.asarray(nu, float))
        return nu ** 2 * (norm.cdf(t - nu) - norm.cdf(-t - nu)) + self._omega(t - nu) + self._omega(t + nu)

    @staticmethod
    def _F_frontier(M, P):
        if M <= 0: return 0.0
        lam = M / 2.0; kmax = int(lam + 12 * np.sqrt(lam + 1) + 30)
        k = np.arange(kmax + 1)
        logw = k * np.log(lam) - lam - gammaln(k + 1); w = np.exp(logw - logw.max()); w /= w.sum()
        return float(M * (1.0 - np.sum(w * (1 + 2 * k) / (P + 2 * k))))

    # ------------------------------------------------------------------ driver
    def run(self, X, y):
        t0 = time.time()
        self.X_ = pd.DataFrame(X).reset_index(drop=True)
        self.y = np.asarray(y, float).ravel()
        n = len(self.y)
        rng = np.random.default_rng(4000 + self.seed)
        perm = rng.permutation(n)
        n_test = int(round(self.test_frac * n))
        self.idx_test = perm[:n_test]; tr = perm[n_test:]
        self.n2_ = int(round(self.s2_frac * len(tr)))
        self.idx_s2 = tr[:self.n2_]; self.idx_s1 = tr[self.n2_:]; self.idx_train = tr
        self.binary_ = set(np.unique(self.y)) <= {0.0, 1.0}
        self._fit_engine()
        self._build_dictionary()
        self.union_, self.trace_ = self._sieve_run(self.idx_s1)
        self.core_ = self._core_run(self.idx_s1, self.union_)
        cert = self._certify_run(self.idx_s2, self.core_)
        self.certified_, self.khat_, self.core_res_, self.cert_sel_ = cert['certified'], cert['khat'], cert['res'], cert['sel']
        self.core_t_ = np.abs(np.asarray(self.core_res_.tvalues)[1:]) if self.core_res_ is not None else np.zeros(0)
        self._fit_cert_predictors()
        self._fit_halo()
        self._benchmarks()
        self._diagnostics()
        if self.stability_reps > 0: self._stability()
        else: self.stab_ = None
        self.elapsed_ = time.time() - t0
        self._assemble()
        if self.verbose: self._print_report()
        return self.report_

    # ------------------------------------------------------------------ engine
    def _fit_engine(self):
        Xd = pd.get_dummies(self.X_, dtype=float)
        Xd.columns = [str(c).replace('<', 'lt').replace('[', '(').replace(']', ')') for c in Xd.columns]
        self._Xd = Xd
        tr = self.idx_train
        rngv = np.random.default_rng(9000 + self.seed)
        pv = rngv.permutation(len(tr)); nval = max(1, int(round(0.15 * len(tr))))
        iv, it = tr[pv[:nval]], tr[pv[nval:]]
        params = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist', random_state=9000 + self.seed, n_jobs=-1, early_stopping_rounds=50, eval_metric='logloss' if self.binary_ else 'rmse')
        if self.xgb_params: params.update(self.xgb_params)
        m = XGBClassifier(**params) if self.binary_ else XGBRegressor(**params)
        m.fit(Xd.iloc[it], self.y[it], eval_set=[(Xd.iloc[iv], self.y[iv])], verbose=False)
        self.engine_pred_ = (m.predict_proba(Xd)[:, 1] if self.binary_ else m.predict(Xd)).astype(float)
        self.engine_ = m

    # ------------------------------------------------------------------ dictionary
    def _build_dictionary(self):
        X, tr, i1 = self.X_, self.idx_train, self.idx_s1
        y1 = self.y[i1]
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]
        base, tol = [], {}
        for c in num_cols:
            xt = pd.to_numeric(X[c].iloc[tr], errors='coerce').dropna().values.astype(float)
            if len(xt) < 20: continue
            u = np.unique(xt)
            tol[c] = (np.quantile(xt, .99) - np.quantile(xt, .01) + 1e-12) / (4 * max(self.n_cuts, 1))
            if len(u) < 2: continue
            if len(u) == 2:
                base.append(dict(kind='ge', col=c, thr=float(u[1]), name=f'{c}>={u[1]:g}', src='cut')); continue
            for thr in self.forced_cuts.get(c, []):
                if u[0] < float(thr) <= u[-1]: base.append(dict(kind='ge', col=c, thr=float(thr), name=f'{c}>={float(thr):g}', src='guideline'))
            gl = [s2['thr'] for s2 in base if s2['src'] == 'guideline' and s2['col'] == c]
            for thr in np.unique(np.quantile(xt, np.arange(1, self.n_cuts + 1) / (self.n_cuts + 1))):
                if thr <= u[0]: continue
                if gl and np.min(np.abs(np.array(gl) - thr)) < tol[c]: continue
                base.append(dict(kind='ge', col=c, thr=float(thr), name=f'{c}>={thr:g}', src='cut'))
        if self.cart_cuts:
            r1 = y1 - y1.mean()
            for c in num_cols:
                xv = pd.to_numeric(X[c].iloc[i1], errors='coerce').values.astype(float)
                mk = ~np.isnan(xv)
                if mk.sum() < 50 or len(np.unique(xv[mk])) < 3: continue
                dt = DecisionTreeRegressor(max_depth=self.cart_depth, min_samples_leaf=max(20, int(self.cart_leaf_frac * len(i1))), random_state=0)
                dt.fit(xv[mk].reshape(-1, 1), r1[mk])
                for thr in dt.tree_.threshold[dt.tree_.feature == 0]:
                    exist = [s['thr'] for s in base if s['kind'] == 'ge' and s['col'] == c]
                    if exist and np.min(np.abs(np.array(exist) - thr)) < tol[c]: continue
                    base.append(dict(kind='ge', col=c, thr=float(thr), name=f'{c}>={thr:g}', src='cart'))
        for c in cat_cols:
            vc = X[c].iloc[tr].value_counts(normalize=True); cnt = 0
            for v, f in vc.items():
                if cnt >= self.max_onehot: break
                if f < self.prune_support or f > 1 - self.prune_support: continue
                base.append(dict(kind='eq', col=c, val=v, name=f'{c}=={v}', src='onehot')); cnt += 1
        if self.n_harvest > 0:
            tdf = self.engine_.get_booster().trees_to_dataframe()
            tdf = tdf[tdf['Feature'].isin(num_cols) & tdf['Split'].notna()]
            if len(tdf):
                g = tdf.groupby(['Feature', 'Split'])['Gain'].sum().sort_values(ascending=False)
                added = 0
                for (c, thr) in g.index:
                    if added >= self.n_harvest: break
                    exist = [s['thr'] for s in base if s['kind'] == 'ge' and s['col'] == c]
                    if exist and np.min(np.abs(np.array(exist) - float(thr))) < tol.get(c, 0): continue
                    base.append(dict(kind='ge', col=c, thr=float(thr), name=f'{c}>={float(thr):g}', src='harvest')); added += 1
        if self.le_cuts:
            base += [dict(kind='le', col=s['col'], thr=s['thr'], name=f"{s['col']}<={s['thr']:g}", src=s['src']) for s in base if s['kind'] == 'ge']
        if self.n_path_harvest > 0:
            tdf = self.engine_.get_booster().trees_to_dataframe()
            nodes_all = {r.ID: r for r in tdf.itertuples()}
            paths = {}
            for tid in tdf['Tree'].unique():
                stack = [(f'{tid}-0', [], 0.0)]
                while stack:
                    nid, conds, gain = stack.pop()
                    r = nodes_all.get(nid)
                    if r is None or r.Feature == 'Leaf' or pd.isna(r.Split) or r.Feature not in num_cols: continue
                    g2 = gain + (float(r.Gain) if np.isfinite(r.Gain) else 0.0)
                    for op, child in (('le', r.Yes), ('ge', r.No)):
                        cc = conds + [(r.Feature, float(r.Split), op)]
                        simp = {}
                        for c0, t0, o0 in cc:                            # same (col,op) twice: keep the binding cut
                            k0 = (c0, o0)
                            simp[k0] = min(simp.get(k0, t0), t0) if o0 == 'le' else max(simp.get(k0, t0), t0)
                        cs = [(c0, tv, o0) for (c0, o0), tv in simp.items()]
                        if 2 <= len(cs) <= self.path_depth:
                            key = tuple(sorted((c0, round(t0, 10), o0) for c0, t0, o0 in cs))
                            if key not in paths or g2 > paths[key][0]: paths[key] = (g2, cs)
                        if len(cc) < self.path_depth: stack.append((child, cc, g2))
            per_set = {}
            n_taken = 0
            for _, cc in sorted(paths.values(), key=lambda x: -x[0]):
                if n_taken >= self.n_path_harvest: break
                ks = frozenset(c0 for c0, _, _ in cc)
                if per_set.get(ks, 0) >= self.path_per_set: continue
                per_set[ks] = per_set.get(ks, 0) + 1; n_taken += 1
                base.append(dict(kind='path', conds=cc, name='&'.join(f"({c0}{'<=' if o0 == 'le' else '>='}{t0:g})" for c0, t0, o0 in cc), src='path'))
        self._base_specs = base
        Bb = self._materialize_base(X)
        p1 = Bb[i1].mean(0); ok = (p1 > self.prune_support) & (p1 < 1 - self.prune_support)
        yc = (y1 - y1.mean()).astype(np.float32)
        sc = np.abs(Bb[i1].astype(np.float32).T @ yc) / (np.sqrt(p1 * (1 - p1)) + 1e-9)
        sc[~ok] = -1.0
        pairs = []
        if self.interactions and len(base):
            order = np.argsort(-sc)
            live = [int(j) for j in order if sc[j] > 0 and base[j]['kind'] != 'path']
            parents, children = live[:self.n_parents], live[:self.n_children]
            seen = set()
            for a in parents:
                for b in children:
                    if a == b: continue
                    key = (min(a, b), max(a, b))
                    if key in seen: continue
                    seen.add(key)
                    sa, sb = base[a], base[b]
                    if sa['col'] == sb['col'] and sa['kind'] in ('ge', 'le') and sb['kind'] in ('ge', 'le'):
                        if not (self.band_conj and {sa['kind'], sb['kind']} == {'ge', 'le'}): continue
                        ge_s, le_s = (sa, sb) if sa['kind'] == 'ge' else (sb, sa)
                        if le_s['thr'] <= ge_s['thr']: continue
                    pab = float((Bb[i1, a] & Bb[i1, b]).mean())
                    if pab >= self.conj_shadow * min(p1[a], p1[b]): continue
                    pairs.append(key)
        self._conj_pairs = pairs
        Bc = np.zeros((len(X), len(pairs)), bool)
        for k, (a, b) in enumerate(pairs): Bc[:, k] = Bb[:, a] & Bb[:, b]
        Full = np.concatenate([Bb, Bc], axis=1)
        names = [s['name'] for s in base] + [f"({base[a]['name']})&({base[b]['name']})" for a, b in pairs]
        srcs = [s['src'] for s in base] + ['conj'] * len(pairs)
        depth = [0] * len(base) + [1] * len(pairs)
        pt = Full[tr].mean(0)
        cand = np.where((pt > self.prune_support) & (pt < 1 - self.prune_support))[0]
        sub = tr[:min(8000, len(tr))]
        Xs = Full[np.ix_(sub, cand)].astype(np.float32)
        mS, sS = Xs.mean(0), Xs.std(0)
        good = sS > 1e-6; cand = cand[good]; Xs = (Xs[:, good] - mS[good]) / sS[good]
        Cm = (Xs.T @ Xs) / len(sub)
        C = Cm if self.le_cuts else np.abs(Cm)   # complements correlate at -1; |r| would delete every <= twin
        keep = []
        for pos in range(len(cand)):
            if not keep or C[pos, keep].max() <= self.prune_dup_r: keep.append(pos)
        kept = cand[np.array(keep, int)]
        self._kept = kept; self.P_ = len(kept)
        colv = Full[np.ix_(tr, kept)].astype(np.float32)
        self._mu, self._sd = colv.mean(0), colv.std(0)
        self.atom_names_ = [names[j] for j in kept]
        self.atom_src_ = np.array([srcs[j] for j in kept])
        self.atom_depth_ = np.array([depth[j] for j in kept])
        self.dict_counts_ = {s: int((self.atom_src_ == s).sum()) for s in ['guideline', 'cut', 'cart', 'onehot', 'harvest'] + (['path'] if self.n_path_harvest > 0 else []) + ['conj']}
        self.Z = ((Full[:, kept].astype(np.float32)) - self._mu) / self._sd

    def _materialize_base(self, X):
        B = np.zeros((len(X), len(self._base_specs)), bool)
        with np.errstate(invalid='ignore'):
            for j, s in enumerate(self._base_specs):
                if s['kind'] == 'ge': B[:, j] = pd.to_numeric(X[s['col']], errors='coerce').values >= s['thr']
                elif s['kind'] == 'le': B[:, j] = pd.to_numeric(X[s['col']], errors='coerce').values <= s['thr']
                elif s['kind'] == 'path':
                    m = np.ones(len(X), bool)
                    for c0, t0, o0 in s['conds']:
                        v = pd.to_numeric(X[c0], errors='coerce').values
                        m &= (v >= t0) if o0 == 'ge' else (v <= t0)
                    B[:, j] = m
                else: B[:, j] = (X[s['col']].astype(object).values == s['val'])
        return B

    def _transform(self, X):
        Bb = self._materialize_base(pd.DataFrame(X))
        Bc = np.zeros((len(Bb), len(self._conj_pairs)), bool)
        for k, (a, b) in enumerate(self._conj_pairs): Bc[:, k] = Bb[:, a] & Bb[:, b]
        return ((np.concatenate([Bb, Bc], axis=1)[:, self._kept].astype(np.float32)) - self._mu) / self._sd

    # ------------------------------------------------------------------ sieve
    def _sieve_run(self, i1):
        Z1, y1, n1 = self.Z[i1], self.y[i1], len(i1)
        F = np.full(n1, y1.mean()); inU = np.zeros(self.P_, bool); union = []; trace = []
        for r in range(1, self.rounds + 1):
            res = (y1 - F).astype(np.float32); sr = float(res.std())
            t = (Z1.T @ res).astype(float) / (np.sqrt(n1) * max(sr, 1e-12))
            cnew = np.where((np.abs(t) >= self.entry_t) & (~inU))[0]
            cnew = cnew[np.argsort(-np.abs(t[cnew]))][:self.entry_cap]
            union.extend(int(j) for j in cnew); inU[cnew] = True
            if union:
                ZU = Z1[:, union]
                A = (ZU.T @ ZU).astype(np.float64); A[np.diag_indices_from(A)] += self.union_ridge
                beta = np.linalg.solve(A, (ZU.T @ (y1 - y1.mean()).astype(np.float32)).astype(np.float64))
                fit = y1.mean() + ZU @ beta.astype(np.float32)
            else: fit = F
            F = F + self.damping * (fit - F)
            trace.append(dict(rnd=r, new=int(len(cnew)), union=int(len(union)), s1_mse=float(np.mean((y1 - F) ** 2)), max_t=float(np.max(np.abs(t))) if len(t) else 0.0))
        return union, trace

    def _core_run(self, i1, union):
        if not union: return []
        Z1 = self.Z[i1][:, union]; n1 = len(i1)
        yc = self.y[i1] - self.y[i1].mean()
        C = (Z1.T @ Z1).astype(np.float64) / n1
        b = (Z1.T @ yc.astype(np.float32)).astype(np.float64) / n1
        yy = float(np.var(yc)); dC = np.diag(C).copy()
        S, rest = [], list(range(len(union)))
        while len(S) < self.core_cap and rest:
            if S:
                A = C[np.ix_(S, S)]
                sb = np.linalg.solve(A, b[S]); rss = max(yy - float(b[S] @ sb), 1e-12)
                W = np.linalg.solve(A, C[np.ix_(S, rest)])
                rho = b[rest] - C[np.ix_(rest, S)] @ sb
                v = dC[rest] - np.einsum('ij,ij->j', C[np.ix_(S, rest)], W)
            else:
                rho, v, rss = b[np.array(rest)].copy(), dC[np.array(rest)].copy(), yy
            tt = np.sqrt(n1) * rho / np.sqrt(np.maximum(v, 1e-12) * rss)
            tt[np.asarray(v) < self.guard_v] = 0.0
            placed = False
            for oi in np.argsort(-np.abs(tt), kind='stable'):
                if np.abs(tt[oi]) < self.core_tmin: break
                if self.core_dedup_r is not None and S:
                    jc = rest[oi]
                    rmax = float(np.max(np.abs(C[jc, S] / np.sqrt(np.maximum(dC[jc] * dC[np.array(S)], 1e-12)))))
                    if rmax > self.core_dedup_r:
                        rest.pop(oi); placed = True; break     # shadow of a selected member: drop, re-rank
                S.append(rest.pop(oi)); placed = True
                break
            if not placed: break
        return [union[j] for j in S]

    def _certify_run(self, i2, core):
        out = dict(core=core, certified=[], khat=0, res=None, sel=np.zeros(len(core), bool))
        if not core: return out
        Z2 = self.Z[np.ix_(i2, np.array(core, int))].astype(np.float64)
        res = sm.OLS(self.y[i2], sm.add_constant(Z2, has_constant='add')).fit()
        pv = np.asarray(res.pvalues)[1:]
        sel, khat = self._bh(pv, self.fdr_q)
        out.update(certified=[core[i] for i in np.where(sel)[0]], khat=khat, res=res, sel=sel)
        return out

    # ------------------------------------------------------------------ certified predictors
    def _fit_cert_predictors(self):
        self.cert_params_, self.cert_pred_all_ = {}, {}
        for scope, rows in (('s2', self.idx_s2), ('train', self.idx_train)):
            if self.certified_:
                exog = sm.add_constant(self.Z[np.ix_(rows, np.array(self.certified_, int))].astype(np.float64), has_constant='add')
                pr = np.asarray(sm.OLS(self.y[rows], exog).fit().params)
            else: pr = np.array([self.y[rows].mean()])
            self.cert_params_[scope] = pr
            pall = np.full(len(self.y), pr[0])
            if self.certified_: pall = pall + self.Z[:, np.array(self.certified_, int)].astype(np.float64) @ pr[1:]
            self.cert_pred_all_[scope] = pall

    # ------------------------------------------------------------------ halo
    def _fit_halo(self):
        tr, ntr = self.idx_train, len(self.idx_train)
        base = self.cert_pred_all_[self.refit_scope]
        rres = self.y[tr] - base[tr]; hc = float(rres.mean()); rc = (rres - hc).astype(np.float32)
        srh = float(rc.std())
        certset = set(self.certified_)
        cand = np.array([j for j in range(self.P_) if j not in certset], int)
        info = dict(mode='none', reason='halo disabled', dof=0.0, lam=None, const=hc, share=np.nan, nsig=0)
        beta = np.zeros(len(cand))
        if self.halo != 'none' and len(cand) and srh > 1e-10:
            Zc = self.Z[np.ix_(tr, cand)]
            th = (Zc.T @ rc).astype(float) / (np.sqrt(ntr) * srh)
            massv = np.clip(th ** 2 - 1, 0, None) * (np.abs(th) > 1)
            mass = float(massv.sum()); nsig = int((np.abs(th) > self.halo_u).sum())
            share = float(np.sort(massv)[::-1][:10].sum() / mass) if mass > 0 else 1.0
            excess = mass - 0.483941 * len(cand)
            z_ex = excess / (1.1697 * np.sqrt(len(cand)))
            if self.halo in ('ridge', 'soft'): mode, reason = self.halo, f'user-fixed; dividend-mass excess z={z_ex:.1f}, top-10 share={share:.2f}'
            elif z_ex < 3.0: mode, reason = 'none', f'auto: residual spectrum consistent with noise (dividend-mass excess z={z_ex:.1f} < 3) -> no halo'
            else: mode, reason = ('soft' if share > 0.5 else 'ridge'), f'auto: dividend-mass excess z={z_ex:.1f}; top-10 share={share:.2f} -> ' + ('soft' if share > 0.5 else 'ridge')
            if mode == 'none' or mass == 0:
                info.update(mode=mode, reason=reason, share=share, nsig=nsig)
            elif mode == 'ridge':
                ZtZ = (Zc.T @ Zc).astype(np.float64)
                d, V = np.linalg.eigh(ZtZ); d = np.clip(d, 0, None)
                target = max(min(self.halo_dof_frac * ntr, 0.9 * len(cand), max(excess, 0.0) + 1.0), 1.0)
                lo, hi = 1e-9 * max(d.max(), 1.0), 1e9 * max(d.max(), 1.0)
                for _ in range(80):
                    mid = np.sqrt(lo * hi)
                    if float(np.sum(d / (d + mid))) > target: lo = mid
                    else: hi = mid
                lam = np.sqrt(lo * hi)
                beta = V @ ((V.T @ (Zc.T @ rc).astype(np.float64)) / (d + lam))
                info.update(mode='ridge', reason=reason + f'; dof target min(tau*N, .9P_h, mass excess)={target:.0f}', dof=float(np.sum(d / (d + lam))), lam=float(lam), share=share, nsig=nsig)
            else:
                bh_ = (Zc.T @ rc).astype(float) / ntr
                thr = self.halo_u * srh / np.sqrt(ntr)
                beta = np.sign(bh_) * np.clip(np.abs(bh_) - thr, 0, None)
                hp_tr = (Zc @ beta.astype(np.float32)).astype(float)
                den = float(hp_tr @ hp_tr)
                cstar = float(hp_tr @ rc) / den if den > 1e-12 else 0.0
                beta = beta * cstar
                info.update(mode='soft', reason=reason + f'; scalar Gram calibration c*={cstar:.2f} (1 dof, guards marginal double-count)', dof=float((beta != 0).sum() + 1), share=share, nsig=nsig)
        self.halo_ = dict(**info, cand=cand, beta=beta)
        hp = hc + (self.Z[:, cand].astype(np.float32) @ beta.astype(np.float32)).astype(float) if (len(cand) and np.any(beta)) else (np.full(len(self.y), hc) if info['mode'] != 'none' else np.zeros(len(self.y)))
        self.two_tier_pred_all_ = base + hp

    # ------------------------------------------------------------------ benchmarks
    def _benchmarks(self):
        tr, ntr = self.idx_train, len(self.idx_train)
        ytr = self.y[tr]; ybar = float(ytr.mean())
        Ztr = self.Z[tr]
        G = (Ztr.T @ Ztr).astype(np.float64)
        c = (Ztr.T @ (ytr - ybar).astype(np.float32)).astype(np.float64)
        d, V = np.linalg.eigh(G); d = np.clip(d, 0, None)
        Vc = V.T @ c
        te = self.idx_test
        best = (np.inf, None, None, None)
        for lam in np.logspace(-7, 3, 30) * max(d.mean(), 1e-9):
            bta = V @ (Vc / (d + lam))
            pte = ybar + self.Z[te].astype(np.float64) @ bta
            mse = float(np.mean((self.y[te] - pte) ** 2))
            if mse < best[0]: best = (mse, lam, bta, float(np.sum(d / (d + lam))))
        _, lamstar, bstar, dofstar = best
        ridge_all = ybar + self.Z.astype(np.float64) @ bstar
        self.ridge_star_ = dict(lam=float(lamstar), dof=dofstar, beta=bstar)
        feasible = ntr > self.P_ + 2 and d.max() > 0
        if feasible:
            inv = np.where(d > d.max() * 1e-10, 1.0 / np.maximum(d, 1e-300), 0.0)
            bo = V @ (Vc * inv)
            ols_all = ybar + self.Z.astype(np.float64) @ bo
            ols_dof = float((d > d.max() * 1e-10).sum() + 1)
        else: ols_all, ols_dof = None, None
        k = len(self.certified_)
        self.models_ = [('refusal (intercept)', np.full(len(self.y), ybar), 1.0)]
        self.models_.append(('certified core [S2-fit]' + (' *' if self.refit_scope == 's2' else ''), self.cert_pred_all_['s2'], k + 1.0))
        self.models_.append(('certified core [train-fit]' + (' *' if self.refit_scope == 'train' else ''), self.cert_pred_all_['train'], k + 1.0))
        self.models_.append(('two-tier (core + halo) *', self.two_tier_pred_all_, k + 1.0 + self.halo_['dof']))
        if feasible: self.models_.append(('OLS-all atoms', ols_all, ols_dof))
        self.models_.append(('ridge* atoms [hindsight lam]', ridge_all, dofstar + 1.0))
        self.models_.append(('xgboost [raw features]', self.engine_pred_, float('nan')))
        self.perf_ = {lbl: self._metrics(p) for lbl, p, _ in self.models_}
        self.l1_off_ = float(np.sum(np.abs(bstar[[j for j in range(self.P_) if j not in set(self.certified_)]]))) if self.P_ else 0.0

    def _metrics(self, pred_all):
        te, tr = self.idx_test, self.idx_train
        mse_te = float(np.mean((self.y[te] - pred_all[te]) ** 2))
        mse_tr = float(np.mean((self.y[tr] - pred_all[tr]) ** 2))
        r2 = 1.0 - mse_te / float(np.var(self.y[te]))
        auc = float(roc_auc_score(self.y[te], pred_all[te])) if self.binary_ and len(np.unique(self.y[te])) == 2 and np.std(pred_all[te]) > 0 else (0.5 if self.binary_ else None)
        return dict(test_mse=mse_te, test_r2=r2, test_auc=auc, train_mse=mse_tr)

    # ------------------------------------------------------------------ theory diagnostics
    def _diagnostics(self):
        q, P = self.fdr_q, self.P_
        tr, ntr, n2 = self.idx_train, len(self.idx_train), len(self.idx_s2)
        cpred = self.cert_pred_all_[self.refit_scope]
        rres = self.y[tr] - cpred[tr]; rc = (rres - rres.mean()).astype(np.float32)
        sr = float(rc.std())
        t_scan = (self.Z[tr].T @ rc).astype(float) / (np.sqrt(ntr) * max(sr, 1e-12))
        at = np.abs(t_scan)
        zbar_pool = float(norm.isf(q / (2 * P))); tbonf = float(np.sqrt(2 * np.log(P)))
        Mcore = len(self.core_); k = len(self.certified_)
        zbar_core = float(norm.isf(q / (2 * Mcore))) if Mcore else np.nan
        z_k = float(norm.isf(q * self.khat_ / (2 * Mcore))) if self.khat_ else np.nan
        edges = [(0, 1), (1, 2), (2, zbar_pool), (zbar_pool, np.inf)]
        bands = []
        for lo, hi in edges:
            mk = (at > lo) & (at <= hi)
            bands.append(dict(band=f'({lo:g}, {hi:g}]', count=int(mk.sum()), mass=float(np.clip(t_scan[mk] ** 2 - 1, 0, None).sum())))
        pen_mk = (at > 1) & (at <= zbar_pool)
        penumbra = float(np.clip(t_scan[pen_mk] ** 2 - 1, 0, None).sum())
        sigma2n = min(v['test_mse'] for v in self.perf_.values())
        conv = sigma2n / ntr
        tc = self.core_t_[self.cert_sel_] if Mcore else np.zeros(0)
        min_c = float(tc.min()) if k else np.nan
        N3_hat = float(zbar_core ** 2 * n2 / min_c ** 2) if k else np.nan
        nu_b = np.sqrt(np.clip(tc ** 2 - 1, 0, None))
        nu_d = np.sqrt(np.clip(t_scan[pen_mk] ** 2 - 1, 0, None))
        Mb, Md = float(np.sum(nu_b ** 2)), float(np.sum(nu_d ** 2)); M = Mb + Md
        s_cfg = k + int(pen_mk.sum())
        lam_t = float(2 * self._omega(tbonf))
        Rct = float(np.sum(self._rho(nu_b, tbonf)) + np.sum(self._rho(nu_d, tbonf)) + max(P - s_cfg, 0) * lam_t)
        Ffront = self._F_frontier(M, P)
        M_C = Ffront - Rct
        ridge_bench = M * P / (M + P) if M > 0 else 0.0
        phi_spot = Mb - k - M ** 2 / (M + P) if M + P > 0 else 0.0
        toll_up = self.l1_off_ * np.sqrt(sigma2n) * (zbar_pool + tbonf) / np.sqrt(ntr)
        vault = (at > 2) & (at <= zbar_pool)
        kd = int(vault.sum())
        toll_lo = float(kd * (at[vault].max() / np.sqrt(ntr)) * np.sqrt(2 * np.log(P / max(kd, 1)) / ntr) * sigma2n) if kd else 0.0
        pf = self.perf_
        lab = {'cert': [l for l in pf if l.startswith('certified core') and l.endswith('*')][0], 'tt': 'two-tier (core + halo) *', 'xgb': 'xgboost [raw features]', 'ref': 'refusal (intercept)', 'rid': 'ridge* atoms [hindsight lam]'}
        m = {kk: pf[vv]['test_mse'] for kk, vv in lab.items()}
        m['ols'] = pf['OLS-all atoms']['test_mse'] if 'OLS-all atoms' in pf else np.inf
        inspan_best = min(m['tt'], m['rid'], m['ols'])
        gaps = dict(cert_vs_engine=m['cert'] - m['xgb'], twotier_vs_engine=m['tt'] - m['xgb'], refund_obs=m['cert'] - m['tt'], inspan_gap_obs=m['cert'] - inspan_best, fb_margin_obs=m['rid'] - m['cert'])
        if k == 0: phase = f'I  (instrument refuses: nothing certifiable at q={q}; refusal vs engine test MSE {m["ref"]:.4g} vs {m["xgb"]:.4g})'
        elif m['ref'] <= m['xgb'] * 1.005: phase = 'I  (mass < dof territory: refusal matches or beats the engine)'
        elif m['cert'] <= m['xgb']: phase = 'III-like  (certified core ahead of the engine on test -- win observed)'
        else:
            sub = 'II-a (spotlight: estimated frame-blind margin M_C > 0)' if M_C > 0 else 'II-b (estimated frame-blind margin M_C <= 0)'
            phase = f'{sub}; engine lead {gaps["cert_vs_engine"]:.3g} vs in-span toll cap {toll_up:.3g} (excess, if any, = off-span wall D_perp + proxy slack)'
        pv_rows = [('penumbra refund (in-span, per-obs)', f'{penumbra * conv:.3g}', f'{gaps["refund_obs"]:.3g}', 'refund observed' if gaps['refund_obs'] > 0 else 'no refund observed'),
                   ('in-span certification gap vs toll upper (proxy)', f'<= {toll_up:.3g}', f'{gaps["inspan_gap_obs"]:.3g}', 'capped' if gaps['inspan_gap_obs'] <= toll_up else 'exceeds proxy cap'),
                   ('residual vs engine after refund (off-span wall D_perp + estimation)', 'no bound claimed', f'{gaps["twotier_vs_engine"]:.3g}', 'dictionary-level, not a certification cost'),
                   ('frame-blind margin sign (spotlight, plug-in)', 'cert ahead of ridge*' if M_C > 0 else 'cert behind ridge*', f'ridge* - cert = {gaps["fb_margin_obs"]:.3g}', 'consistent' if (M_C > 0) == (gaps['fb_margin_obs'] > 0) else 'inconsistent')]
        var_te = float(np.var(self.y[self.idx_test]))
        ledger = dict(var_y_test=var_te, legible_certified=var_te - m['cert'], dark_visible_to_engine=m['cert'] - m['xgb'], noise_plus_dark_to_both=m['xgb'])
        self.diag_ = dict(P=P, zbar_pool=zbar_pool, tbonf=tbonf, zbar_core=zbar_core, z_k=z_k, bands=bands, penumbra_score=penumbra, penumbra_perobs=penumbra * conv, sigma2n=sigma2n, sr_train=sr, max_scan=float(at.max()), stationary_entry=bool(at.max() < self.entry_t), stationary_bar=bool(at.max() < zbar_pool), min_cert_t=min_c, med_cert_t=float(np.median(tc)) if k else np.nan, max_cert_t=float(tc.max()) if k else np.nan, N3_hat=N3_hat, Mb=Mb, Md=Md, M=M, s_cfg=s_cfg, F=Ffront, Rct=Rct, M_C=M_C, M_C_perobs=M_C * conv, ridge_bench=ridge_bench, phi_spot=phi_spot, dividend=tbonf ** 2 - 1, harvest_term=M ** 2 / (M + P) if M + P > 0 else 0.0, toll_up=toll_up, toll_lo=toll_lo, kd=kd, conv=conv, gaps=gaps, phase=phase, pv_rows=pv_rows, ledger=ledger)

    # ------------------------------------------------------------------ stability
    def _stability(self):
        B, n2 = self.stability_reps, self.n2_
        sets = []
        for b in range(B):
            rr = np.random.default_rng(31000 + b)
            pm = rr.permutation(self.idx_train)
            u, _ = self._sieve_run(pm[n2:])
            co = self._core_run(pm[n2:], u)
            ce = self._certify_run(pm[:n2], co)
            sets.append(set(ce['certified']))
        Fset = sorted(set().union(*sets, set(self.certified_))) if sets else sorted(set(self.certified_))
        sub = self.idx_train[:min(8000, len(self.idx_train))]
        tiers = {}
        if Fset:
            Zs = self.Z[np.ix_(sub, np.array(Fset, int))]
            Cf = np.abs((Zs.T @ Zs).astype(np.float64) / len(sub))
            parent = list(range(len(Fset)))
            def find(a):
                while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
                return a
            for i in range(len(Fset)):
                for jj in range(i + 1, len(Fset)):
                    if Cf[i, jj] >= self.family_r: parent[find(i)] = find(jj)
            pos = {j: i for i, j in enumerate(Fset)}
            for j in self.certified_:
                comp = [Fset[i] for i in range(len(Fset)) if find(i) == find(pos[j])]
                fA = float(np.mean([j in s for s in sets]))
                fF = float(np.mean([any(mm in s for mm in comp) for s in sets]))
                tier = 'invariant' if fA >= self.invariant_freq else ('span-stable' if fF >= self.invariant_freq else 'fragile')
                tiers[j] = dict(f_A=fA, f_F=fF, tier=tier, fam_size=len(comp))
        self.stab_ = dict(B=B, tiers=tiers, sizes=[len(s) for s in sets])

    # ------------------------------------------------------------------ assembly & report
    def _assemble(self):
        res, sel = self.core_res_, self.cert_sel_
        rows = []
        if res is not None and sel.any():
            par = np.asarray(res.params)[1:]; se = np.asarray(res.bse)[1:]
            tv = np.asarray(res.tvalues)[1:]; pv = np.asarray(res.pvalues)[1:]
            ci = np.asarray(res.conf_int())[1:]
            for pos in np.where(sel)[0]:
                j = self.core_[pos]; sd = float(self._sd[j])
                row = dict(atom=self.atom_names_[j], src=self.atom_src_[j], depth=int(self.atom_depth_[j]), effect=par[pos] / sd, ci_lo=ci[pos, 0] / sd, ci_hi=ci[pos, 1] / sd, t=float(tv[pos]), p=float(pv[pos]), atom_id=int(j))
                if self.stab_ and j in self.stab_['tiers']: row.update({k2: self.stab_['tiers'][j][k2] for k2 in ('f_A', 'f_F', 'tier')})
                rows.append(row)
        self.claims_ = pd.DataFrame(rows).sort_values('p').reset_index(drop=True) if rows else pd.DataFrame()
        self.report_ = dict(setup=dict(n=len(self.y), n_s1=len(self.idx_s1), n_s2=len(self.idx_s2), n_test=len(self.idx_test), P=self.P_, dict_counts=self.dict_counts_, task='binary (LPM)' if self.binary_ else 'regression', seed=self.seed, refit_scope=self.refit_scope), trace=self.trace_, certification=dict(pool=self.P_, union=len(self.union_), core=len(self.core_), k=len(self.certified_), khat=self.khat_, q=self.fdr_q), performance=self.perf_, halo={k2: self.halo_[k2] for k2 in ('mode', 'reason', 'dof', 'lam', 'share', 'nsig')}, diagnostics=self.diag_, claims=self.claims_.to_dict('records'), stability=(dict(B=self.stab_['B'], sizes=self.stab_['sizes']) if self.stab_ else None), elapsed_s=self.elapsed_)

    def _print_report(self):
        d, fmt = self.diag_, lambda v, nd=5: ('' if v is None else (f'{v:.{nd}f}' if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else str(v)))
        print('\n================ certified-discovery report ================')
        s = self.report_['setup']
        print(tabulate([[s['n'], s['n_s1'], s['n_s2'], s['n_test'], s['P'], str(s['dict_counts']), s['task'], s['seed'], s['refit_scope']]], headers=['N', 'S1', 'S2', 'test', 'P atoms', 'pool composition', 'task', 'seed', 'refit'], tablefmt='psql'))
        print('\n-- sieve (RS proposal on S1: nu=%.2f, R=%d, entry |t|>=%.1f) --' % (self.damping, self.rounds, self.entry_t))
        print(tabulate([[t['rnd'], t['new'], t['union'], fmt(t['s1_mse']), fmt(t['max_t'], 2)] for t in self.trace_], headers=['round', 'new', 'union', 'S1 mse', 'max |t|'], tablefmt='psql'))
        c = self.report_['certification']
        print('\n-- certification (one OLS pass on S2, BH q=%.2f over partial-t) --' % self.fdr_q)
        print(tabulate([[c['pool'], c['union'], c['core'], c['k'], fmt(self.entry_t, 1), fmt(self.core_tmin, 1), fmt(d['z_k'], 2), fmt(d['zbar_core'], 2), fmt(d['zbar_pool'], 2), fmt(d['tbonf'], 2), fmt(d['min_cert_t'], 2), fmt(d['med_cert_t'], 2), fmt(d['max_cert_t'], 2)]], headers=['P', 'union', 'core', 'k cert', 'entry', 'use bar', 'z(khat)', 'zbar core', 'zbar pool', 'sqrt2lnP', 'min|t|', 'med|t|', 'max|t|'], tablefmt='psql'))
        if len(self.claims_):
            print('\n-- certified claims (S2 one-pass partial effects; raw predicate scale; top 15 by p) --')
            cl = self.claims_.head(15).copy()
            cols = ['atom', 'src', 'effect', 'ci_lo', 'ci_hi', 't', 'p'] + ([cc for cc in ('tier', 'f_A', 'f_F') if cc in cl.columns])
            rows = [[(r['atom'][:52] + '..') if len(r['atom']) > 54 else r['atom'], r['src'], fmt(r['effect'], 4), fmt(r['ci_lo'], 4), fmt(r['ci_hi'], 4), fmt(r['t'], 2), f"{r['p']:.2e}"] + ([r.get('tier', ''), fmt(r.get('f_A'), 2), fmt(r.get('f_F'), 2)] if 'tier' in cl.columns else []) for _, r in cl.iterrows()]
            print(tabulate(rows, headers=cols, tablefmt='psql'))
        h = self.report_['halo']
        print('\n-- halo (claim-free) -- mode=%s | dof=%.1f | lam=%s | %s' % (h['mode'], h['dof'], fmt(h['lam'], 4), h['reason']))
        print('\n-- performance (* = canonical tier used by diagnostics; hindsight lam disclosed) --')
        hdr = ['model', 'test MSE', 'test R2', 'train MSE', 'dof'] + (['test AUC'] if self.binary_ else [])
        rows = [[lbl, fmt(self.perf_[lbl]['test_mse']), fmt(self.perf_[lbl]['test_r2'], 4), fmt(self.perf_[lbl]['train_mse']), fmt(dof, 1)] + ([fmt(self.perf_[lbl]['test_auc'], 4)] if self.binary_ else []) for lbl, _, dof in self.models_]
        print(tabulate(rows, headers=hdr, tablefmt='psql'))
        print('\n-- residual scan on train after certified core (marginal |t|, sigma_r-studentized) --')
        print(tabulate([[b['band'], b['count'], fmt(b['mass'], 1)] for b in d['bands']], headers=['|t| band', 'atoms', 'mass sum(t^2-1)+'], tablefmt='psql'))
        print('\n-- theory panel (score units unless stated; per-obs = x sigma2_noise/N_train; proxies labeled) --')
        tp = [['penumbra mass (1 < |t| <= zbar_pool)', fmt(d['penumbra_score'], 1), 'per-obs ' + f"{d['penumbra_perobs']:.3g}", 'Thm 6a: refundable evidence-band cost'],
              ['certification toll upper (stationarity)', f"{d['toll_up']:.3g}", 'per-obs, proxy', 'l1 off-support from ridge*; c=1'],
              ['certification toll lower (concealment)', f"{d['toll_lo']:.3g}", 'per-obs, proxy', f"kappa_d={d['kd']} in (2, zbar]; c0=1"],
              ['spotlight Phi_spot = M_b - k - M^2/(M+P)', fmt(d['phi_spot'], 1), 'score', f"M_b={d['Mb']:.1f}, M_d={d['Md']:.1f}, k={len(self.certified_)}"],
              ['frame-blind frontier F(M,P)', fmt(d['F'], 1), 'score', f"M={d['M']:.1f}, P={d['P']}"],
              ['certification risk R(C,t) at t=sqrt(2lnP)', fmt(d['Rct'], 1), 'score', 'plug-in config from S2 |t| and scan'],
              ['exact frame-blind margin M_C = F - R', fmt(d['M_C'], 1), 'per-obs ' + f"{d['M_C_perobs']:.3g}", 'Thm 12; >0 -> cert beats frame-blind class'],
              ['per-atom dividend 2lnP - 1', fmt(d['dividend'], 1), 'score', 'what each certified atom pays'],
              ['ridge benchmark MP/(M+P)', fmt(d['ridge_bench'], 1), 'score', 'Cor 14 fixed-c class'],
              ['completion: max residual |t| on pool', fmt(d['max_scan'], 2), f"< entry: {d['stationary_entry']} | < zbar: {d['stationary_bar']}", 'Thm 5 stationarity check'],
              ['N3_hat (weakest certified at claims bar)', f"{d['N3_hat']:.3g}", f"N_train = {len(self.idx_train)}", 'phase III boundary, plug-in'],
              ['noise proxy sigma2 (best test MSE)', f"{d['sigma2n']:.4g}", 'y-units^2', 'upper bound on sigma^2'],
              ['sigma_r on train (post-core residual sd)', fmt(d['sr_train'], 4), 'y-units', 'studentization scale']]
        print(tabulate(tp, headers=['quantity', 'value', 'units/aux', 'notes'], tablefmt='psql'))
        L = d['ledger']
        print('\n-- dark-mass ledger on test (Var(y) = legible + dark-visible-to-engine + floor) --')
        print(tabulate([[fmt(L['var_y_test']), fmt(L['legible_certified']), fmt(L['dark_visible_to_engine']), fmt(L['noise_plus_dark_to_both'])]], headers=['Var(y_test)', 'legible (certified)', 'cert - engine (dark to cert, visible to engine)', 'engine MSE (noise + dark to both)'], tablefmt='psql'))
        print('\n-- phase & predicted-vs-observed --')
        print(tabulate(d['pv_rows'], headers=['quantity', 'predicted / bound', 'observed (test)', 'status'], tablefmt='psql'))
        print(f"\nPHASE: {d['phase']}")
        if self.stab_: print(f"stability: B={self.stab_['B']} replicates, certified-set sizes {self.stab_['sizes']}")
        print(f"elapsed: {self.elapsed_:.1f}s")

    # ------------------------------------------------------------------ prediction on new data
    def predict(self, X, model='two_tier'):
        X = pd.DataFrame(X)
        if model == 'xgb':
            Xd = pd.get_dummies(X, dtype=float); Xd.columns = [str(c).replace('<','lt').replace('[','(').replace(']',')') for c in Xd.columns]; Xd = Xd.reindex(columns=self._Xd.columns, fill_value=0.0)
            return (self.engine_.predict_proba(Xd)[:, 1] if self.binary_ else self.engine_.predict(Xd)).astype(float)
        Zn = self._transform(X)
        pr = self.cert_params_[self.refit_scope]
        p = np.full(len(X), pr[0])
        if self.certified_: p = p + Zn[:, np.array(self.certified_, int)].astype(np.float64) @ pr[1:]
        if model == 'certified': return p
        h = self.halo_
        if h['mode'] != 'none' and len(h['cand']) and np.any(h['beta']): p = p + h['const'] + (Zn[:, h['cand']].astype(np.float32) @ h['beta'].astype(np.float32)).astype(float)
        elif h['mode'] != 'none': p = p + h['const']
        return p

    # ------------------------------------------------------------------ figure
    def plot(self):
        d = self.diag_
        fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'secondary_y': True}, {}]], subplot_titles=('core |t| on S2 vs bars', 'test MSE by model', 'sieve trace on S1', 'pool residual |t| spectrum (post-core)'))
        if len(self.core_t_):
            o = np.argsort(-self.core_t_); ts = self.core_t_[o]; cs = self.cert_sel_[o]
            xs = np.arange(1, len(ts) + 1)
            fig.add_trace(go.Scatter(x=xs[cs], y=ts[cs], mode='markers', name='certified', marker=dict(color='#1f77b4', size=7)), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs[~cs], y=ts[~cs], mode='markers', name='core, not certified', marker=dict(color='#bbbbbb', size=6)), row=1, col=1)
            for val, nm, colr in [(self.core_tmin, 'using bar (1)', '#2ca02c'), (self.entry_t, 'entry (2)', '#ff7f0e'), (d['z_k'], 'z(khat)', '#d62728'), (d['zbar_core'], 'zbar core', '#9467bd')]:
                if np.isfinite(val): fig.add_trace(go.Scatter(x=[1, len(ts)], y=[val, val], mode='lines', name=nm, line=dict(color=colr, dash='dot', width=1.5)), row=1, col=1)
        labels = [lbl for lbl, _, _ in self.models_]
        fig.add_trace(go.Bar(x=labels, y=[self.perf_[l]['test_mse'] for l in labels], name='test MSE', marker=dict(color='#1f77b4'), showlegend=False), row=1, col=2)
        rr = [t['rnd'] for t in self.trace_]
        fig.add_trace(go.Scatter(x=rr, y=[t['s1_mse'] for t in self.trace_], mode='lines+markers', name='S1 MSE', line=dict(color='#1f77b4')), row=2, col=1)
        fig.add_trace(go.Scatter(x=rr, y=[t['union'] for t in self.trace_], mode='lines+markers', name='union size', line=dict(color='#ff7f0e')), row=2, col=1, secondary_y=True)
        tr, ntr = self.idx_train, len(self.idx_train)
        rres = self.y[tr] - self.cert_pred_all_[self.refit_scope][tr]; rc = (rres - rres.mean()).astype(np.float32)
        at = np.sort(np.abs((self.Z[tr].T @ rc).astype(float) / (np.sqrt(ntr) * max(rc.std(), 1e-12))))[::-1]
        fig.add_trace(go.Scatter(x=np.arange(1, len(at) + 1), y=at, mode='lines', name='residual |t| (sorted)', line=dict(color='#1f77b4'), showlegend=False), row=2, col=2)
        for val, nm, colr in [(1.0, 'using bar', '#2ca02c'), (2.0, 'entry', '#ff7f0e'), (d['zbar_pool'], 'zbar pool', '#9467bd')]:
            fig.add_trace(go.Scatter(x=[1, len(at)], y=[val, val], mode='lines', name=nm, line=dict(color=colr, dash='dot', width=1.5), showlegend=False), row=2, col=2)
        fig.update_yaxes(title_text='|t|', row=1, col=1); fig.update_yaxes(title_text='test MSE', row=1, col=2)
        fig.update_yaxes(title_text='S1 MSE', row=2, col=1); fig.update_yaxes(title_text='union', row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text='|t|', type='log', row=2, col=2); fig.update_xaxes(title_text='atom rank', row=2, col=2)
        fig.update_layout(height=820, width=1150, title='certified discovery vs black box -- run report', legend=dict(orientation='h', y=-0.08))
        fig.show()


if __name__ == '__main__':
    CSV_PATH = 'adult-all.csv'
    CSV_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv'
    COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    FEATURES = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    SEED = 0
    STABILITY_REPS = 8
    SHOW_PLOT = True
    try: df = pd.read_csv(CSV_PATH, header=None, names=COLS, skipinitialspace=True)
    except FileNotFoundError: df = pd.read_csv(CSV_URL, header=None, names=COLS, skipinitialspace=True)
    df = df.replace('?', np.nan).dropna().reset_index(drop=True)
    y = (df['income'] == '>50K').astype(float).values
    X = df[FEATURES]
    cd = CertifiedDiscovery(stability_reps=STABILITY_REPS, seed=SEED)
    report = cd.run(X, y)
    if SHOW_PLOT: cd.plot()
