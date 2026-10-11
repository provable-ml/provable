"""adapters_xgb.py -- native xgboost I/O for provable: load external artifacts, parse to
leaf boxes, realize surgical edits as appended trees, save a still-valid native model.

Schema target: xgboost JSON (raw_format='json') as the running xgboost writes it; fields
mirrored from the loaded artifact so appended trees carry exactly the keys the host file
uses. base_score is parsed from learner_model_param and belongs to the empty atom; it is
written back in the form the host uses, a bracketed vector under xgboost >= 3.1 and a
plain float string before, as the float32 the model stores with 9 significant digits
(xgboost < 3.1 reads a bracketed value, or one with more digits, as the default 0.5
without an error).
Missing-value policy of appended trees: default_left=1, so NaN fails the rule (house
convention: NaN evaluates every predicate to False). Off-rule rows receive leaf 0.0,
which leaves float predictions bit-identical.
"""
import json
import hashlib
import numpy as np
import xgboost as xgb


class XGBAdapter:
    """One loaded xgboost artifact: parse, edit by appended trees, save, predict."""

    def __init__(self, path=None, booster=None):
        if booster is not None: self.bst = booster
        elif path is not None: self.bst = xgb.Booster(); self.bst.load_model(path)
        else: raise ValueError('path or booster required')
        self.raw = json.loads(bytearray(self.bst.save_raw(raw_format='json')))
        lp = self.raw['learner']['learner_model_param']
        self.base_score_stored = float(str(lp['base_score']).strip('[]'))
        self.objective = str(self.raw['learner'].get('objective', {}).get('name', ''))
        self.base_score = self._stored_to_margin(self.base_score_stored)
        self.num_feature = int(lp['num_feature'])
        self.feature_names = self.raw['learner'].get('feature_names') or [f'f{i}' for i in range(self.num_feature)]
        self.version = tuple(self.raw.get('version', (0, 0, 0)))

    # ------------------------------------------------------------------ link space
    _IDENTITY_PREFIXES = ('reg:squarederror', 'reg:absoluteerror', 'reg:pseudohubererror', 'reg:quantileerror', 'reg:tweedie', 'reg:gamma', 'reg:linear', 'binary:logitraw', 'rank:', 'multi:')

    def _stored_to_margin(self, b):
        """The stored base_score of an artifact is objective-space for
        link objectives under xgboost >= 2.0; the transported spectrum
        lives in margin space, so the empty-atom intercept is the stored
        value mapped through the link. Unrecognized objectives refuse
        with code R-09 rather than transport a wrong constant."""
        obj = self.objective
        if obj == 'binary:logistic':
            if tuple(self.raw.get('version', (0, 0, 0))) >= (2, 0, 0) and 0.0 < b < 1.0:
                return float(np.log(b / (1.0 - b)))
            return float(b)
        if obj.startswith('count:poisson') or obj in ('reg:gamma', 'reg:tweedie'):
            if b > 0.0: return float(np.log(b))
            return float(b)
        if obj == '' or any(obj.startswith(p) for p in self._IDENTITY_PREFIXES):
            return float(b)
        raise ValueError(f"R-09: objective '{obj}' has no registered base_score link mapping; transported constant would be wrong")

    def _margin_to_stored(self, m):
        obj = self.objective
        if obj == 'binary:logistic':
            if tuple(self.raw.get('version', (0, 0, 0))) >= (2, 0, 0):
                return float(1.0 / (1.0 + np.exp(-m)))
            return float(m)
        if obj.startswith('count:poisson') or obj in ('reg:gamma', 'reg:tweedie'):
            return float(np.exp(m))
        return float(m)

    def describe(self):
        """The model in one block: task, objective, size, and the
        link-mapped intercept."""
        trees = self.raw['learner']['gradient_booster']['model']['trees']
        leaves = sum(sum(1 for c in tr['left_children'] if c == -1) for tr in trees)
        def _depth(tr):
            L, R = tr['left_children'], tr['right_children']
            best, stack = 0, [(0, 0)]
            while stack:
                nid, d = stack.pop()
                if L[nid] == -1: best = max(best, d); continue
                stack.append((L[nid], d + 1)); stack.append((R[nid], d + 1))
            return best
        used = {int(s) for tr in trees for nid, s in enumerate(tr['split_indices']) if tr['left_children'][nid] != -1}
        return {'objective': self.objective or 'unknown', 'trees': len(trees), 'max_depth': max((_depth(tr) for tr in trees), default=0), 'leaves': leaves, 'num_feature': self.num_feature, 'features_split_on': len(used), 'features_unused': self.num_feature - len(used), 'base_score_stored': self.base_score_stored, 'base_score_margin': self.base_score, 'xgboost_artifact_version': '.'.join(str(v) for v in self.version)}

    # ------------------------------------------------------------------ parse
    def leaf_boxes(self):
        """All leaves as (box, w): box = {feat: (lo, hi)} meaning lo <= x < hi (routing:
        x < cond -> left), w = leaf value. Dead branches (lo >= hi) pruned."""
        out = []
        for tr in self.raw['learner']['gradient_booster']['model']['trees']:
            L, R, SI, SC = tr['left_children'], tr['right_children'], tr['split_indices'], tr['split_conditions']
            stack = [(0, {})]
            while stack:
                nid, box = stack.pop()
                if L[nid] == -1: out.append((box, float(SC[nid]))); continue
                f, c = int(SI[nid]), float(SC[nid])
                lo, hi = box.get(f, (-np.inf, np.inf))
                if lo < c: b = dict(box); b[f] = (lo, min(hi, c)); stack.append((L[nid], b))
                if c > lo and c < hi: b = dict(box); b[f] = (max(lo, c), hi); stack.append((R[nid], b))
                elif c <= lo: stack.append((R[nid], box))
        return out

    def thresholds(self):
        """Per-feature sorted unique split thresholds actually used by the artifact."""
        th = {}
        for tr in self.raw['learner']['gradient_booster']['model']['trees']:
            L, SI, SC = tr['left_children'], tr['split_indices'], tr['split_conditions']
            for nid in range(len(L)):
                if L[nid] != -1: th.setdefault(int(SI[nid]), set()).add(float(SC[nid]))
        return {f: np.array(sorted(s)) for f, s in th.items()}

    # ------------------------------------------------------------------ edit
    def _chain_tree(self, conds, value, tid, keys):
        """One appended tree: -value on the conjunction AND_f (x_f >= t_f), 0 elsewhere.
        conds: sorted list of (feat, thr). Node layout: cond_i at 2i, fail-leaf at 2i+1,
        success leaf at 2k. Mirrors exactly the key set of the host file's trees."""
        k = len(conds); n = 2 * k + 1
        left = [0] * n; right = [0] * n; si = [0] * n; sc = [0.0] * n
        dl = [0] * n; par = [2147483647] * n
        for i, (f, t) in enumerate(conds):
            c = 2 * i
            left[c], right[c] = c + 1, (c + 2 if i < k - 1 else 2 * k)
            si[c], sc[c], dl[c] = int(f), float(t), 1
            left[c + 1] = right[c + 1] = -1; sc[c + 1] = 0.0; par[c + 1] = c
            if i > 0: par[c] = 2 * (i - 1)
        left[2 * k] = right[2 * k] = -1; sc[2 * k] = float(np.float32(value)); par[2 * k] = 2 * (k - 1) if k else 2147483647
        base = dict(base_weights=[float(sc[i]) if left[i] == -1 else 0.0 for i in range(n)],
                    categories=[], categories_nodes=[], categories_segments=[], categories_sizes=[],
                    default_left=dl, id=tid, left_children=left, loss_changes=[0.0] * n,
                    parents=par, right_children=right, split_conditions=[float(v) for v in sc],
                    split_indices=si, split_type=[0] * n, sum_hessian=[1.0] * n,
                    tree_param={'num_deleted': '0', 'num_feature': str(self.num_feature), 'num_nodes': str(n), 'size_leaf_vector': '1'})
        return {kk: base[kk] for kk in keys}

    def append_rules(self, atoms):
        """Append one tree per atom subtracting coef * AND (x_f >= t). atoms: dict
        {frozenset((feat, thr), ...): coef}. Empty-support coef folded into base_score.
        Returns a NEW raw dict (self.raw untouched)."""
        raw = json.loads(json.dumps(self.raw))
        gm = raw['learner']['gradient_booster']['model']
        keys = list(gm['trees'][0].keys())
        tid = len(gm['trees'])
        shift = 0.0
        for A, c in atoms.items():
            if not A: shift += c; continue
            conds = sorted(((int(f), float(t)) for f, t in A))
            gm['trees'].append(self._chain_tree(conds, -float(c), tid, keys))
            gm['tree_info'].append(0)
            if 'iteration_indptr' in gm: gm['iteration_indptr'].append(gm['iteration_indptr'][-1] + 1)
            tid += 1
        gm['gbtree_model_param']['num_trees'] = str(tid)
        if shift != 0.0:
            lp = raw['learner']['learner_model_param']
            value = format(np.float32(self._margin_to_stored(float(self.base_score - shift))), '.9g')
            lp['base_score'] = f'[{value}]' if str(lp['base_score']).strip().startswith('[') else value
        return raw

    # ------------------------------------------------------------------- io
    @staticmethod
    def save_raw_json(raw, path):
        with open(path, 'w') as fh: json.dump(raw, fh)
        return path

    @staticmethod
    def sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''): h.update(chunk)
        return h.hexdigest()

    def predict_margin(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        return self.bst.predict(xgb.DMatrix(X, feature_names=self.feature_names), output_margin=True).astype(np.float64)

    @classmethod
    def reload(cls, path):
        """Fresh stock Booster from a saved file -- the no-trust-needed route."""
        return cls(path=path)
