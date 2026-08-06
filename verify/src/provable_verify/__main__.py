"""provable_verify -- independent certificate checker. numpy + stdlib ONLY.

Shares no code with provable: it re-implements from the xgboost JSON spec its own model
parser, its own vectorized tree-walk predictor, its own transport expansion, and its own
probes. Checks (attested in output):
  1  bundle canonical hash == certificate id
  2  model file sha256 == bundle model_out_sha256 (enforce) / model sha256 (audit)
  3  policy canonical sha256 == bundle policy sha256
  4  structural probe: own transport re-derivation -> max |forbidden atom coef| <= tol
  5  behavioral probe: own predictor, consecutive-cell second differences over the
     artifact's full pair grid at data-free random grid-corner contexts <= tol
  6  verdict re-derivation from own probes == printed verdict state
  7  limits panel present and non-empty
Usage: python -m provable_verify evidence.json model.json
"""
import sys
import json
import hashlib
import itertools
import numpy as np

TOL_STRUCT, TOL_PROBE = 1e-4, 1e-3


def canonical(o): return json.dumps(o, sort_keys=True, separators=(',', ':'), default=lambda x: float(x) if isinstance(x, (int, float)) else str(x))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''): h.update(c)
    return h.hexdigest()


class Model:
    """Own parser + predictor for an xgboost JSON artifact."""

    def __init__(self, path):
        self.raw = json.load(open(path))
        lp = self.raw['learner']['learner_model_param']
        self.base = float(str(lp['base_score']).strip('[]')); self.nf = int(lp['num_feature'])
        self.trees = self.raw['learner']['gradient_booster']['model']['trees']

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32); out = np.full(len(X), self.base, dtype=np.float64)
        for tr in self.trees:
            L = np.asarray(tr['left_children']); R = np.asarray(tr['right_children'])
            SI = np.asarray(tr['split_indices']); SC = np.asarray(tr['split_conditions'], dtype=np.float32)
            DL = np.asarray(tr['default_left'])
            node = np.zeros(len(X), dtype=int); act = L[node] != -1
            while act.any():
                nid = node[act]; f = SI[nid]; v = X[act, f]
                miss = np.isnan(v); go_left = np.where(miss, DL[nid] == 1, v < SC[nid])
                node[act] = np.where(go_left, L[nid], R[nid]); act = L[node] != -1
            out += SC[node].astype(np.float64)
        return out

    def boxes(self):
        out = []
        for tr in self.trees:
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

    def transported(self):
        atoms = {}
        for box, w in self.boxes():
            factors = []
            for f, (lo, hi) in box.items():
                t = []
                if np.isfinite(lo): t.append(((f, float(lo)), +1.0))
                else: t.append((None, +1.0))
                if np.isfinite(hi): t.append(((f, float(hi)), -1.0))
                factors.append(t)
            for combo in itertools.product(*factors) if factors else [()]:
                key = frozenset(c for c, _ in combo if c is not None); s = 1.0
                for _, sg in combo: s *= sg
                atoms[key] = atoms.get(key, 0.0) + s * w
        return atoms

    def thresholds(self):
        th = {}
        for tr in self.trees:
            L, SI, SC = tr['left_children'], tr['split_indices'], tr['split_conditions']
            for nid in range(len(L)):
                if L[nid] != -1: th.setdefault(int(SI[nid]), set()).add(float(SC[nid]))
        return {f: sorted(s) for f, s in th.items()}


def probe_values(th):
    th = sorted(th)
    if not th: return [0.0]
    if len(th) == 1: return [th[0] - 1.0, th[0] + 1.0]
    return [th[0] - 1.0] + [(a + b) / 2.0 for a, b in zip(th[:-1], th[1:])] + [th[-1] + 1.0]


def main(bundle_path, model_path):
    b = json.load(open(bundle_path)); checks = []
    body = {k: v for k, v in b.items() if k != 'certificate_id'}
    cid = 'PV-' + hashlib.sha256(canonical(body).encode()).hexdigest()[:16].upper()
    checks.append(('1 bundle hash = certificate id', cid == b.get('certificate_id')))
    want = (b.get('action') or {}).get('model_out_sha256') or b['model']['sha256']
    checks.append(('2 model file hash matches bundle', sha_file(model_path) == want))
    pol = dict(b['policy']); pol.pop('sha256', None)
    checks.append(('3 policy hash matches bundle', hashlib.sha256(canonical(pol).encode()).hexdigest() == b['policy']['sha256']))
    M = Model(model_path); names = M.raw['learner'].get('feature_names') or [f'f{i}' for i in range(M.nf)]
    idx = {n: i for i, n in enumerate(names)}
    atoms = M.transported(); th = M.thresholds()
    struct = 0.0; pairs = []
    for cl in b['policy']['forbid']:
        if cl['kind'] == 'between':
            i, j = idx[cl['features'][0]], idx[cl['features'][1]]; pairs.append((i, j))
            for A, c in atoms.items():
                sup = {f for f, _ in A}
                if i in sup and j in sup: struct = max(struct, abs(c))
        elif cl['kind'] == 'involving':
            i = idx[cl['feature']]
            for A, c in atoms.items():
                sup = {f for f, _ in A}
                if i in sup and len(sup) >= cl['min_order']: struct = max(struct, abs(c))
    is_enforce = b['verb'] == 'enforce'
    checks.append(('4 structural probe (own transport)', (struct <= TOL_STRUCT) if is_enforce else True))
    rng = np.random.default_rng(int(hashlib.sha256(b['certificate_id'].encode()).hexdigest()[:8], 16))
    C = np.zeros((200, M.nf))
    for f in range(M.nf): C[:, f] = rng.choice(probe_values(th.get(f, [])), size=200)
    probe = 0.0
    for i, j in pairs:
        vi, vj = probe_values(th.get(i, [])), probe_values(th.get(j, []))
        G = np.repeat(C, len(vi) * len(vj), axis=0)
        G[:, i] = np.tile(np.repeat(vi, len(vj)), len(C)); G[:, j] = np.tile(vj, len(vi) * len(C))
        P = M.predict(G).reshape(len(C), len(vi), len(vj))
        dd = P[:, 1:, 1:] - P[:, 1:, :-1] - P[:, :-1, 1:] + P[:, :-1, :-1]
        if dd.size: probe = max(probe, float(np.max(np.abs(dd))))
    checks.append(('5 behavioral probe (own predictor)', (probe <= TOL_PROBE) if is_enforce else True))
    supported = not (b['verdict']['state'] == 'PASS' and (struct > TOL_STRUCT or probe > TOL_PROBE))
    checks.append(('6 verdict supported by own probes', supported if is_enforce else True))
    checks.append(('7 limits panel present', bool(b.get('limits'))))
    ok = all(p for _, p in checks)
    print('provable-verify v0.1.0 (independent; numpy+stdlib; shares no code with provable)')
    for name, p in checks: print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"  measured: structural={struct:.3e}  probe={probe:.3e}")
    print('VERIFICATION: ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
