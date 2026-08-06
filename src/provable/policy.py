"""policy.py -- declarative policies over named features. Clauses:
  between(a, b)              forbid any rule whose support contains BOTH a and b
  involving(f, min_order)    forbid any rule containing f with support size >= min_order
Serialization is canonical JSON (sorted keys); the policy hash is part of the certificate.
"""
import json
import hashlib


def between(a, b): return {'kind': 'between', 'features': sorted([str(a), str(b)])}


def involving(f, min_order=2): return {'kind': 'involving', 'feature': str(f), 'min_order': int(min_order)}


class Policy:
    def __init__(self, forbid, name='policy', tolerance=0.0):
        self.name, self.forbid, self.tolerance = str(name), list(forbid), float(tolerance)

    def to_dict(self): return {'name': self.name, 'forbid': self.forbid, 'tolerance': self.tolerance}

    def canonical(self): return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))

    def sha256(self): return hashlib.sha256(self.canonical().encode()).hexdigest()

    @classmethod
    def from_dict(cls, d): return cls(d['forbid'], d.get('name', 'policy'), d.get('tolerance', 0.0))

    def resolve(self, feature_names):
        """Map clause feature names to column indices; unknown names -> refusal reason."""
        idx = {n: i for i, n in enumerate(feature_names)}
        resolved, refusals = [], []
        for cl in self.forbid:
            try:
                if cl['kind'] == 'between': resolved.append(dict(cl, i=idx[cl['features'][0]], j=idx[cl['features'][1]]))
                elif cl['kind'] == 'involving': resolved.append(dict(cl, i=idx[cl['feature']]))
                else: refusals.append({'code': 'R-08', 'reason': f"unknown clause kind {cl['kind']!r}"})
            except KeyError as e: refusals.append({'code': 'R-05', 'reason': f'feature {e} not in model'})
        return resolved, refusals
