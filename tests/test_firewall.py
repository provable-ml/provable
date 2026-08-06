"""The firewall, machine-checked: provable_verify may import ONLY stdlib + numpy,
and must never import provable. Run: python tests/test_firewall.py"""
import ast, os, sys

ALLOWED = {'sys', 'json', 'hashlib', 'itertools', 'numpy', 'os', 'math'}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = []
for dirpath, _, files in os.walk(os.path.join(ROOT, 'verify', 'src', 'provable_verify')):
    for fn in files:
        if not fn.endswith('.py'): continue
        tree = ast.parse(open(os.path.join(dirpath, fn)).read())
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) else [])
            for m in mods:
                top = (m or '').split('.')[0]
                if top == 'provable': bad.append((fn, m, 'imports the engine'))
                elif top not in ALLOWED: bad.append((fn, m, 'outside allowed set'))
if bad:
    for fn, m, why in bad: print(f'FIREWALL BREACH: {fn} imports {m} ({why})')
    sys.exit(1)
print('firewall: PASS (provable_verify imports subset of stdlib+numpy; no engine imports)')
