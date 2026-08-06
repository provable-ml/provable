import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'verify', 'src'))
os.environ['PYTHONPATH'] = os.path.join(ROOT, 'verify', 'src')
