# The demo of docs/scenarios/c18-end-to-end.md: the page's code blocks, in order.
# From the repository root: uv run python docs/scenarios/c18_holdout_demo.py

import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import provable as pv
from provable import certificates as ct
from provable.certificates import transcript as tr

os.chdir(tempfile.mkdtemp(prefix="holdout_demo_"))
rng = np.random.default_rng(20)
n = 4500
panel = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
signal = 0.9 * panel["x0"] - 0.6 * panel["x1"] + 0.8 * panel["x2"] * (panel["x3"] > 0)
panel["y"] = (signal + rng.normal(size=n) > 0).astype(float)
panel["start_utc"] = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
feats = [f"x{i}" for i in range(6)]
train, block_a, block_b, block_c = panel.iloc[:2000], panel.iloc[2000:3000], panel.iloc[3000:4000], panel.iloc[4000:]
rows = lambda frame: ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=frame.index))
S, A, B, C = rows(train), rows(block_a), rows(block_b), rows(block_c)


def fit(model, cols):
    model.fit(train[cols], train["y"])
    return lambda X: model.predict_proba(X[cols])[:, 1]


candidates = [("logit x0", fit(LogisticRegression(), ["x0"])),
              ("logit x0 x1", fit(LogisticRegression(), ["x0", "x1"])),
              ("logit all", fit(LogisticRegression(), feats)),
              ("gbm depth 2", fit(GradientBoostingClassifier(max_depth=2, random_state=0), feats)),
              ("gbm depth 3", fit(GradientBoostingClassifier(max_depth=3, random_state=0), feats)),
              ("gbm depth 5", fit(GradientBoostingClassifier(max_depth=5, random_state=0), feats))]
keep = dict(candidates)


book = ct.Ledger("demo_book", budget=0.20)
ho_a = pv.seal(block_a[feats], block_a["y"], book, lineage=A, loss="brier",
               baseline=float(train["y"].mean()), baseline_lineage=S, label="block A: model choice")
print(ho_a)


for name, model in candidates:
    print(f"{name:12s}", ho_a.ask(model, lineage=S, label=name))
try:
    ho_a.ask(keep["logit all"], lineage=S.union(rows(block_a.iloc[:25])), label="logit all, refit")
except ValueError as e:
    print(e)
print(ho_a.transcript())


L_a = ho_a.state["bits"]
print(f"{L_a:.2f} bits: z_L {tr.z_bar(0.05, L_a):.3f}, s(L) {tr.clean_share(0.05, L_a):.3f}; "
      f"an exam slice of 0.2 narrower than the transcript certificate: {tr.exam_wins(0.05, L_a, 0.2)}")
incumbent = next(r["label"] for r in ho_a.transcript().rows if r["seq"] == ho_a.state["incumbent"])
cert_a = ho_a.open(keep[incumbent], lineage=S, label=incumbent)
print(cert_a.claim)
print(f"{cert_a.id} carries the tuning transcript as a peek: {ct.peeks_of(cert_a)[0]['bits']:.2f} bits on the 800 tuning rows")


release_a = ho_a.retire()
print(release_a)
print(ho_a.transcript())


ho_b = pv.seal(block_b[feats], block_b["y"], book, lineage=B, loss="zero_one", mode="numeric",
               exam_share=0.25, label="block B: error rate")
print(ho_b)
keep["chosen"] = lambda X: (keep[incumbent](X) > 0.5).astype(float)
keep["runner-up"] = lambda X: (keep["logit all"](X) > 0.5).astype(float)
for name, offset in (("chosen", 0.0), ("chosen", 0.10), ("runner-up", 0.0)):
    guess = float((keep[name](release_a.X) != release_a.y).mean()) + offset
    width = 2 * ho_b.min_width()
    print(f"{name:9s} guess {guess:.4f} width {width:.4f} ->", ho_b.check(keep[name], guess, width, lineage=S, label=name))
print(ho_b.transcript())


cert_b = ho_b.open(keep["chosen"], lineage=S, label="chosen")
print(cert_b.claim)
release_b = ho_b.retire()
print(release_b)
for ho in (ho_a, ho_b):
    models = {r["seq"]: keep[r["label"]] for r in ho.transcript().rows if r["label"] in keep}
    print(ho.audit(models=models))


for name, rows_ in (("block C", C), ("A's tuning rows", ho_a.tuning_lineage()), ("A's exam rows", ho_a.exam_lineage())):
    try:
        v = pv.reuse([cert_a], rows_, ledger=book, level=0.05)
        print(f"{name:15s}: {v.verdict.label} | bits {v.bits:.2f} | inflation {v.inflation:.4g} | pass level {v.level:.4g}")
    except ct.FreshnessError as e:
        print(f"{name:15s}: REFUSED -- {str(e)[:80]} ...")
v = pv.reuse([cert_b], ho_b.tuning_lineage(), ledger=book, level=0.05)
print(f"B's tuning rows: {v.verdict.label} | bits {v.bits:.2f} | inflation {v.inflation:.4g} | pass level {v.level:.4g}")
print(f"A's 800 tuning rows after {L_a:.2f} bits certify like {800 * tr.clean_share(0.05, L_a):.0f} clean rows; "
      f"matching 800 clean rows takes {ct.rows_for_bits(800, L_a):,.0f}")


print(book.statement())
record = [["book", book.path, "every answer, certificate and release, in log order"],
          ["holdout A", ho_a.id, f"comparative, exam route; {L_a:.2f} bits; retired"],
          ["certificate A", cert_a.id, f"mean Brier loss at most {cert_a.artifact['bound']:.4f}, FWER 0.05; peek {L_a:.2f} bits"],
          ["holdout B", ho_b.id, f"numeric; transcript {ho_b.state['bits']:.2f} bits; retired"],
          ["family B", ho_b.state["family"], "every interval check released covers its value, FWER 0.05"],
          ["certificate B", cert_b.id, f"error rate at most {cert_b.artifact['bound']:.4f}, FWER 0.05; peek {ho_b.state['bits']:.2f} bits"],
          ["models", ", ".join(keep), "under the labels given at each call; audit takes them by log number"]]
print(tabulate(record, headers=["kept", "value", "what it is"], tablefmt="psql"))
