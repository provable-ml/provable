# The demo of docs/scenarios/c20-few-rows.md: the page's code blocks, in order.
# From the repository root: uv run python docs/scenarios/c20_few_rows_demo.py

import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
import provable as pv
from provable import certificates as ct

os.chdir(tempfile.mkdtemp(prefix="few_rows_"))
P, K = 100, 30
beta = np.r_[np.full(5, 0.25), np.zeros(P - 5)]
names = [f"x{j}" for j in range(P)]


def world(rep, m):
    rng = np.random.default_rng(3000 + rep)
    Xs = rng.normal(size=(m, P))
    ys = Xs @ beta + rng.normal(size=m)
    Xh = rng.normal(size=(1000, P))
    yh = Xh @ beta + rng.normal(size=1000)
    panel = pd.DataFrame(np.vstack([Xs, Xh]), columns=names).assign(y=np.r_[ys, yh])
    panel["start_utc"] = pd.date_range("2026-01-01", periods=len(panel), freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    train, hold = panel.iloc[:m], panel.iloc[m:]
    S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=f.index)) for f in (train, hold))
    return rng, train, hold, S, H


def candidates(X, y):
    order = np.argsort(-np.abs((X - X.mean(0)).T @ (y - y.mean())))
    fits = []
    for k in range(1, K + 1):
        keep = order[:k]
        fits.append((keep, np.linalg.lstsq(np.column_stack([np.ones(len(y)), X[:, keep]]), y, rcond=None)[0]))
    return fits


def cv_mse(rng, X, y):
    sq = np.zeros(K)
    for fold in np.array_split(rng.permutation(len(y)), 5):
        rest = np.setdiff1d(np.arange(len(y)), fold)
        for k, (keep, b) in enumerate(candidates(X[rest], y[rest])):
            sq[k] += ((y[fold] - b[0] - X[fold][:, keep] @ b[1:]) ** 2).sum()
    return sq / len(y)


def live_mse(keep, b):
    full = np.zeros(P)
    full[keep] = b[1:]
    return 1.0 + b[0] ** 2 + float(((full - beta) ** 2).sum())


def predictor(keep, b):
    cols = [names[j] for j in keep]
    return lambda Z: b[0] + Z[cols].to_numpy() @ b[1:]


rng, train, hold, S, H = world(0, m=400)
Xs, ys = train[names].to_numpy(), train["y"].to_numpy()
fits = candidates(Xs, ys)
live = [live_mse(keep, b) for keep, b in fits]
base = float(ys.mean())


cv = cv_mse(rng, Xs, ys)
k_cv = int(np.argmin(cv)) + 1
ho_a = pv.seal(hold[names], hold["y"], "book_no_look", lineage=H, baseline=base, baseline_lineage=S,
               exam_share=0.0, label="no look")
cert_a = ho_a.open(predictor(*fits[k_cv - 1]), lineage=S, label=f"k={k_cv}")
print(f"cross-validation picks k = {k_cv}")
print(cert_a.claim)


ho_b = pv.seal(hold[names], hold["y"], "book_look", lineage=H, baseline=base, baseline_lineage=S, label="look")
answers = [ho_b.ask(predictor(*fits[k - 1]), lineage=S, label=f"k={k}") for k in range(1, K + 1)]
k_look = int(next(r["label"] for r in ho_b.transcript().rows if r["seq"] == ho_b.state["incumbent"])[2:])
cert_b = ho_b.open(predictor(*fits[k_look - 1]), lineage=S, label=f"k={k_look}")
print(f"{len(answers)} answers, {sum(a.accepted for a in answers)} accepts, {ho_b.state['bits']:.2f} bits; the incumbent is k = {k_look}")
print(cert_b.claim)


signal = lambda j: "signal" if j < 5 else "noise"
print(tabulate([[k, f"x{fits[k - 1][0][-1]}", signal(fits[k - 1][0][-1]), cv[k - 1], "ACCEPT" if answers[k - 1].accepted else "reject",
                 answers[k - 1].bits, live[k - 1]] for k in range(1, 16)],
               headers=["k", "feature added", "", "CV MSE (training rows)", "holdout answer", "bits", "live MSE"],
               tablefmt="psql", floatfmt=".4f"))


k_best = int(np.argmin(live)) + 1
print(tabulate([["no look", k_cv, 0.0, cert_a.artifact["bound"], live[k_cv - 1]],
                ["look", k_look, ho_b.state["bits"], cert_b.artifact["bound"], live[k_look - 1]],
                ["best of the 30", k_best, None, None, live[k_best - 1]]],
               headers=["route", "k", "bits", "95% bound", "live MSE"], tablefmt="psql", floatfmt=".4f"))


def study(rep, m):
    """Both routes on world rep with m training rows: {route: (k, bits, bound, live MSE)}."""
    rng, train, hold, S, H = world(rep, m)
    X, y = train[names].to_numpy(), train["y"].to_numpy()
    fits = candidates(X, y)
    k_cv = int(np.argmin(cv_mse(rng, X, y))) + 1
    books = tempfile.mkdtemp(dir=".")
    no_look = pv.seal(hold[names], hold["y"], os.path.join(books, "a"), lineage=H, baseline=float(y.mean()),
                      baseline_lineage=S, exam_share=0.0, seed=rep)
    look = pv.seal(hold[names], hold["y"], os.path.join(books, "b"), lineage=H, baseline=float(y.mean()),
                   baseline_lineage=S, seed=rep)
    for k in range(1, K + 1):
        look.ask(predictor(*fits[k - 1]), lineage=S, label=f"k={k}")
    k_look = int(next(r["label"] for r in look.transcript().rows if r["seq"] == look.state["incumbent"])[2:])
    live = [live_mse(keep, b) for keep, b in fits]
    out = {}
    for route, ho, k in (("no look", no_look, k_cv), ("look", look, k_look)):
        cert = ho.open(predictor(*fits[k - 1]), lineage=S, label=f"k={k}")
        out[route] = (k, ho.state["bits"], cert.artifact["bound"], live[k - 1])
    out["best of the 30"] = (int(np.argmin(live)) + 1, None, None, min(live))
    return out


one = study(0, 4000)
print(tabulate([[route, *v] for route, v in one.items()], headers=["route", "k", "bits", "95% bound", "live MSE"],
               tablefmt="psql", floatfmt=".4f"))


table = []
for m in (400, 4000):
    runs = [study(rep, m) for rep in range(60)]
    for route in ("no look", "look", "best of the 30"):
        r = [run[route] for run in runs]
        priced = r[0][2] is not None
        table.append([m, route, np.mean([x[3] for x in r]), np.mean([x[2] for x in r]) if priced else None,
                      f"{sum(x[2] >= x[3] for x in r)} of 60" if priced else "", np.mean([x[1] for x in r]) if priced else None,
                      np.mean([x[0] for x in r])])
    gain = [run["no look"][3] - run["look"][3] for run in runs]
    print(f"{m:,} training rows: the look chose a better model in {sum(g > 1e-12 for g in gain)} worlds, a worse one in {sum(g < -1e-12 for g in gain)}, the same in {sum(abs(g) <= 1e-12 for g in gain)}")
    print(f"{m:,} training rows: the best of the 30 chosen in", {route: sum(run[route][0] == run["best of the 30"][0] for run in runs) for route in ("no look", "look")}, "worlds of 60")
print(tabulate(table, headers=["training rows", "route", "mean live MSE", "mean 95% bound", "bound held", "mean bits", "mean k"],
               tablefmt="psql", floatfmt=".4f"))
