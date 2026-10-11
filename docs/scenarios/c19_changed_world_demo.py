# The demo of docs/scenarios/c19-changed-world.md: the page's code blocks, in order.
# From the repository root: uv run python docs/scenarios/c19_changed_world_demo.py

import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
import provable as pv
from provable import certificates as ct

os.chdir(tempfile.mkdtemp(prefix="changed_world_"))
feats = [f"x{i}" for i in range(6)]


def draw(rng, n):
    return pd.DataFrame(rng.normal(size=(n, 6)), columns=feats), rng.normal(size=n)


def label(X, noise, b1):
    return X.assign(y=(0.9 * X["x0"] + b1 * X["x1"] + 0.8 * X["x2"] * (X["x3"] > 0) + noise > 0).astype(float))


def world(rep, b_new):
    rng = np.random.default_rng(2000 + rep)
    panel = pd.concat([label(*draw(rng, 2000), -0.6), label(*draw(rng, 1000), b_new)], ignore_index=True)
    panel["start_utc"] = pd.date_range("2026-01-01", periods=3000, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    train, hold = panel.iloc[:2000], panel.iloc[2000:]
    S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=f.index)) for f in (train, hold))
    return train, hold, S, H


later = {b: label(*draw(np.random.default_rng(999), 200_000), b) for b in (0.6, -0.6)}
models = {"logit x0": (LogisticRegression, {}, ["x0"]),
          "logit x0 x1": (LogisticRegression, {}, ["x0", "x1"]),
          "logit all": (LogisticRegression, {}, feats),
          "gbm depth 2": (GradientBoostingClassifier, {"max_depth": 2, "random_state": 0}, feats),
          "gbm depth 3": (GradientBoostingClassifier, {"max_depth": 3, "random_state": 0}, feats),
          "gbm depth 5": (GradientBoostingClassifier, {"max_depth": 5, "random_state": 0}, feats)}


def fit(name, rows):
    make, kw, cols = models[name]
    m = make(**kw).fit(rows[cols], rows["y"])
    return lambda X: m.predict_proba(X[cols])[:, 1]


def brier(f, rows):
    return float(((rows["y"].to_numpy() - f(rows)) ** 2).mean())


def error(f, rows):
    return float(((f(rows) > 0.5) != rows["y"].to_numpy()).mean())


def cv_brier(name, rows, rep):
    return float(np.mean([brier(fit(name, rows.iloc[a]), rows.iloc[b]) for a, b in KFold(5, shuffle=True, random_state=rep).split(rows)]))


train, hold, S, H = world(0, b_new=0.6)
candidates = {name: fit(name, train) for name in models}
base = float(train["y"].mean())


cv = {name: cv_brier(name, train, 0) for name in models}
print(tabulate(list(cv.items()), headers=["model", "5-fold CV Brier on the training rows"], tablefmt="psql", floatfmt=".4f"))
pick = min(cv, key=cv.get)
ho_a = pv.seal(hold[feats], hold["y"], "book_no_look", lineage=H, loss="brier", baseline=base,
               baseline_lineage=S, exam_share=0.0, label="no look")
cert_a = ho_a.open(candidates[pick], lineage=S, label=pick)
print(cert_a.claim)


ho_b = pv.seal(hold[feats], hold["y"], "book_look", lineage=H, loss="brier", baseline=base,
               baseline_lineage=S, label="look")
for name, f in candidates.items():
    print(f"{name:12s}", ho_b.ask(f, lineage=S, label=name))
chosen = next(r["label"] for r in ho_b.transcript().rows if r["seq"] == ho_b.state["incumbent"])
cert_b = ho_b.open(candidates[chosen], lineage=S, label=chosen)
print(cert_b.claim)


live = later[0.6]
best = min(models, key=lambda name: brier(candidates[name], live))
rows = [["no look", pick, 0.0, cert_a.artifact["bound"]], ["look", chosen, ho_b.state["bits"], cert_b.artifact["bound"]], ["best of the six", best, None, None]]
print(tabulate([r + [brier(candidates[r[1]], live), error(candidates[r[1]], live)] for r in rows],
               headers=["route", "model", "bits", "95% bound", "live Brier", "live error"], tablefmt="psql", floatfmt=".4f"))


def study(rep):
    """Both routes on world rep, with the change and without it: the training rows, the candidates and the
    cross-validated pick are shared, and the holdout differs only in the coefficient of x1."""
    train, _, S, _ = world(rep, 0.6)
    candidates = {name: fit(name, train) for name in models}
    cv = {name: cv_brier(name, train, rep) for name in models}
    pick, base, out = min(cv, key=cv.get), float(train["y"].mean()), {}
    for b_new in (0.6, -0.6):
        _, hold, _, H = world(rep, b_new)
        books = tempfile.mkdtemp(dir=".")
        no_look = pv.seal(hold[feats], hold["y"], os.path.join(books, "a"), lineage=H, loss="brier", baseline=base,
                          baseline_lineage=S, exam_share=0.0, seed=rep)
        look = pv.seal(hold[feats], hold["y"], os.path.join(books, "b"), lineage=H, loss="brier", baseline=base,
                       baseline_lineage=S, seed=rep)
        for name, f in candidates.items():
            look.ask(f, lineage=S, label=name)
        chosen = next(r["label"] for r in look.transcript().rows if r["seq"] == look.state["incumbent"])
        for route, ho, name in (("no look", no_look, pick), ("look", look, chosen)):
            cert = ho.open(candidates[name], lineage=S, label=name)
            out[b_new, route] = (name, ho.state["bits"], cert.artifact["bound"], brier(candidates[name], later[b_new]))
        best = min(models, key=lambda name: brier(candidates[name], later[b_new]))
        out[b_new, "best of the six"] = (best, None, None, brier(candidates[best], later[b_new]))
    return out


one = study(0)
print(tabulate([[("changed" if b == 0.6 else "unchanged"), route, *v] for (b, route), v in one.items()],
               headers=["holdout", "route", "model", "bits", "95% bound", "live Brier"], tablefmt="psql", floatfmt=".4f"))


runs = [study(rep) for rep in range(40)]
table = []
for b_new, what in ((0.6, "changed"), (-0.6, "unchanged")):
    for route in ("no look", "look", "best of the six"):
        r = [run[b_new, route] for run in runs]
        held = f"{sum(x[2] >= x[3] for x in r)} of 40" if r[0][2] is not None else ""
        table.append([what, route, np.mean([x[3] for x in r]), np.mean([x[2] for x in r]) if r[0][2] is not None else None, held,
                      np.mean([x[1] for x in r]) if r[0][1] is not None else None])
print(tabulate(table, headers=["holdout", "route", "mean live Brier", "mean 95% bound", "bound held", "mean bits"], tablefmt="psql", floatfmt=".4f"))
for b_new, what in ((0.6, "changed"), (-0.6, "unchanged")):
    print(f"{what:9s} models chosen:", {route: pd.Series([run[b_new, route][0] for run in runs]).value_counts().to_dict() for route in ("no look", "look")})
    print(f"{what:9s} the best of the six chosen in:", {route: sum(run[b_new, route][0] == run[b_new, "best of the six"][0] for run in runs) for route in ("no look", "look")}, "worlds of 40")
