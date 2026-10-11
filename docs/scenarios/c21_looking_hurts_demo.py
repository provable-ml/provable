# The demo of docs/scenarios/c21-looking-hurts.md: the page's code blocks, in order.
# From the repository root: uv run python docs/scenarios/c21_looking_hurts_demo.py

import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
import provable as pv
from provable import certificates as ct
from provable.certificates import transcript as tr

os.chdir(tempfile.mkdtemp(prefix="looking_hurts_"))
P, K = 200, 30
beta = np.r_[np.full(5, 0.5), np.zeros(P - 5)]
names = [f"x{j}" for j in range(P)]


def world(rep, m=400):
    rng = np.random.default_rng(5000 + rep)
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


def ranked(X, y):
    return np.argsort(-np.abs((X - X.mean(0)).T @ (y - y.mean())))


def fit(X, y, keep):
    return np.linalg.lstsq(np.column_stack([np.ones(len(y)), X[:, keep]]), y, rcond=None)[0]


def cv_pick(rng, X, y):
    sq = np.zeros(K)
    for fold in np.array_split(rng.permutation(len(y)), 5):
        rest = np.setdiff1d(np.arange(len(y)), fold)
        order = ranked(X[rest], y[rest])
        for k in range(1, K + 1):
            b = fit(X[rest], y[rest], order[:k])
            sq[k - 1] += ((y[fold] - b[0] - X[fold][:, order[:k]] @ b[1:]) ** 2).sum()
    return int(np.argmin(sq)) + 1


def live_mse(keep, b):
    full = np.zeros(P)
    full[list(keep)] = b[1:]
    return 1.0 + b[0] ** 2 + float(((full - beta) ** 2).sum())


def predictor(keep, b):
    cols = [names[j] for j in keep]
    return lambda Z: b[0] + Z[cols].to_numpy() @ b[1:]


rng, train, hold, S, H = world(0)
Xs, ys = train[names].to_numpy(), train["y"].to_numpy()
Xh, yh = hold[names].to_numpy(), hold["y"].to_numpy()
order = ranked(Xs, ys)
k_cv = cv_pick(rng, Xs, ys)
start = [int(j) for j in order[:k_cv]]
b_start = fit(Xs, ys, start)
rest = [int(j) for j in order[k_cv:]]
print(f"cross-validation picks k = {k_cv}: {', '.join(names[j] for j in start)}; {len(rest)} features left to try")


ho_a = pv.seal(hold[names], hold["y"], "book_no_look", lineage=H, baseline=float(ys.mean()), baseline_lineage=S,
               exam_share=0.0, label="no look")
cert_a = ho_a.open(predictor(start, b_start), lineage=S, label=f"k={k_cv}")
print(cert_a.claim)


def holdout_mse(keep, b):
    return float(((yh - b[0] - Xh[:, keep] @ b[1:]) ** 2).mean())


keep_u, b_u = list(start), b_start
for j in rest:
    b = fit(Xs, ys, keep_u + [j])
    if holdout_mse(keep_u + [j], b) < holdout_mse(keep_u, b_u):
        keep_u, b_u = keep_u + [j], b
loss = (yh - b_u[0] - Xh[:, keep_u] @ b_u[1:]) ** 2
reported = loss.mean() + tr.z_bar(0.05, 0) * loss.std(ddof=1) / np.sqrt(len(loss))
added = keep_u[len(start):]
print(f"kept {len(added)} of {len(rest)} features tried, {sum(j >= 5 for j in added)} of them noise")
print(f"holdout MSE {holdout_mse(start, b_start):.4f} -> {holdout_mse(keep_u, b_u):.4f}; reported bound at the clean bar {reported:.4f}")


ho_b = pv.seal(hold[names], hold["y"], "book_look", lineage=H, baseline=float(ys.mean()), baseline_lineage=S, label="look")
first = ho_b.ask(predictor(start, b_start), lineage=S, label=f"k={k_cv}")
keep_v, b_v = list(start), b_start
answers = [first]
for j in rest:
    b = fit(Xs, ys, keep_v + [j])
    try:
        a = ho_b.ask(predictor(keep_v + [j], b), lineage=S, label=f"+{names[j]}")
    except ValueError as e:
        print(f"after {len(answers)} answers: {e}")
        break
    answers.append(a)
    if a.accepted:
        keep_v, b_v = keep_v + [j], b
added_v = keep_v[len(start):]
print(f"{len(answers)} answers, {sum(a.accepted for a in answers)} accepts, {ho_b.state['bits']:.2f} bits; "
      f"kept {len(added_v)} features, {sum(j >= 5 for j in added_v)} of them noise")
print(ho_b)
cert_b = ho_b.open(predictor(keep_v, b_v), lineage=S, label="incumbent")
print(cert_b.claim)


rows = [["no look", k_cv, 0.0, cert_a.artifact["estimate"], cert_a.artifact["bound"], live_mse(start, b_start)],
        ["look, no price", len(keep_u), None, holdout_mse(keep_u, b_u), reported, live_mse(keep_u, b_u)],
        ["look through ask", len(keep_v), ho_b.state["bits"], cert_b.artifact["estimate"], cert_b.artifact["bound"], live_mse(keep_v, b_v)]]
print(tabulate(rows, headers=["route", "features", "bits", "reported MSE", "95% bound", "live MSE"], tablefmt="psql", floatfmt=".4f"))


def study(rep):
    """The three routes on world rep: {route: (features, bits, reported MSE, bound, live MSE)}."""
    rng, train, hold, S, H = world(rep)
    Xs, ys = train[names].to_numpy(), train["y"].to_numpy()
    Xh, yh = hold[names].to_numpy(), hold["y"].to_numpy()
    order = ranked(Xs, ys)
    k_cv = cv_pick(rng, Xs, ys)
    start, rest = [int(j) for j in order[:k_cv]], [int(j) for j in order[k_cv:]]
    hmse = lambda keep, b: float(((yh - b[0] - Xh[:, keep] @ b[1:]) ** 2).mean())
    books = tempfile.mkdtemp(dir=".")
    out = {}
    b0 = fit(Xs, ys, start)
    no_look = pv.seal(hold[names], hold["y"], os.path.join(books, "a"), lineage=H, baseline=float(ys.mean()),
                      baseline_lineage=S, exam_share=0.0, seed=rep)
    cert = no_look.open(predictor(start, b0), lineage=S)
    out["no look"] = (start, 0.0, cert.artifact["estimate"], cert.artifact["bound"], live_mse(start, b0))
    keep, bk = list(start), b0
    for j in rest:
        b = fit(Xs, ys, keep + [j])
        if hmse(keep + [j], b) < hmse(keep, bk):
            keep, bk = keep + [j], b
    loss = (yh - bk[0] - Xh[:, keep] @ bk[1:]) ** 2
    out["look, no price"] = (keep, None, loss.mean(), loss.mean() + tr.z_bar(0.05, 0) * loss.std(ddof=1) / np.sqrt(len(loss)), live_mse(keep, bk))
    look = pv.seal(hold[names], hold["y"], os.path.join(books, "b"), lineage=H, baseline=float(ys.mean()),
                   baseline_lineage=S, seed=rep)
    keep, bk, retired = list(start), b0, False
    look.ask(predictor(start, b0), lineage=S)
    for j in rest:
        b = fit(Xs, ys, keep + [j])
        try:
            if look.ask(predictor(keep + [j], b), lineage=S).accepted:
                keep, bk = keep + [j], b
        except ValueError:
            retired = True
            break
    cert = look.open(predictor(keep, bk), lineage=S)
    out["look through ask"] = (keep, look.state["bits"], cert.artifact["estimate"], cert.artifact["bound"], live_mse(keep, bk))
    return out, len(start), retired


runs = [study(rep) for rep in range(60)]
table = []
for route in ("no look", "look, no price", "look through ask"):
    r = [run[route] for run, _, _ in runs]
    noise = np.mean([sum(j >= 5 for j in x[0][n:]) for (run, n, _), x in zip(runs, r)])
    table.append([route, np.mean([x[2] for x in r]), np.mean([x[4] for x in r]), np.mean([x[3] for x in r]), f"{sum(x[3] >= x[4] for x in r)} of 60",
                  noise, np.mean([x[1] for x in r]) if r[0][1] is not None else None])
print(tabulate(table, headers=["route", "mean reported MSE", "mean live MSE", "mean 95% bound", "bound held", "noise features kept", "mean bits"],
               tablefmt="psql", floatfmt=".4f"))
for route in ("look, no price", "look through ask"):
    d = [run[route][4] - run["no look"][4] for run, _, _ in runs]
    print(f"{route:16s} against no look: live worse in {sum(x > 1e-12 for x in d)} worlds, better in {sum(x < -1e-12 for x in d)}")
print(f"the tuning slice retired before the loop ended in {sum(ret for _, _, ret in runs)} worlds of 60")
