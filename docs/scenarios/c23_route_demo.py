# The demo of docs/scenarios/c23-transport-screen.md: the page's code blocks, in order.
# From the repository root: uv run python docs/scenarios/c23_route_demo.py

import os, tempfile
import numpy as np, pandas as pd
from tabulate import tabulate
import provable as pv
from provable import certificates as ct

os.chdir(tempfile.mkdtemp(prefix="route_"))
per, M_S, M_H = 168, 48, 4
a1 = 0.3


def world(rep):
    rng = np.random.default_rng(23000 + rep)
    M = M_S + M_H
    state = rng.choice(["A", "B"], size=M)
    panel = pd.DataFrame(rng.normal(size=(M * per, 2)), columns=["x0", "x1"])
    panel["week"] = np.repeat(np.arange(M), per)
    panel["state"] = np.repeat(state, per)
    panel["y"] = 0.4 * panel["x0"] + np.where(panel["state"] == "A", a1, 0.0) * panel["x1"] + rng.normal(size=M * per)
    panel["start_utc"] = pd.date_range("2026-01-05", periods=M * per, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    return panel


def ols(frame, cols):
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in cols])
    b = np.linalg.lstsq(X, frame["y"].to_numpy(), rcond=None)[0]
    return np.r_[b[0], [b[1 + cols.index(c)] if c in cols else 0.0 for c in ("x0", "x1")]]


def fit(choice, frame):
    """The two alternatives of a choice, fitted on frame: (the default, the alternative)."""
    if choice == "x0":
        return ols(frame, []), ols(frame, ["x0"])
    return ols(frame, ["x0"]), ols(frame[frame["state"] == "A"], ["x0", "x1"])


def loss(b, frame):
    return (frame["y"].to_numpy() - b[0] - b[1] * frame["x0"].to_numpy() - b[2] * frame["x1"].to_numpy()) ** 2


def crossfit(choice, train):
    """The default's loss minus the alternative's on every training row, each week scored by models fitted on the other weeks."""
    d = np.empty(len(train))
    for wk in train["week"].unique():
        here = (train["week"] == wk).to_numpy()
        base, alt = fit(choice, train[~here])
        d[here] = loss(base, train[here]) - loss(alt, train[here])
    return d


def transported(b):
    """The live risk of a fitted model over the long run, half the weeks in each state."""
    return 1.0 + b[0] ** 2 + (0.4 - b[1]) ** 2 + 0.5 * (a1 - b[2]) ** 2 + 0.5 * b[2] ** 2


panel = world(29)
train, hold = panel[panel["week"] < M_S], panel[panel["week"] >= M_S]
S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "weeks", rows=f.index)) for f in (train, hold))
print(f"{M_S} training weeks ({(train.groupby('week')['state'].first() == 'A').sum()} in state A), {M_H} holdout weeks in states {''.join(hold.groupby('week')['state'].first())}")


holdouts, routes = {}, {}
for choice in ("x0", "x1"):
    base, alt = fit(choice, train)
    book = ct.Ledger(f"book_{choice}", budget=0.10)
    holdouts[choice] = pv.seal(hold[["x0", "x1"]], hold["y"], book, lineage=H, claim="gain", exam_share=0.25,
                               baseline=lambda Z, b=base: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"], baseline_lineage=S, label=choice)
    routes[choice] = pv.route(crossfit(choice, train), train["week"], holdout=holdouts[choice], lineage=S, ledger=book,
                              label="x0 in" if choice == "x0" else "x1 at its state-A coefficient")
    print(routes[choice])


for choice in ("x0", "x1"):
    base, alt = fit(choice, train)
    ho = holdouts[choice]
    ans = ho.ask(lambda Z, b=alt: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"], lineage=S, label="alternative")
    line = f"{choice}: route {routes[choice].verdict}; the holdout answers {ans}"
    if ans.accepted:
        cert = ho.open(lambda Z, b=alt: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"], lineage=S)
        line += f"; certified gain at least {cert.artifact['bound']:.4f}, long-run gain {transported(base) - transported(alt):.4f}"
    print(line)


def study(rep):
    panel = world(rep)
    train, hold = panel[panel["week"] < M_S], panel[panel["week"] >= M_S]
    S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "weeks", rows=f.index)) for f in (train, hold))
    out = {"mostly A": int((hold.groupby("week")["state"].first() == "A").sum()) >= 3}
    for choice in ("x0", "x1"):
        base, alt = fit(choice, train)
        book = ct.Ledger(tempfile.mkdtemp(dir="."), budget=0.10)
        ho = pv.seal(hold[["x0", "x1"]], hold["y"], book, lineage=H, claim="gain", exam_share=0.25, seed=rep,
                     baseline=lambda Z, b=base: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"], baseline_lineage=S)
        f = lambda Z, b=alt: b[0] + b[1] * Z["x0"] + b[2] * Z["x1"]
        out[choice + " route"] = pv.route(crossfit(choice, train), train["week"], holdout=ho, lineage=S).verdict
        out[choice + " long run"] = transported(base) - transported(alt)
        cert = ho.open(f, lineage=S) if ho.ask(f, lineage=S).accepted else None
        out[choice + " estimate"] = None if cert is None else cert.artifact["estimate"]
        out[choice + " bound"] = None if cert is None else cert.artifact["bound"]
    return out


r = pd.DataFrame([study(rep) for rep in range(100)])
rows = []
for label, choice, keep in (("x0 in", "x0", r.index == r.index), ("x1, asked every time", "x1", r.index == r.index), ("x1, asked when memory says mostly A", "x1", r["mostly A"])):
    s = r[keep]
    cert = s[s[choice + " bound"].notna()]
    rows.append([label, f"{(s[choice + ' route'] == 'VAULT').sum()} of {len(s)}", f"{len(cert)} of {len(s)}", cert[choice + " estimate"].mean(), cert[choice + " bound"].mean(),
                 cert[choice + " long run"].mean(), int((cert[choice + " bound"] > cert[choice + " long run"]).sum())])
print(tabulate(rows, headers=["choice", "routed to VAULT", "certified", "mean gain on the exam rows", "mean certified bound", "mean long-run gain", "bound above the long-run gain"], tablefmt="psql", floatfmt=".4f"))
