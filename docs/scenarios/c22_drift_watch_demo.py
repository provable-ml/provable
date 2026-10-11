# The demo of docs/scenarios/c22-drift-watch.md: the page's code blocks, in order.
# From the repository root: uv run python docs/scenarios/c22_drift_watch_demo.py

import os, tempfile
import numpy as np, pandas as pd
from scipy.stats import norm
from tabulate import tabulate
import provable as pv
from provable import certificates as ct

os.chdir(tempfile.mkdtemp(prefix="drift_"))
sigma, tau, w = 1.0, 0.05, 35
n_hist, h, J = 2000, 200, 1500
m = n_hist - h
RANGE = (0.0, 16.0)


def world(rep, tau=tau):
    rng = np.random.default_rng(22000 + rep)
    level = np.cumsum(rng.normal(scale=tau, size=n_hist + J))
    panel = pd.DataFrame({"row": np.arange(n_hist + J), "y": level + rng.normal(scale=sigma, size=n_hist + J)})
    panel["start_utc"] = pd.date_range("2026-01-01", periods=n_hist + J, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    return level, panel


def clipped_risk(gap, var, c=RANGE[1]):
    """E min(r^2, c) for r ~ N(gap, var): the live risk of a row, on the certificate's range."""
    s, a = np.sqrt(var), np.sqrt(c)
    lo, hi = (-a - gap) / s, (a - gap) / s
    inside = norm.cdf(hi) - norm.cdf(lo)
    return (gap ** 2 + var) * inside + 2 * gap * s * (norm.pdf(lo) - norm.pdf(hi)) + var * (lo * norm.pdf(lo) - hi * norm.pdf(hi)) + c * (1 - inside)


level, panel = world(0)
train, hold, live = panel.iloc[:m], panel.iloc[m:n_hist], panel.iloc[n_hist:]
S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "drift", rows=f.index)) for f in (train, hold))
f_S = float(train["y"].iloc[-w:].mean())
f_n = float(panel["y"].iloc[n_hist - w:n_hist].mean())
print(f"trailing mean of the last {w} rows: {f_S:.4f} at the end of training, {f_n:.4f} refit through the holdout")


ho = pv.seal(hold[["row"]], hold["y"], ct.Ledger("drift_book", budget=0.10), lineage=H, baseline=float(train["y"].mean()),
             baseline_lineage=S, exam_share=0.0, loss_range=RANGE, clock=(hold["start_utc"], hold["end_utc"]))
cert = ho.open(lambda Z: np.full(len(Z), f_S), lineage=S, label="trailing mean")
U = cert.artifact["bound"]
print(cert.claim)


def leads_table(reps=20000, leads=(1, 25, 50, 75, 100, 101, 125, 150, 200, 300)):
    rng = np.random.default_rng(22)
    steps = rng.normal(scale=tau, size=(reps, w + h))
    lev = np.cumsum(steps, axis=1)
    y = lev + rng.normal(scale=sigma, size=(reps, w + h))
    fs = y[:, :w].mean(axis=1)
    fn = y[:, h:].mean(axis=1)
    score = ((y[:, w:] - fs[:, None]) ** 2).mean(axis=1)
    rows = []
    for j in leads:
        live = (lev[:, -1] - fn) ** 2 + j * tau ** 2 + sigma ** 2
        rows.append([j, live.mean(), score.mean(), live.mean() - score.mean(), tau ** 2 * (j - (h + 1) / 2)])
    return rows


print(tabulate(leads_table(), headers=["lead j", "live risk of the refit model", "holdout score of f_S", "difference", "tau^2 (j - (h+1)/2)"], tablefmt="psql", floatfmt=".4f"))


wa = pv.watch(cert, "drift_book")
lead = 0
while lead < J:
    day = live.iloc[lead:lead + 24]
    st = wa.update(day["y"], np.full(len(day), f_n), times=day["start_utc"])
    lead += len(day)
    if st.status != "watching":
        break
risk = clipped_risk(level[n_hist - 1:n_hist + J - 1] - f_n, tau ** 2 + sigma ** 2)
cross = int(np.argmax(risk > U)) + 1
print(st)
print(f"the refit model's live risk first exceeds the certified level {U:.4f} at lead {cross}; the watch alarmed at lead {st.alarm['row']}")
a0, a1 = st.alarm["start"], st.alarm["row"]
gap = risk - U
print(f"leads {cross} to {a0 - 1}: live risk above the certified level at {(gap[cross - 1:a0 - 1] > 0).mean():.0%} of leads, by {gap[cross - 1:a0 - 1].mean():.2f} on average; leads {a0} to {a1}: by {gap[a0 - 1:a1].mean():.2f}")
print(pv.Ledger("drift_book").statement())


def study(rep, tau=tau):
    """One world: certify f_S on the latest h rows, deploy the refit f_n, and watch it twice, without a horizon and with a horizon of J rows."""
    level, panel = world(rep, tau)
    train, hold, live = panel.iloc[:m], panel.iloc[m:n_hist], panel.iloc[n_hist:]
    S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "drift", rows=f.index)) for f in (train, hold))
    f_S, f_n = float(train["y"].iloc[-w:].mean()), float(panel["y"].iloc[n_hist - w:n_hist].mean())
    book = ct.Ledger(tempfile.mkdtemp(dir="."), budget=0.15)
    ho = pv.seal(hold[["row"]], hold["y"], book, lineage=H, baseline=float(train["y"].mean()), baseline_lineage=S,
                 exam_share=0.0, loss_range=RANGE, clock=(hold["start_utc"], hold["end_utc"]), seed=rep)
    cert = ho.open(lambda Z: np.full(len(Z), f_S), lineage=S)
    U = cert.artifact["bound"]
    risk = clipped_risk(level[n_hist - 1:n_hist + J - 1] - f_n, tau ** 2 + sigma ** 2)
    out = {"held": bool(risk[0] <= U), "cross": int(np.argmax(risk > U)) + 1 if (risk > U).any() else None}
    for name, horizon in (("anytime", None), ("horizon", J)):
        wa = pv.watch(cert, book, horizon=horizon)
        for k in range(0, J, 168):
            week = live.iloc[k:k + 168]
            st = wa.update(week["y"], np.full(len(week), f_n), times=week["start_utc"])
            if st.status != "watching":
                break
        out[name] = st.alarm["row"] if st.alarm else None
    return out


rows = []
for label, t in (("drift, tau 0.05", tau), ("no drift", 0.0)):
    r = pd.DataFrame([study(rep, t) for rep in range(60)])
    for name in ("anytime", "horizon"):
        a = r[name]
        early = int((a.notna() & (r["cross"].isna() | (a < r["cross"]))).sum())
        delay = (a - r["cross"]).dropna()
        rows.append([label, "none" if name == "anytime" else J, int(r["held"].sum()), int(r["cross"].notna().sum()), r["cross"].median() if r["cross"].notna().any() else None,
                     int(a.notna().sum()), early, a.median() if a.notna().any() else None, delay.median() if len(delay) else None])
print(tabulate(rows, headers=["world, 60 each", "horizon", "bound held at lead 1", "risk exceeded the bound", "median lead it did", "alarms", "alarms before it did", "median alarm lead", "median delay"], tablefmt="psql", floatfmt=".0f"))
