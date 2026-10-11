import os
import tempfile
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tabulate import tabulate
import provable as pv
from provable import certificates as ct
from provable.certificates import transcript as tr


class NullResultValidation:
    """Simulations of the null result for the mediated holdout: one stationary study, its repetition with 400 training rows, the twenty-question adversary, the two designs where a look gains (a changed law, too few training rows), and the coverage of the Gaussian bar on skewed losses. Every look runs through seal, ask and open in comparative mode under the Gaussian bar (tail="gaussian", the tail these records were made with; a bounded loss now defaults to the exact betting bound); the output is deterministic and carries no timings, so a run reproduces null_result.txt."""

    FEATS = [f"x{i}" for i in range(6)]
    SPECS = [("logit x0", ["x0"], None), ("logit x0 x1", ["x0", "x1"], None), ("logit all", [f"x{i}" for i in range(6)], None), ("gbm depth 2", [f"x{i}" for i in range(6)], 2), ("gbm depth 3", [f"x{i}" for i in range(6)], 3), ("gbm depth 5", [f"x{i}" for i in range(6)], 5)]

    def __init__(self, delta: float): self.delta, self.z0, self.work = delta, tr.z_bar(delta, 0), tempfile.mkdtemp(prefix="null_result_")

    def book(self) -> ct.Ledger: return ct.Ledger(tempfile.mkdtemp(dir=self.work), budget=0.5)

    def world(self, rng: np.random.Generator, n: int, b1: float) -> pd.DataFrame: return (lambda X: X.assign(y=((0.9 * X["x0"] + b1 * X["x1"] + 0.8 * X["x2"] * (X["x3"] > 0) + rng.normal(size=n)) > 0).astype(float)))(pd.DataFrame(rng.normal(size=(n, 6)), columns=self.FEATS))

    def timed(self, frame: pd.DataFrame) -> pd.DataFrame: return frame.assign(start_utc=pd.date_range("2026-01-01", periods=len(frame), freq="h", tz="UTC")).assign(end_utc=lambda f: f["start_utc"] + pd.Timedelta(hours=1))

    def model(self, depth): return LogisticRegression() if depth is None else GradientBoostingClassifier(max_depth=depth, random_state=0)

    def fit(self, train: pd.DataFrame) -> dict: return {name: (lambda m, cols: (lambda X: m.predict_proba(X[cols])[:, 1]))(self.model(depth).fit(train[cols], train["y"]), cols) for name, cols, depth in self.SPECS}

    def cv(self, train: pd.DataFrame, seed: int) -> dict: return {name: float(np.mean([((train.iloc[b]["y"].to_numpy() - self.model(depth).fit(train.iloc[a][cols], train.iloc[a]["y"]).predict_proba(train.iloc[b][cols])[:, 1]) ** 2).mean() for a, b in KFold(5, shuffle=True, random_state=seed).split(train)])) for name, cols, depth in self.SPECS}

    def brier(self, f, frame: pd.DataFrame) -> float: return float(((frame["y"].to_numpy() - f(frame)) ** 2).mean())

    def hand_bound(self, f, hold: pd.DataFrame) -> float: return (lambda l: float(l.mean() + self.z0 * l.std(ddof=1) / np.sqrt(len(l))))((hold["y"].to_numpy() - f(hold)) ** 2)

    def through_ask(self, hold: pd.DataFrame, H, base: float, S, preds: dict, share: float, seed: int, asks: bool) -> dict:
        ho = pv.seal(hold[self.FEATS], hold["y"], self.book(), lineage=H, loss="brier", tail="gaussian", baseline=base, baseline_lineage=S, exam_share=share, seed=seed)
        answers = ["A" if ho.ask(f, lineage=S, label=name).accepted else "R" for name, f in preds.items()] if asks else []
        inc = next((r["label"] for r in ho.transcript().rows if r["seq"] == ho.state["incumbent"]), None) if ho.state["incumbent"] else None
        return {"pick": inc, "answers": answers, "bits": float(ho.state["bits"]), "cert": ho.open(preds[inc] if inc else base, lineage=S, label=str(inc))}

    def no_look(self, hold: pd.DataFrame, H, base: float, S, preds: dict, pick: str, seed: int) -> dict: return (lambda ho: {"pick": pick, "bits": 0.0, "cert": ho.open(preds[pick], lineage=S, label=pick)})(pv.seal(hold[self.FEATS], hold["y"], self.book(), lineage=H, loss="brier", tail="gaussian", baseline=base, baseline_lineage=S, exam_share=0.0, seed=seed))

    def single_study(self, truth_rows: int) -> None:
        rng = np.random.default_rng(20)
        panel = self.timed(self.world(rng, 4500, -0.6))
        train, hold = panel.iloc[:2000], panel.iloc[2000:3000]
        S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "demo", rows=f.index)) for f in (train, hold))
        preds, cv, base = self.fit(train), self.cv(train, 0), float(train["y"].mean())
        truth = self.world(np.random.default_rng(12345), truth_rows, -0.6)
        live = {name: (self.brier(f, truth), float(((f(truth) > 0.5) != truth["y"].to_numpy()).mean())) for name, f in preds.items()}
        bayes = np.asarray(norm.cdf((0.9 * truth["x0"] - 0.6 * truth["x1"] + 0.8 * truth["x2"] * (truth["x3"] > 0)).to_numpy()))
        print(f"S1  one stationary study: 2,000 training rows, 1,000 holdout rows (data seed 20), live risks on {truth_rows:,} fresh rows (seed 12345); cross-validation folds and the exam split with seed 0")
        print(tabulate([[name, cv[name], self.brier(f, hold), live[name][0], live[name][1]] for name, f in preds.items()] + [["base rate", None, float(((hold["y"] - base) ** 2).mean()), float(((truth["y"] - base) ** 2).mean()), float(((base > 0.5) != truth["y"]).mean())], ["Bayes rule", None, None, float(((truth["y"].to_numpy() - bayes) ** 2).mean()), float(((bayes > 0.5) != truth["y"].to_numpy()).mean())]], headers=["model", "5-fold CV Brier", "holdout Brier (1,000)", "live Brier", "live error"], tablefmt="psql", floatfmt=".4f"))
        pick = min(cv, key=cv.get)
        unpriced = min(preds, key=lambda k: self.brier(preds[k], hold))
        routes = [("no look: CV pick, all 1,000 rows at z_0", self.no_look(hold, H, base, S, preds, pick, 0)), ("look: 6 answers on 800 rows, exam slice of 200 at z_0", self.through_ask(hold, H, base, S, preds, 0.2, 0, True)), ("look: 6 answers on 1,000 rows, bar z_L", self.through_ask(hold, H, base, S, preds, 0.0, 0, True))]
        rows = [[name, r["pick"], " ".join(r.get("answers", [])), r["bits"], r["cert"].guarantee.family.get("bar", self.z0), r["cert"].artifact["estimate"], r["cert"].artifact["bound"], r["cert"].artifact["bound"] - r["cert"].artifact["estimate"], live[r["pick"]][0], live[r["pick"]][1]] for name, r in routes]
        rows.append(["unpriced: best of six scores on 1,000 rows at z_0", unpriced, "", None, self.z0, self.brier(preds[unpriced], hold), self.hand_bound(preds[unpriced], hold), self.hand_bound(preds[unpriced], hold) - self.brier(preds[unpriced], hold), live[unpriced][0], live[unpriced][1]])
        print(tabulate(rows, headers=["route", "model", "answers", "bits", "bar", "estimate", "95% bound", "margin", "live Brier", "live error"], tablefmt="psql", floatfmt=".4f"))
        print(f"    no-look bound through seal and open {routes[0][1]['cert'].artifact['bound']:.10f} vs by hand {self.hand_bound(preds[pick], hold):.10f}; s(L) at {routes[2][1]['bits']:.4f} bits = {tr.clean_share(self.delta, routes[2][1]['bits']):.4f}")

    def worlds(self, n_train: int, b_hold: float, reps: int, seed0: int, truth_rows: int, source: str, tag: str) -> None:
        truth = self.world(np.random.default_rng(999), truth_rows, b_hold)
        out = []
        for rep in range(reps):
            rng = np.random.default_rng(seed0 + rep)
            data = self.timed(self.world(rng, n_train + 1000, -0.6) if b_hold == -0.6 else pd.concat([self.world(rng, n_train, -0.6), self.world(rng, 1000, b_hold)], ignore_index=True))
            train, hold = data.iloc[:n_train], data.iloc[n_train:]
            S, H = (ct.Lineage.of(ct.slice_from_frame(data, source, rows=f.index)) for f in (train, hold))
            preds, cv, base = self.fit(train), self.cv(train, rep), float(train["y"].mean())
            live = {name: self.brier(f, truth) for name, f in preds.items()} | {None: float(((truth["y"] - base) ** 2).mean())}
            pick = min(cv, key=cv.get)
            routes = {"no look": self.no_look(hold, H, base, S, preds, pick, rep), "exam": self.through_ask(hold, H, base, S, preds, 0.2, rep, True), "transcript": self.through_ask(hold, H, base, S, preds, 0.0, rep, True)}
            unpriced = min(preds, key=lambda k: self.brier(preds[k], hold))
            out.append({"best": min(v for k, v in live.items() if k is not None), "unpriced live": live[unpriced], "unpriced bound": self.hand_bound(preds[unpriced], hold)} | {f"{k} {q}": v for k, r in routes.items() for q, v in (("pick", r["pick"]), ("live", live[r["pick"]]), ("bound", r["cert"].artifact["bound"]), ("bits", r["bits"]))})
        r = pd.DataFrame(out)
        print(f"{tag}  {reps} worlds, {n_train} training rows, 1,000 holdout rows, x1 coefficient -0.6 in training and {b_hold} in the holdout; seeds {seed0}..{seed0 + reps - 1}; live risks on {truth_rows:,} fresh rows (seed 999)")
        print(tabulate([[k, r[f"{k} live"].mean(), (r[f"{k} live"] - r["best"]).mean(), r[f"{k} bound"].mean(), int((r[f"{k} bound"] >= r[f"{k} live"]).sum()), r[f"{k} bits"].mean()] for k in ("no look", "exam", "transcript")] + [["unpriced best score", r["unpriced live"].mean(), (r["unpriced live"] - r["best"]).mean(), r["unpriced bound"].mean(), int((r["unpriced bound"] >= r["unpriced live"]).sum()), None], ["best candidate", r["best"].mean(), 0.0, None, None, None]], headers=["route", "mean live Brier", "excess over best", "mean 95% bound", f"bound held (of {reps})", "mean bits"], tablefmt="psql", floatfmt=".4f"))
        print("    picks:", {k: r[f"{k} pick"].value_counts(dropna=False).to_dict() for k in ("no look", "exam", "transcript")})
        print("    against no look:", {k: {"better": int((r[f"{k} live"] < r["no look live"] - 1e-12).sum()), "worse": int((r[f"{k} live"] > r["no look live"] + 1e-12).sum()), "same": int((abs(r[f"{k} live"] - r["no look live"]) <= 1e-12).sum())} for k in ("exam", "transcript")}, f"of {reps}")
        print("    pick is the best candidate:", {k: int((abs(r[f"{k} live"] - r["best"]) <= 1e-12).sum()) for k in ("no look", "exam", "transcript")}, f"of {reps}")

    def twenty_questions(self, reps: int, k: int, h: int) -> None:
        rng, wrong_clean, wrong_priced, bits = np.random.default_rng(15), 0, 0, []
        for _ in range(reps):
            X = pd.DataFrame(rng.normal(size=(h, k)), columns=[f"c{j}" for j in range(k)])
            y = rng.normal(size=h)
            hn = pv.seal(X, y, self.book(), lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), claim="gain", eta=0.0, exam_share=0.0)
            cols, inc, signs = list(X.columns), np.zeros(k), []
            for j in range(k):
                w = inc.copy()
                w[j] += 1e-4
                ok = hn.ask(lambda Z, w=w: Z[cols].to_numpy() @ w, lineage=ct.Lineage()).accepted
                signs.append(1.0 if ok else -1.0)
                inc = w if ok else inc
            c = np.sqrt(2 / (np.pi * h))
            f = lambda Z: Z[cols].to_numpy() @ (c * np.asarray(signs))
            cert = hn.open(f, lineage=ct.Lineage())
            d = y ** 2 - (y - f(X)) ** 2
            wrong_clean += int(d.mean() - self.z0 * d.std(ddof=1) / np.sqrt(h) > -c * c * k)
            wrong_priced += int(cert.artifact["bound"] > -c * c * k)
            bits.append(hn.state["bits"])
        print(f"Q20  {reps} null holdouts of {h} rows, {k} sign answers each (eta = 0), seed 15: mean bits {np.mean(bits):.2f} (min {min(bits):.2f}, max {max(bits):.2f}); gain bound wrong at the clean bar {wrong_clean}/{reps}, at the transcript bar {wrong_priced}/{reps}")

    def few_rows(self, m: int, reps: int) -> None:
        P, H, K, beta = 100, 1000, 30, np.r_[np.full(5, 0.25), np.zeros(95)]
        ols = lambda X, y: np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
        risk = lambda b, cols: 1.0 + b[0] ** 2 + float(((np.bincount(cols, weights=b[1:], minlength=P) - beta) ** 2).sum())
        rank = lambda X, y: np.argsort(-np.abs((X - X.mean(0)).T @ (y - y.mean())))
        out = []
        for rep in range(reps):
            rng = np.random.default_rng(3000 + rep)
            Xs = rng.normal(size=(m, P))
            ys = Xs @ beta + rng.normal(size=m)
            Xh = rng.normal(size=(H, P))
            yh = Xh @ beta + rng.normal(size=H)
            order = rank(Xs, ys)
            cands = [(order[:k], ols(Xs[:, order[:k]], ys)) for k in range(1, K + 1)]
            live = np.array([risk(b, cols) for cols, b in cands])
            sq = np.zeros((m, K))
            for f in np.array_split(rng.permutation(m), 5):
                keep = np.setdiff1d(np.arange(m), f)
                o = rank(Xs[keep], ys[keep])
                for k in range(1, K + 1):
                    b = ols(Xs[keep][:, o[:k]], ys[keep])
                    sq[f, k - 1] = (ys[f] - (b[0] + Xs[f][:, o[:k]] @ b[1:])) ** 2
            k_min = int(np.argmin(sq.mean(0)))
            k_1se = next(k for k in range(K) if k == k_min or sq[:, k].mean() - sq[:, k_min].mean() <= (sq[:, k] - sq[:, k_min]).std(ddof=1) / np.sqrt(m))
            frame = pd.DataFrame(Xh, columns=[f"x{j}" for j in range(P)])
            fns = {k: (lambda Z, cols=cols, b=b: b[0] + Z.to_numpy()[:, cols] @ b[1:]) for k, (cols, b) in enumerate(cands)}
            base, base_risk = float(ys.mean()), 1.0 + float((beta ** 2).sum()) + float(ys.mean()) ** 2
            res = {}
            for route, share, asks in (("no look", 0.0, False), ("exam", 0.2, True), ("transcript", 0.0, True)):
                ho = pv.seal(frame, yh, self.book(), lineage="self", baseline=base, baseline_lineage=ct.Lineage(), exam_share=share, seed=rep)
                for k in (range(K) if asks else []):
                    ho.ask(fns[k], lineage=ct.Lineage(), label=str(k))
                kv = k_min if not asks else (None if ho.state["incumbent"] == 0 else int(next(r["label"] for r in ho.transcript().rows if r["seq"] == ho.state["incumbent"])))
                c = ho.open(fns[kv] if kv is not None else base, lineage=ct.Lineage())
                res[route] = (kv, base_risk if kv is None else live[kv], c.artifact["bound"], ho.state["bits"])
            out.append({"best": live.min(), "k best": int(np.argmin(live)) + 1, "cv 1se live": live[k_1se], "k cv 1se": k_1se + 1} | {f"{k} {q}": v for k, t in res.items() for q, v in zip(("k", "live", "bound", "bits"), (None if t[0] is None else t[0] + 1, t[1], t[2], t[3]))})
        r = pd.DataFrame(out)
        print(f"F{m}  {reps} worlds, regression, 100 features of which 5 carry 0.25, noise variance 1; {m} training rows, 1,000 holdout rows; 30 candidates (OLS on the top k by training correlation); seeds 3000..{3000 + reps - 1}; live MSE exact")
        print(tabulate([[k, r[f"{k} live"].mean(), (r[f"{k} live"] - r["best"]).mean(), r[f"{k} bound"].mean(), int((r[f"{k} bound"] >= r[f"{k} live"]).sum()), r[f"{k} bits"].mean(), r[f"{k} k"].mean()] for k in ("no look", "exam", "transcript")] + [["no look, one-standard-error rule", r["cv 1se live"].mean(), (r["cv 1se live"] - r["best"]).mean(), None, None, None, r["k cv 1se"].mean()], ["best candidate", r["best"].mean(), 0.0, None, None, None, r["k best"].mean()]], headers=["route", "mean live MSE", "excess over best", "mean 95% bound", f"bound held (of {reps})", "mean bits", "mean k"], tablefmt="psql", floatfmt=".4f"))
        print("    against no look:", {k: {"better": int((r[f"{k} live"] < r["no look live"] - 1e-12).sum()), "worse": int((r[f"{k} live"] > r["no look live"] + 1e-12).sum()), "same": int((abs(r[f"{k} live"] - r["no look live"]) <= 1e-12).sum())} for k in ("exam", "transcript")}, f"of {reps}")
        print("    pick is the best candidate:", {k: int((abs(r[f"{k} live"] - r["best"]) <= 1e-12).sum()) for k in ("no look", "exam", "transcript")}, f"of {reps}")

    def gaussian_form(self, reps: int, seed: int) -> None:
        rng, rows = np.random.default_rng(seed), []
        for n in (200, 1000):
            sq = 1.02 * rng.standard_normal((reps, n)) ** 2
            p = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 1.6, size=(reps, n))))
            br = ((rng.random((reps, n)) < p) - p) ** 2
            rows.append([n, float(np.mean(sq.mean(1) + self.z0 * sq.std(1, ddof=1) / np.sqrt(n) >= 1.02)), float(np.mean(br.mean(1) + self.z0 * br.std(1, ddof=1) / np.sqrt(n) >= float(np.mean(p * (1.0 - p)))))])
        print(f"G   Gaussian-form upper bound for one fixed model, nominal {1 - self.delta:.2f}, {reps:,} replications, seed {seed}: squared error of a N(0, 1.02) residual, Brier loss of a calibrated model with logit N(0, 1.6^2)")
        print(tabulate(rows, headers=["rows", "coverage, squared error", "coverage, Brier"], tablefmt="psql", floatfmt=".4f"))

    def run(self) -> None:
        self.single_study(2_000_000)
        self.worlds(400, -0.6, 60, 1000, 200_000, "mc", "R400")
        self.twenty_questions(200, 20, 400)
        self.few_rows(400, 60)
        self.few_rows(4000, 60)
        self.worlds(2000, 0.6, 40, 2000, 200_000, "sh", "SHIFT")
        self.gaussian_form(20_000, 7)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    v = NullResultValidation(0.05)
    v.run()
