import os
import tempfile
import warnings
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tabulate import tabulate
import provable as pv
from provable import certificates as ct


class GuaranteeValidation:
    """Checks of the value-side results for a look: safety and power of the gated look against the minimax threshold, the closed-form harm of an ungated look on noise candidates, the gated look in the stationary, scarce-row and changed-law worlds, and the recency premium under a persistent block effect."""

    def __init__(self, delta: float, seed: int): self.delta, self.rng = delta, np.random.default_rng(seed)

    @staticmethod
    def z(level: float) -> float: return float(norm.isf(level))

    def gate(self, d: np.ndarray, zc: float) -> np.ndarray: return np.where(d.max(1) > zc, np.argmax(d, 1) + 1, 0)

    def menu(self, K: int, reps: int, beta: float) -> None:
        zc = self.z(self.delta / K)
        rows = []
        for t in (0.0, 0.1, 0.5, 1.0):
            d = self.rng.standard_normal((reps, K)) - t
            rows.append([f"all {K} candidates worse by {t} s.e.", float((self.gate(d, zc) > 0).mean()), float((d.max(1) > 0).mean())])
        d = self.rng.standard_normal((reps, K)) + np.r_[3.0, np.full(K - 1, -0.1)]
        j = self.gate(d, zc)
        rows.append([f"one better by 3 s.e., {K - 1} worse by 0.1 s.e.", float((j > 1).mean()), float(((np.argmax(d, 1) > 0) & (d.max(1) > 0)).mean())])
        print(f"A1  P(the look hurts), gated at z_(delta/K) = {zc:.3f} against the ungated argmax; delta {self.delta}, K {K}, {reps:,} replications")
        print(tabulate(rows, headers=["world", "gated", "argmax of the holdout gains"], tablefmt="psql", floatfmt=".4f"))
        rows = []
        for theta in (2.0, 3.0, 4.0, 5.0):
            d = self.rng.standard_normal((reps, K)) + np.r_[theta, np.zeros(K - 1)]
            rows.append([theta, float((self.gate(d, zc) == 1).mean()), float(norm.cdf(theta - zc)), float(norm.cdf(theta - zc) - self.delta), float(norm.cdf(theta - self.z(self.delta)))])
        print(f"A2  power to buy a gain carried by one unnamed candidate of {K}; necessity: no delta-safe look exceeds Phi(theta - z_(delta/K)) for every candidate")
        print(tabulate(rows, headers=["theta = gain sqrt(h)/sigma", "gated look", "minimax bound Phi(theta - z_(d/K))", "gate lower bound (bound - delta)", "single named candidate Phi(theta - z_d)"], tablefmt="psql", floatfmt=".4f"))
        print(f"    rows to buy a gain with probability {1 - beta} from a menu of {K}, relative to one named candidate: {((zc + self.z(beta)) / (self.z(self.delta) + self.z(beta))) ** 2:.3f}")
        ts = np.linspace(0.0, 3.0, 3001)
        bound = lambda Kk: Kk * max(t * norm.sf(self.z(self.delta / Kk) + t) for t in ts)
        worst = lambda Kk: max(t * (1 - norm.cdf(self.z(self.delta / Kk) + t) ** Kk) for t in ts)
        print(f"A3  expected harm of the gated look, in standard errors of the paired gain: bound K max_t t Phibar(z_(delta/K) + t): K=1 {bound(1):.4f}, K={K} {bound(K):.4f}; worst case with all K independent candidates worse by t: K=1 {worst(1):.4f}, K={K} {worst(K):.4f}")

    def harm(self, m: int, h_unpriced: int, h_ask: int, n_ask: float, n_noise: int, signal_var: float, draws: int) -> list:
        cu, ca = np.sqrt(h_unpriced / m) / 2, np.sqrt(h_ask / m) / 2
        f = lambda eta, c: 2 * integrate.quad(lambda x: x * x * norm.cdf(-eta - c * x) * norm.pdf(x), 0.0, np.inf)[0]
        g = lambda eta, c: 2 * integrate.quad(lambda x: norm.cdf(-eta - c * x) * norm.pdf(x), 0.0, np.inf)[0]
        A, Z = self.rng.standard_normal(draws), self.rng.standard_normal(draws)
        asked = np.abs(np.sqrt(signal_var) * A + Z) > np.sqrt(1.0 + signal_var) * norm.isf(n_ask / (2 * n_noise))
        acc = norm.cdf(-1.0 - ca * np.abs(Z)) * asked
        return [(n_noise * g(0.0, cu), n_noise * f(0.0, cu) / m), (n_noise * float(acc.mean()), n_noise * float((Z * Z * acc).mean()) / m)]

    def nothing_to_find(self, reps: int, P: int, m: int, coef: float, seed0: int, rows: int) -> None:
        beta = np.r_[np.full(5, coef), np.zeros(P - 5)]
        live = lambda keep, b: 1.0 + b[0] ** 2 + float(((np.bincount(np.asarray(keep, int), weights=b[1:], minlength=P) - beta) ** 2).sum())
        work, out = tempfile.mkdtemp(prefix="guarantee_"), []
        names = [f"x{j}" for j in range(P)]
        for rep in range(reps):
            rng = np.random.default_rng(seed0 + rep)
            Xs = rng.normal(size=(m, P))
            ys = Xs @ beta + rng.normal(size=m)
            Xh = rng.normal(size=(1000, P))
            yh = Xh @ beta + rng.normal(size=1000)
            order = self.rank(Xs, ys)
            k0 = self.cv_k(rng, Xs, ys, 30)
            start, rest = [int(j) for j in order[:k0]], [int(j) for j in order[k0:]]
            b0 = self.ols(Xs[:, start], ys)
            loss = lambda keep, b, X, y: (y - b[0] - X[:, keep] @ b[1:]) ** 2
            keep_u, b_u = list(start), b0
            for j in rest:
                b = self.ols(Xs[:, keep_u + [j]], ys)
                if loss(keep_u + [j], b, Xh, yh).mean() < loss(keep_u, b_u, Xh, yh).mean():
                    keep_u, b_u = keep_u + [j], b
            ho = pv.seal(pd.DataFrame(Xh, columns=names), yh, ct.Ledger(os.path.join(work, f"b{rep}"), budget=0.5), lineage="self", baseline=float(ys.mean()), baseline_lineage=ct.Lineage(), seed=rep)
            mk = lambda keep, b: (lambda Zf, keep=list(keep), b=b: b[0] + Zf.to_numpy()[:, keep] @ b[1:])
            ho.ask(mk(start, b0), lineage=ct.Lineage())
            keep_v, b_v, asked = list(start), b0, 0
            for j in rest:
                b = self.ols(Xs[:, keep_v + [j]], ys)
                try:
                    accepted = ho.ask(mk(keep_v + [j], b), lineage=ct.Lineage()).accepted
                except ValueError:
                    break
                asked += 1
                if accepted:
                    keep_v, b_v = keep_v + [j], b
            cands = [start + [j] for j in rest]
            fits = [self.ols(Xs[:, c], ys) for c in cands]
            d = np.column_stack([loss(start, b0, Xh, yh) - loss(c, b, Xh, yh) for c, b in zip(cands, fits)])[:rows]
            jg = self.paired_gate(d, self.z(self.delta / len(cands)))
            l0 = live(start, b0)
            out.append({"no look": l0, "no price": live(keep_u, b_u), "ask": live(keep_v, b_v), "gated": l0 if jg < 0 else live(cands[jg], fits[jg]), "best": min([l0] + [live(c, b) for c, b in zip(cands, fits)]),
                        "kept no price": sum(j >= 5 for j in keep_u[k0:]), "kept ask": sum(j >= 5 for j in keep_v[k0:]), "asked": asked, "noise left": sum(j >= 5 for j in rest), "holds all": all(j in start for j in range(5))})
        r = pd.DataFrame(out)
        print(f"B   nothing left to find: {P} features of which 5 carry {coef}, {m} training rows, 1,000 holdout rows; the default (cross-validation over the top-k fits) holds all five signal features in {int(r['holds all'].sum())} of {reps} worlds; seeds {seed0}..{seed0 + reps - 1}")
        rows_t = []
        for route in ("no look", "no price", "ask", "gated"):
            dlt = r[route] - r["no look"]
            rows_t.append([route, r[route].mean(), dlt.mean(), int((dlt > 1e-12).sum()), int((dlt < -1e-12).sum())])
        rows_t.append(["best candidate (default plus one feature)", r["best"].mean(), (r["best"] - r["no look"]).mean(), None, None])
        print(tabulate(rows_t, headers=["route", "mean live MSE", "live cost against no look", "worlds worse", "worlds better"], tablefmt="psql", floatfmt=".4f"))
        pred = self.harm(m, 1000, rows, float(r["asked"].mean()), int(r["noise left"].mean()), 5 * coef ** 2, 4_000_000)
        print(tabulate([["no price: eta 0, all 1,000 rows, every remaining feature", float(r["noise left"].mean()), pred[0][0], float(r["kept no price"].mean()), pred[0][1], float((r["no price"] - r["no look"]).mean())],
                        ["through ask: eta 1, 800 tuning rows, until the slice retires", float(r["asked"].mean()), pred[1][0], float(r["kept ask"].mean()), pred[1][1], float((r["ask"] - r["no look"]).mean())]],
                       headers=["ungated route", "noise features tried", "kept, predicted", "kept, simulated", "live cost, predicted", "live cost, simulated"], tablefmt="psql", floatfmt=".4f"))

    @staticmethod
    def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray: return np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]

    @staticmethod
    def rank(X: np.ndarray, y: np.ndarray) -> np.ndarray: return np.argsort(-np.abs((X - X.mean(0)).T @ (y - y.mean())))

    def cv_k(self, rng: np.random.Generator, X: np.ndarray, y: np.ndarray, K: int) -> int:
        sq = np.zeros(K)
        for fold in np.array_split(rng.permutation(len(y)), 5):
            rest = np.setdiff1d(np.arange(len(y)), fold)
            o = self.rank(X[rest], y[rest])
            for k in range(1, K + 1):
                b = self.ols(X[rest][:, o[:k]], y[rest])
                sq[k - 1] += ((y[fold] - b[0] - X[fold][:, o[:k]] @ b[1:]) ** 2).sum()
        return int(np.argmin(sq)) + 1

    def paired_gate(self, d: np.ndarray, zc: float) -> int:
        lcb = d.mean(0) - zc * d.std(0, ddof=1) / np.sqrt(len(d))
        return int(np.argmax(lcb)) if lcb.max() > 0 else -1

    def nested(self, tag: str, P: int, m: int, coef: float, seed0: int, reps: int, extra: bool, rows: int) -> None:
        beta = np.r_[np.full(5, coef), np.zeros(P - 5)]
        live = lambda keep, b: 1.0 + b[0] ** 2 + float(((np.bincount(np.asarray(keep, int), weights=b[1:], minlength=P) - beta) ** 2).sum())
        out = []
        for rep in range(reps):
            rng = np.random.default_rng(seed0 + rep)
            Xs = rng.normal(size=(m, P))
            ys = Xs @ beta + rng.normal(size=m)
            Xh = rng.normal(size=(1000, P))
            yh = Xh @ beta + rng.normal(size=1000)
            order = self.rank(Xs, ys)
            k0 = self.cv_k(rng, Xs, ys, 30)
            start = list(order[:k0])
            b0 = self.ols(Xs[:, start], ys)
            loss = lambda keep, b: (yh - b[0] - Xh[:, keep] @ b[1:]) ** 2
            if extra:
                cands = [start + [int(j)] for j in order[k0:]]
            else:
                cands = [list(order[:k]) for k in range(1, 31) if k != k0]
            fits = [self.ols(Xs[:, c], ys) for c in cands]
            d = np.column_stack([loss(start, b0) - loss(c, b) for c, b in zip(cands, fits)])[:rows]
            j = self.paired_gate(d, self.z(self.delta / len(cands)))
            l0 = live(start, b0)
            lj = l0 if j < 0 else live(cands[j], fits[j])
            best = min([l0] + [live(c, b) for c, b in zip(cands, fits)])
            out.append({"no look": l0, "gated": lj, "best": best, "moved": j >= 0})
        r = pd.DataFrame(out)
        g = r["no look"] - r["gated"]
        print(f"{tag}  gated look (each candidate against the default, Bonferroni at delta {self.delta}, on the first {rows} holdout rows), {reps} worlds, seeds {seed0}..{seed0 + reps - 1}; menu of {len(cands)}")
        print(tabulate([["no look (cross-validation)", r["no look"].mean(), 0, 0, 0], ["gated look", r["gated"].mean(), int((g > 1e-12).sum()), int((g < -1e-12).sum()), int(r["moved"].sum())], ["best candidate", r["best"].mean(), None, None, None]],
                       headers=["route", "mean live MSE", "worlds better", "worlds worse", "worlds the gate moved"], tablefmt="psql", floatfmt=".4f"))

    def changed_law(self, reps: int, rows: int) -> None:
        feats = [f"x{i}" for i in range(6)]
        specs = [("logit x0", ["x0"], None), ("logit x0 x1", ["x0", "x1"], None), ("logit all", feats, None), ("gbm depth 2", feats, 2), ("gbm depth 3", feats, 3), ("gbm depth 5", feats, 5)]
        model = lambda depth: LogisticRegression() if depth is None else GradientBoostingClassifier(max_depth=depth, random_state=0)
        world = lambda rng, n, b1: (lambda X: X.assign(y=((0.9 * X["x0"] + b1 * X["x1"] + 0.8 * X["x2"] * (X["x3"] > 0) + rng.normal(size=n)) > 0).astype(float)))(pd.DataFrame(rng.normal(size=(n, 6)), columns=feats))
        truth = world(np.random.default_rng(999), 200_000, 0.6)
        out = []
        for rep in range(reps):
            rng = np.random.default_rng(2000 + rep)
            train, hold = world(rng, 2000, -0.6), world(rng, 1000, 0.6)
            fits = {name: (model(depth).fit(train[cols], train["y"]), cols) for name, cols, depth in specs}
            proba = lambda name, X: fits[name][0].predict_proba(X[fits[name][1]])[:, 1]
            cv = {name: float(np.mean([((train.iloc[b]["y"].to_numpy() - model(depth).fit(train.iloc[a][cols], train.iloc[a]["y"]).predict_proba(train.iloc[b][cols])[:, 1]) ** 2).mean() for a, b in KFold(5, shuffle=True, random_state=rep).split(train)])) for name, cols, depth in specs}
            default = min(cv, key=cv.get)
            others = [n for n, _, _ in specs if n != default]
            yh = hold["y"].to_numpy()
            d = np.column_stack([(yh - proba(default, hold)) ** 2 - (yh - proba(n, hold)) ** 2 for n in others])[:rows]
            j = self.paired_gate(d, self.z(self.delta / len(others)))
            chosen = default if j < 0 else others[j]
            lv = {n: float(((truth["y"].to_numpy() - proba(n, truth)) ** 2).mean()) for n in fits}
            out.append({"no look": lv[default], "gated": lv[chosen], "best": min(lv.values()), "picked": chosen})
        r = pd.DataFrame(out)
        g = r["no look"] - r["gated"]
        print(f"C3  gated look under the changed law (x1 coefficient -0.6 in training, +0.6 in the holdout and live), on the first {rows} holdout rows, {reps} worlds, seeds 2000..{2000 + reps - 1}; menu of 5 against the default")
        print(tabulate([["no look (cross-validation)", r["no look"].mean(), 0, 0], ["gated look", r["gated"].mean(), int((g > 1e-12).sum()), int((g < -1e-12).sum())], ["best candidate", r["best"].mean(), None, None]], headers=["route", "mean live Brier", "worlds better", "worlds worse"], tablefmt="psql", floatfmt=".4f"))
        print("    gated picks:", r["picked"].value_counts().to_dict())

    def recency(self, reps: int, tau: float, blocks: int) -> None:
        rows, j = [], 1
        for rho in (0.0, 0.3, 0.6, 0.9, 1.0):
            for B in ((blocks,) if rho < 1 else (10, blocks)):
                for mh in (1, 4):
                    e = tau * self.rng.standard_normal(reps)
                    last_s = e if B == 1 else None
                    for b in range(2, B + mh + j + 1):
                        e = rho * e + (np.sqrt(1 - rho ** 2) if rho < 1 else 1.0) * tau * self.rng.standard_normal(reps)
                        if b == B:
                            last_s = e
                        if b == B + mh:
                            last_h = e
                    w_s, w_h = (rho ** (mh + j), rho ** j) if rho < 1 else (1.0, 1.0)
                    mc = float((e * (w_h * last_h > 0)).mean() - (e * (w_s * last_s > 0)).mean())
                    formula = tau * (rho ** j - rho ** (mh + j)) / np.sqrt(2 * np.pi) if rho < 1 else tau * (np.sqrt(B + mh) - np.sqrt(B)) / np.sqrt(2 * np.pi)
                    rows.append(["AR(1)" if rho < 1 else "random walk from 0", rho, B, mh, mc, formula, tau * np.sqrt(1 - rho ** (2 * j)) if rho < 1 else tau * np.sqrt(j)])
        print(f"D   recency premium of a look at the last M_H blocks for a decision {j} block after the holdout; block effect with sd {tau} (stationary AR(1)) or increment sd {tau} (random walk from 0), B training blocks, {reps:,} paths each")
        print(tabulate(rows, headers=["effect", "rho", "B", "M_H", "value of the look, simulated", "formula", "sd of the drift to the decision block"], tablefmt="psql", floatfmt=".4f"))

    def run(self) -> None:
        self.menu(10, 200_000, 0.2)
        self.nothing_to_find(60, 200, 400, 0.5, 5000, 800)
        self.nested("C2", 100, 400, 0.25, 3000, 60, False, 800)
        self.changed_law(40, 800)
        self.recency(400_000, 1.0, 50)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    v = GuaranteeValidation(0.05, 20261007)
    v.run()
