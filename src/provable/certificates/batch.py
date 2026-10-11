"""The online batch budget (the batch step-up rule).

Activates the certificate-algebra paper's online budget theorem (Thm 6.5,
"Online budget, imported"): batch levels produced by the batch step-up
update rule of Zrnic, Jiang, Ramdas, and Jordan, "The power of batching
in multiple hypothesis testing", AISTATS 2020 (PMLR 108:3806-3815), at
target q, give E[FDP(union of R_s, s <= t)] <= q for every t, provided
each batch satisfies the fresh-lineage condition (Lem 6.4: conditional
on its past, exactly uniform independent nulls) and the schedule is
monotone. Single-member batches recover alpha-investing and the
generalized alpha-investing families.

Transcribed objects (Zrnic20):
  eq (1)        R+_t = max_i |A_t(P_t with P_{t,i} set to 0)|
  Definition 1  FDPhat(t) = sum_s alpha_s R+_s / (R+_s + sum_{r<=t, r!=s} R_r) <= alpha
  Algorithm 1   alpha_1 = gamma_1 alpha; after batch t:
                beta_{t+1} = the Definition-1 sum over s <= t,
                alpha_{t+1} = (sum_{s<=t+1} gamma_s alpha - beta_{t+1})
                              * (n_{t+1} + sum_{s<=t} R_s) / n_{t+1}
  Theorem 1     independent nulls + monotone schedule => anytime FDR <= alpha

The batch gate here is the BH step-up (the kit's khat at the batch
level). Nothing is invented: deviations from the prescribed schedule are
detected and reported, and a deviating book falls back to the additive
bound (Prop 6.1), which always holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .kit import khat

SOURCE = "M2 Thm 6.5; Zrnic20 Def 1, Alg 1, Thm 1"


def bh_rejections(p: Sequence[float], level: float) -> int:
    """R_t: the BH step-up rejection count of one batch at its level."""
    return int(khat(np.asarray(p, float), float(level)))


def r_plus(p: Sequence[float], level: float) -> int:
    """Zrnic20 eq (1): the maximum augmented rejection count when one
    p-value is set to zero, level held fixed."""
    p = np.asarray(p, float)
    if p.size == 0:
        return 0
    best = 0
    for i in range(p.size):
        q2 = p.copy()
        q2[i] = 0.0
        best = max(best, int(khat(q2, float(level))))
    return best


def default_gamma(s: int) -> float:
    """The paper's conservative investment sequence gamma_s proportional
    to s^-2, normalized (sum over s >= 1 equals one)."""
    return (6.0 / np.pi ** 2) / (s * s)


@dataclass
class BatchRecord:
    n: int
    level: float
    R: int
    R_plus: int


@dataclass
class BatchSchedule:
    """The Algorithm 1 account at target q. Quote next_level before a
    batch runs; record the batch after. conforms() checks a recorded
    book against the prescribed schedule."""
    q: float
    gamma: Optional[Sequence[float]] = None
    batches: List[BatchRecord] = field(default_factory=list)

    def _gamma(self, s: int) -> float:
        if self.gamma is not None:
            return float(self.gamma[s - 1]) if s - 1 < len(self.gamma) else 0.0
        return default_gamma(s)

    def _beta(self) -> float:
        t = len(self.batches)
        total_R = sum(b.R for b in self.batches)
        out = 0.0
        for b in self.batches:
            out += b.level * b.R_plus / (b.R_plus + (total_R - b.R)) if b.R_plus > 0 else 0.0
        return out

    def fdp_hat(self) -> float:
        """Definition 1's estimator over the recorded batches."""
        return self._beta()

    def next_level(self, n_next: int) -> float:
        """Algorithm 1's level for the coming batch of size n_next."""
        t = len(self.batches)
        if t == 0:
            return self._gamma(1) * self.q
        invested = sum(self._gamma(s) for s in range(1, t + 2)) * self.q
        total_R = sum(b.R for b in self.batches)
        lvl = (invested - self._beta()) * (n_next + total_R) / n_next
        return max(0.0, float(lvl))

    def record(self, p: Sequence[float], level: float) -> BatchRecord:
        p = np.asarray(p, float)
        rec = BatchRecord(n=int(p.size), level=float(level), R=bh_rejections(p, level), R_plus=r_plus(p, level))
        self.batches.append(rec)
        return rec

    def conforms(self, tol: float = 1e-9) -> bool:
        """True iff every recorded level equals the level Algorithm 1
        prescribes given the earlier batches."""
        probe = BatchSchedule(q=self.q, gamma=self.gamma)
        for b in self.batches:
            if abs(b.level - probe.next_level(b.n)) > tol:
                return False
            probe.batches.append(b)
        return True
