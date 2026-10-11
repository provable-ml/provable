"""The rates register.

Quantities never cross currency word families; rates are the only
cross-currency objects, and they say so in their name. Each rate carries
its exactness grade, printed wherever the rate is quoted.

  row_equivalent      columns -> rows   exact in the acquisition model
  rows_for_level      validity -> rows  identity at fixed effect and k
  rows_for_bits       validity -> rows  identity at fixed effect (one-sided bar)
  anchor_to_promise   money -> mass     DECLARED (user-supplied bridge)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from scipy.stats import norm


@dataclass(frozen=True)
class Rate:
    name: str
    converts: str
    grade: str

    def line(self) -> str:
        return f"[rate] {self.name}: {self.converts} ({self.grade})"


ROW_EQUIVALENT = Rate("row_equivalent", "columns -> rows", "exact in the acquisition model (currency law)")
ROWS_FOR_LEVEL = Rate("rows_for_level", "validity -> rows", "identity at fixed effect size and fixed rejection count")
ROWS_FOR_BITS = Rate("rows_for_bits", "validity -> rows", "identity at fixed effect size, one-sided Gaussian bar")
ANCHOR_TO_PROMISE = Rate("anchor_to_promise", "declared money -> required mass", "DECLARED: user-supplied bridge, no theory content")


def row_equivalent(n: int, lam: float, tau_v: float) -> float:
    """Effective extra rows bought by a column of reliability lam against
    visibility share tau_v: n_eff - n = n * lam^2 / (tau_v (1 - lam^2)).
    Exact in the acquisition model (the currency law)."""
    lam2 = float(lam) ** 2
    return float(n) * lam2 / (float(tau_v) * max(1.0 - lam2, 1e-300))


def _bar(q: float, k: int, m: int) -> float:
    return float(norm.isf(max(min(q * max(k, 1) / (2 * m), 0.5 - 1e-12), 1e-300)))


def rows_for_level(n: int, q_from: float, q_to: float, k: int, m: int) -> float:
    """Rows needed to hold a fixed effect at the tighter bar when the
    level moves from q_from to q_to at fixed rejection count k and family
    size m: n' = n * (z(q_to)/z(q_from))^2, z(q) = Phibar^{-1}(q k / 2m).
    An identity at fixed effect and fixed k (t scales as sqrt(n)); the
    realized k may move with the level, which this rate does not model."""
    z0, z1 = _bar(q_from, k, m), _bar(q_to, k, m)
    return float(n) * (z1 / z0) ** 2


def rows_for_bits(n: int, bits: float, delta: float = 0.05) -> float:
    """Rows a holdout that has spent L bits needs to certify a fixed effect
    with the width of a clean holdout of n rows: n' = n (z_L / z_0)^2 =
    n / s(L), z_L = Phibar^{-1}(delta 2^-L) one-sided. Read the other way,
    n rows after L bits certify like n s(L) clean rows. An identity at
    fixed effect (the width scales as z / sqrt(n))."""
    from .transcript import z_bar
    return float(n) * (z_bar(delta, bits) / z_bar(delta, 0.0)) ** 2


def anchor_to_promise(anchor: float, bridge: Optional[Callable[[float], float]] = None) -> float:
    """Money enters only as a declared input. Converting a cost anchor to
    a required-mass promise needs a user-declared bridge; there is no
    theory content here and the register refuses without one."""
    if bridge is None:
        raise ValueError("anchor_to_promise is DECLARED: supply bridge, a callable mapping the declared anchor to the required mass; the register carries no theory for this conversion")
    return float(bridge(float(anchor)))


REGISTER = {r.name: r for r in (ROW_EQUIVALENT, ROWS_FOR_LEVEL, ROWS_FOR_BITS, ANCHOR_TO_PROMISE)}
