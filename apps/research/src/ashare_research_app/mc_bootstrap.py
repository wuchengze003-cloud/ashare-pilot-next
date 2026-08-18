"""Monte Carlo block bootstrap for prob(profit).

Resamples a daily-return series in contiguous blocks (preserving some serial
correlation) and answers: how much of this backtest curve is luck? It outputs
the probability of profit and the 5/50/95 percentile final values, so a single
backtest number is replaced by a distribution.

Self-built equivalent of the reference project's ``mc_bootstrap.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    n_returns: int
    iters: int
    block: int
    prob_profit: float
    final_p5: float
    final_p50: float
    final_p95: float
    orig_final: float

    def describe(self) -> str:
        return (
            f"prob(profit)={self.prob_profit:.1%} "
            f"(P5/P50/P95 = {self.final_p5:.3f}/{self.final_p50:.3f}/{self.final_p95:.3f}, "
            f"actual {self.orig_final:.3f})"
        )


def block_bootstrap(
    returns: Sequence[float],
    *,
    iters: int = 5000,
    block: int = 10,
    seed: int = 7,
    start: float = 1.0,
) -> BootstrapResult:
    """Resample returns in blocks and report the distribution of final values.

    Args:
        returns: daily returns of the strategy (relative, e.g. 0.01 = +1%).
        iters: number of resampled paths.
        block: contiguous block length (retains serial correlation).
        seed: RNG seed for reproducibility.
        start: starting equity (final values are reported on this scale).
    """
    r = np.asarray(list(returns), dtype=float)
    if r.size == 0:
        raise ValueError("returns cannot be empty")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite")
    if block < 1:
        raise ValueError("block must be positive")
    if iters < 1:
        raise ValueError("iters must be positive")

    rng = np.random.default_rng(seed)
    n = r.size
    finals = np.empty(iters, dtype=float)
    for _ in range(iters):
        idx: list[int] = []
        while len(idx) < n:
            s = int(rng.integers(0, max(1, n - block + 1)))
            idx.extend(range(s, min(s + block, n)))
        sampled = np.asarray(idx[:n])
        eq = start * np.cumprod(1.0 + r[sampled])
        finals[_] = eq[-1]

    p5, p50, p95 = (float(v) for v in np.percentile(finals, [5, 50, 95]))
    orig_final = start * float(np.prod(1.0 + r))
    return BootstrapResult(
        n_returns=n,
        iters=iters,
        block=block,
        prob_profit=float((finals > start).mean()),
        final_p5=p5,
        final_p50=p50,
        final_p95=p95,
        orig_final=orig_final,
    )
