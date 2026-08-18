"""Deflated Sharpe Ratio (Bailey & López de Prado).

Answers: "given that I searched N trials and picked the best, how much of this
Sharpe is luck?" The more trials you ran and the more their scores spread, the
higher the Sharpe pure luck could produce (SR0). The observed Sharpe must clear
that luck ceiling for the DSR to approach 1. DSR < 0.95 is treated as
non-significant.

Self-built equivalent of the reference project's ``deflated_sharpe.py``, adapted
to A-shares: annualization defaults to 252 trading days (the reference used 365
for 24/7 crypto).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class DeflatedSharpeResult:
    n_trials: int
    observed_sharpe: float
    expected_max_sharpe: float  # luck ceiling (annualized)
    deflated_sharpe: float
    significant: bool

    def describe(self) -> str:
        verdict = "significant" if self.significant else "not significant (likely luck)"
        return (
            f"DSR={self.deflated_sharpe:.3f} "
            f"(observed {self.observed_sharpe:.3f} vs luck ceiling "
            f"{self.expected_max_sharpe:.3f}, {self.n_trials} trials) — {verdict}"
        )


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via bisection (accurate enough for DSR)."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _skew(x: np.ndarray) -> float:
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma == 0.0:
        return 0.0
    return float(np.mean((x - mu) ** 3) / sigma**3)


def _kurtosis(x: np.ndarray) -> float:
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma == 0.0:
        return 0.0
    return float(np.mean((x - mu) ** 4) / sigma**4)


def deflated_sharpe(
    daily_returns: Sequence[float],
    *,
    n_trials: int,
    trials_sharpe: Sequence[float] | None = None,
    annualization: int = 252,
    significance: float = 0.95,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio for a backtest's daily returns.

    Args:
        daily_returns: daily PnL returns of the chosen strategy.
        n_trials: number of trials searched (used to compute the luck ceiling).
        trials_sharpe: Sharpe of every trial (for the trial-variance estimate).
            When omitted or fewer than two, the luck ceiling is treated as zero.
        annualization: trading days per year (A-shares: 252).
        significance: DSR threshold for the ``significant`` flag.
    """
    r = np.asarray(list(daily_returns), dtype=float)
    if r.size < 30:
        raise ValueError("daily returns sample is too short (need >= 30 days)")
    if not np.all(np.isfinite(r)):
        raise ValueError("daily returns must be finite")
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if r.std() == 0.0:
        raise ValueError("daily returns have zero variance")

    if trials_sharpe is None:
        sr_var = 0.0
    else:
        trials = np.asarray(list(trials_sharpe), dtype=float)
        sr_var = float(np.var(trials, ddof=1)) if trials.size >= 2 else 0.0

    t_days = r.size
    sr_daily = float(r.mean() / r.std())
    sr_annual = sr_daily * math.sqrt(annualization)

    g3 = _skew(r)
    g4 = _kurtosis(r)

    if n_trials <= 1 or sr_var <= 0.0:
        sr0_daily = 0.0  # nothing to "pick the best" from — no luck correction
    else:
        n = float(n_trials)
        e = math.e
        sr0_daily = math.sqrt(sr_var / annualization) * (
            (1.0 - _EULER_GAMMA) * _norm_ppf(1.0 - 1.0 / n)
            + _EULER_GAMMA * _norm_ppf(1.0 - 1.0 / (n * e))
        )

    denom = 1.0 - g3 * sr_daily + (g4 - 1.0) / 4.0 * sr_daily**2
    if denom <= 0.0:
        raise ValueError("DSR denominator is non-positive (extreme return distribution)")
    z = (sr_daily - sr0_daily) * math.sqrt(t_days - 1.0) / math.sqrt(denom)
    dsr = _norm_cdf(z)

    return DeflatedSharpeResult(
        n_trials=n_trials,
        observed_sharpe=sr_annual,
        expected_max_sharpe=sr0_daily * math.sqrt(annualization),
        deflated_sharpe=dsr,
        significant=dsr >= significance,
    )
