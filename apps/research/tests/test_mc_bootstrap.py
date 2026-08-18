import math

import pytest
from ashare_research_app.mc_bootstrap import block_bootstrap


def test_prob_profit_is_a_fraction() -> None:
    result = block_bootstrap([0.001] * 100, iters=200, block=5, seed=1)
    assert 0.0 <= result.prob_profit <= 1.0


def test_reproducible_with_same_seed() -> None:
    returns = [0.001 + 0.01 * math.sin(i * 0.5) for i in range(100)]
    a = block_bootstrap(returns, iters=200, block=5, seed=42)
    b = block_bootstrap(returns, iters=200, block=5, seed=42)
    assert a == b


def test_positive_returns_always_profit() -> None:
    result = block_bootstrap([0.01] * 100, iters=200, block=5, seed=1)
    assert result.prob_profit == 1.0


def test_negative_returns_never_profit() -> None:
    result = block_bootstrap([-0.01] * 100, iters=200, block=5, seed=1)
    assert result.prob_profit == 0.0


def test_orig_final_is_compounded() -> None:
    result = block_bootstrap([0.01, 0.02], iters=10, block=1, seed=1, start=100.0)
    assert result.orig_final == pytest.approx(100.0 * 1.01 * 1.02)


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        block_bootstrap([])


def test_invalid_block_raises() -> None:
    with pytest.raises(ValueError):
        block_bootstrap([0.01] * 50, block=0)


def test_describe_mentions_prob_profit() -> None:
    result = block_bootstrap([0.001] * 100, iters=100, block=5, seed=1)
    assert "prob(profit)" in result.describe()
