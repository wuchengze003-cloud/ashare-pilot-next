import math

import pytest
from ashare_research_app.performance import annualized_sharpe, returns_from_nav_curve


def test_returns_and_sharpe_use_the_costed_nav_curve() -> None:
    nav_curve = (
        {"nav": 1.0},
        {"nav": 1.01},
        {"nav": 1.01 * 0.995},
        {"nav": 1.01 * 0.995 * 1.02},
    )

    returns = returns_from_nav_curve(nav_curve)

    assert returns == pytest.approx((0.01, -0.005, 0.02))
    expected_mean = sum(returns) / len(returns)
    expected_variance = sum((value - expected_mean) ** 2 for value in returns) / 2
    assert annualized_sharpe(returns) == pytest.approx(
        expected_mean / math.sqrt(expected_variance) * math.sqrt(252)
    )


@pytest.mark.parametrize("bad_nav", (0.0, -1.0, math.inf, math.nan))
def test_returns_reject_invalid_nav_values(bad_nav: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        returns_from_nav_curve(({"nav": 1.0}, {"nav": bad_nav}))


def test_sharpe_rejects_non_finite_returns() -> None:
    with pytest.raises(ValueError, match="finite"):
        annualized_sharpe((0.01, math.nan))

