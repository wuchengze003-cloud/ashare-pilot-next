import math

import pytest
from ashare_research_app.deflated_sharpe import deflated_sharpe


def _high_sharpe_returns() -> list[float]:
    return [0.001 + 0.01 * math.sin(i * 0.5) for i in range(200)]


def _zero_sharpe_returns() -> list[float]:
    return [0.01 * math.sin(i * 0.5) for i in range(200)]


def test_result_is_a_probability() -> None:
    result = deflated_sharpe(_high_sharpe_returns(), n_trials=10)
    assert 0.0 <= result.deflated_sharpe <= 1.0


def test_high_sharpe_single_trial_is_significant() -> None:
    result = deflated_sharpe(_high_sharpe_returns(), n_trials=1)
    assert result.significant
    assert result.expected_max_sharpe == 0.0


def test_zero_sharpe_many_trials_is_not_significant() -> None:
    trials = [0.3 + 0.01 * i for i in range(100)]
    result = deflated_sharpe(
        _zero_sharpe_returns(), n_trials=100, trials_sharpe=trials
    )
    assert not result.significant
    assert result.expected_max_sharpe > 0.0


def test_luck_ceiling_grows_with_trials() -> None:
    trials = [0.5 + 0.01 * i for i in range(500)]
    small = deflated_sharpe(_high_sharpe_returns(), n_trials=10, trials_sharpe=trials[:10])
    large = deflated_sharpe(_high_sharpe_returns(), n_trials=500, trials_sharpe=trials)
    assert large.expected_max_sharpe > small.expected_max_sharpe


def test_short_sample_raises() -> None:
    with pytest.raises(ValueError):
        deflated_sharpe([0.01] * 10, n_trials=1)


def test_zero_variance_raises() -> None:
    with pytest.raises(ValueError):
        deflated_sharpe([0.0] * 100, n_trials=1)


def test_non_positive_trials_raises() -> None:
    with pytest.raises(ValueError):
        deflated_sharpe(_high_sharpe_returns(), n_trials=0)


def test_describe_mentions_trials() -> None:
    result = deflated_sharpe(_high_sharpe_returns(), n_trials=7)
    assert "7 trials" in result.describe()
