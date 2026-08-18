"""Pin the frozen champion configuration used by the rolling GBDT pilot."""

from ashare_research_app import champion


def test_hybrid_label_config_is_frozen() -> None:
    assert champion.LABEL_HORIZON == 60
    assert champion.TAIL_LABEL == 20


def test_rolling_champion_config_is_frozen() -> None:
    assert champion.WINDOW == 120
    assert champion.REFIT_EVERY == 20
    assert champion.TOP_K == 10
    assert champion.EXIT_BUFFER == 4
    assert champion.MIN_HOLD == 20
    assert champion.MAX_BUY == 2
    assert champion.MAX_SELL == 2
    assert champion.STOP_LOSS == 0.06
    assert champion.TAKE_PROFIT == 0.25
    assert champion.TIMING_BAND == 0.03


def test_feature_pool_size_is_frozen() -> None:
    assert len(champion.FEATURES) == 27
    assert len(set(champion.FEATURES)) == 27


def test_walk_forward_segments_are_frozen() -> None:
    assert champion.SEGMENTS == [
        ("train", "2023-01-01", "2024-06-30"),
        ("valid", "2024-07-01", "2025-06-30"),
        ("test", "2025-07-01", "2026-08-31"),
    ]
