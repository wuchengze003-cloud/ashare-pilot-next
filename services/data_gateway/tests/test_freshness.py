import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from ashare_data_gateway import assess_dataset_freshness
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[3]
TRADING_DAYS = (
    date(2026, 7, 27),
    date(2026, 7, 28),
    date(2026, 7, 29),
    date(2026, 7, 30),
)


def test_current_dataset_passes_freshness() -> None:
    result = assess_dataset_freshness(
        expected_as_of=date(2026, 7, 30),
        latest_complete_date=date(2026, 7, 30),
        trading_days=TRADING_DAYS,
    )

    assert result.passed is True
    assert result.lag_sessions == 0
    assert result.reason_codes == ()


def test_stale_dataset_fails_by_completed_trading_sessions() -> None:
    result = assess_dataset_freshness(
        expected_as_of=date(2026, 7, 30),
        latest_complete_date=date(2026, 7, 28),
        trading_days=TRADING_DAYS,
        allowed_lag_sessions=1,
    )

    assert result.passed is False
    assert result.lag_sessions == 2
    assert result.reason_codes == ("DATA_STALE",)


def test_unknown_or_future_dates_fail_closed() -> None:
    unknown = assess_dataset_freshness(
        expected_as_of=date(2026, 7, 31),
        latest_complete_date=date(2026, 7, 30),
        trading_days=TRADING_DAYS,
    )
    future = assess_dataset_freshness(
        expected_as_of=date(2026, 7, 29),
        latest_complete_date=date(2026, 7, 30),
        trading_days=TRADING_DAYS,
    )

    assert unknown.passed is False
    assert "EXPECTED_AS_OF_NOT_TRADING_DAY" in unknown.reason_codes
    assert future.passed is False
    assert "LATEST_COMPLETE_IN_FUTURE" in future.reason_codes


def test_freshness_emits_valid_data_stage_health() -> None:
    result = assess_dataset_freshness(
        expected_as_of=date(2026, 7, 30),
        latest_complete_date=date(2026, 7, 29),
        trading_days=TRADING_DAYS,
    )
    document = result.stage_health_document(
        run_id="fixture-data-health",
        started_at=datetime(2026, 7, 30, 8, tzinfo=UTC),
        finished_at=datetime(2026, 7, 30, 8, tzinfo=UTC)
        + timedelta(seconds=1),
        dataset_manifest_sha256="a" * 64,
    )
    schema = json.loads(
        (ROOT / "contracts/schemas/stage-health.schema.json").read_text(
            encoding="utf-8"
        )
    )

    Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    ).validate(document)
    assert document["status"] == "fail"
    assert document["reason_codes"] == ["DATA_STALE"]
