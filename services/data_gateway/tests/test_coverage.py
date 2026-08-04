import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from ashare_data_gateway import (
    SecurityLifecycle,
    UniverseMembership,
    audit_historical_coverage,
)
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[3]
START = date(2026, 7, 20)
DAYS = tuple(START + timedelta(days=index) for index in range(5))
SYMBOLS = ("000001.SZ", "600000.SH")


def memberships() -> tuple[UniverseMembership, ...]:
    return tuple(
        UniverseMembership(symbol, DAYS[0], DAYS[-1]) for symbol in SYMBOLS
    )


def lifecycles() -> tuple[SecurityLifecycle, ...]:
    return tuple(SecurityLifecycle(symbol, date(2020, 1, 1), None) for symbol in SYMBOLS)


def all_keys() -> frozenset[tuple[str, date]]:
    return frozenset((symbol, trade_date) for symbol in SYMBOLS for trade_date in DAYS)


def audit(
    *,
    bar_keys: frozenset[tuple[str, date]] | None = None,
    suspension_keys: frozenset[tuple[str, date]] = frozenset(),
    member_rows: tuple[UniverseMembership, ...] | None = None,
    lifecycle_rows: tuple[SecurityLifecycle, ...] | None = None,
    expected_member_count: int | None = 2,
    provenance_warnings: tuple[str, ...] = (),
):
    return audit_historical_coverage(
        audit_id="fixture-coverage-20260720",
        universe_policy_id="fixture-index",
        universe_policy_version="v1",
        window_start=DAYS[0],
        window_end=DAYS[-1],
        generated_at=datetime(2026, 7, 25, 8, tzinfo=UTC),
        trading_days=DAYS,
        memberships=member_rows or memberships(),
        lifecycles=lifecycle_rows or lifecycles(),
        bar_keys=all_keys() if bar_keys is None else bar_keys,
        suspension_keys=suspension_keys,
        expected_member_count=expected_member_count,
        provenance_warnings=provenance_warnings,
    )


def test_complete_member_day_coverage_passes() -> None:
    result = audit()

    assert result.passed is True
    assert result.expected_member_days == 10
    assert result.bar_member_days == 10
    assert result.expected_delisted_member_day_count == 0
    assert result.expected_delisted_member_days == ()
    assert result.missing_member_days == ()
    assert result.silent_skip_symbols == ()
    assert result.document["schema_version"] == "2.0.0"


def test_missing_member_days_fail_without_silent_skip() -> None:
    bars = frozenset(
        key
        for key in all_keys()
        if key != ("000001.SZ", DAYS[-1])
    )
    result = audit(bar_keys=bars)

    assert result.passed is False
    assert result.reason_codes == ("MISSING_MEMBER_DAYS",)
    assert result.missing_member_days[0].symbol == "000001.SZ"
    assert result.silent_skip_symbols == ()


def test_zero_observation_member_is_reported_as_silent_skip() -> None:
    bars = frozenset(key for key in all_keys() if key[0] == "600000.SH")
    result = audit(bar_keys=bars)

    assert result.passed is False
    assert "SILENT_MEMBER_SKIP" in result.reason_codes
    assert result.silent_skip_symbols == ("000001.SZ",)


def test_declared_suspensions_cover_missing_bars() -> None:
    suspended = frozenset(("000001.SZ", trade_date) for trade_date in DAYS)
    bars = frozenset(key for key in all_keys() if key[0] == "600000.SH")
    result = audit(bar_keys=bars, suspension_keys=suspended)

    assert result.passed is True
    assert result.suspended_member_days == len(DAYS)
    assert result.silent_skip_symbols == ()


def test_delisted_member_expectation_stops_on_delist_date() -> None:
    lifecycle_rows = (
        SecurityLifecycle("000001.SZ", date(2020, 1, 1), DAYS[2]),
        SecurityLifecycle("600000.SH", date(2020, 1, 1), None),
    )
    bars = frozenset(
        [("000001.SZ", trade_date) for trade_date in DAYS[:3]]
        + [("600000.SH", trade_date) for trade_date in DAYS]
    )
    result = audit(
        bar_keys=bars,
        lifecycle_rows=lifecycle_rows,
        expected_member_count=None,
    )

    assert result.passed is True
    assert result.expected_member_days == 8
    assert result.expected_delisted_member_day_count == 0
    assert result.delisted_symbols_in_scope == ("000001.SZ",)


def test_uncovered_delist_date_is_classified_once_and_passes() -> None:
    lifecycle_rows = (
        SecurityLifecycle("000001.SZ", date(2020, 1, 1), DAYS[2]),
        SecurityLifecycle("600000.SH", date(2020, 1, 1), None),
    )
    bars = frozenset(
        [("000001.SZ", trade_date) for trade_date in DAYS[:2]]
        + [("600000.SH", trade_date) for trade_date in DAYS]
    )

    result = audit(
        bar_keys=bars,
        lifecycle_rows=lifecycle_rows,
        expected_member_count=None,
    )

    assert result.passed is True
    assert result.expected_member_days == 8
    assert result.bar_member_days == 7
    assert result.suspended_member_days == 0
    assert result.expected_delisted_member_day_count == 1
    assert [
        (member_day.symbol, member_day.trade_date)
        for member_day in result.expected_delisted_member_days
    ] == [("000001.SZ", DAYS[2])]
    assert result.missing_member_days == ()
    assert result.silent_skip_symbols == ()
    assert (
        result.bar_member_days
        + result.suspended_member_days
        + result.expected_delisted_member_day_count
        + len(result.missing_member_days)
        == result.expected_member_days
    )


def test_bar_precedes_suspension_and_expected_delist_on_same_day() -> None:
    lifecycle_rows = (
        SecurityLifecycle("000001.SZ", date(2020, 1, 1), DAYS[2]),
        SecurityLifecycle("600000.SH", date(2020, 1, 1), None),
    )
    bars = frozenset(
        [("000001.SZ", trade_date) for trade_date in DAYS[:3]]
        + [("600000.SH", trade_date) for trade_date in DAYS]
    )

    result = audit(
        bar_keys=bars,
        suspension_keys=frozenset({("000001.SZ", DAYS[2])}),
        lifecycle_rows=lifecycle_rows,
        expected_member_count=None,
    )

    assert result.passed is True
    assert result.bar_member_days == 8
    assert result.suspended_member_days == 0
    assert result.expected_delisted_member_day_count == 0


def test_filtered_suspension_precedes_expected_delist_on_same_day() -> None:
    lifecycle_rows = (
        SecurityLifecycle("000001.SZ", date(2020, 1, 1), DAYS[2]),
        SecurityLifecycle("600000.SH", date(2020, 1, 1), None),
    )
    bars = frozenset(
        [("000001.SZ", trade_date) for trade_date in DAYS[:2]]
        + [("600000.SH", trade_date) for trade_date in DAYS]
    )

    result = audit(
        bar_keys=bars,
        suspension_keys=frozenset({("000001.SZ", DAYS[2])}),
        lifecycle_rows=lifecycle_rows,
        expected_member_count=None,
    )

    assert result.passed is True
    assert result.bar_member_days == 7
    assert result.suspended_member_days == 1
    assert result.expected_delisted_member_day_count == 0


def test_point_in_time_member_count_anomaly_fails() -> None:
    member_rows = (
        UniverseMembership("000001.SZ", DAYS[0], DAYS[-1]),
        UniverseMembership("600000.SH", DAYS[2], DAYS[-1]),
    )
    result = audit(member_rows=member_rows)

    assert result.passed is False
    assert "MEMBER_COUNT_ANOMALY" in result.reason_codes
    assert [item.trade_date for item in result.member_count_anomalies] == list(DAYS[:2])


def test_coverage_document_is_deterministic_under_input_order() -> None:
    first = audit(
        provenance_warnings=(
            "NON_OFFICIAL_VENDOR_ENDPOINT",
            "NON_OFFICIAL_VENDOR_ENDPOINT",
        )
    )
    second = audit(
        member_rows=tuple(reversed(memberships())),
        lifecycle_rows=tuple(reversed(lifecycles())),
        provenance_warnings=("NON_OFFICIAL_VENDOR_ENDPOINT",),
    )

    assert first.document == second.document
    assert first.provenance_warnings == ("NON_OFFICIAL_VENDOR_ENDPOINT",)
    schema = json.loads(
        (ROOT / "contracts/schemas/coverage-audit-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    ).validate(first.document)


def test_invalid_provenance_warning_is_rejected() -> None:
    with pytest.raises(ValueError, match="uppercase reason codes"):
        audit(provenance_warnings=("non-official",))
