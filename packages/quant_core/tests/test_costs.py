import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from ashare_quant_core import calculate_cost, select_cost_segment

ROOT = Path(__file__).resolve().parents[3]


def load_model() -> dict[str, object]:
    return json.loads(
        (ROOT / "contracts/examples/cost-model.example.json").read_text(encoding="utf-8")
    )


def test_cost_golden_cases() -> None:
    fixture_path = ROOT / "contracts" / "golden_fixtures" / "cost-cases.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    model_path = (fixture_path.parent / fixture["model"]).resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))

    for case in fixture["cases"]:
        result = calculate_cost(
            side=case["side"],
            trade_date=date.fromisoformat(case["trade_date"]),
            market=case["market"],
            gross_amount=Decimal(str(case["gross_amount"])),
            model=model,
        )
        assert result.commission == Decimal(str(case["expected_commission"]))
        assert result.stamp_duty == Decimal(str(case["expected_stamp_duty"]))
        assert result.transfer_fee == Decimal(str(case["expected_transfer_fee"]))
        assert result.total == Decimal(str(case["expected_total_cost"]))


def test_cost_rejects_negative_amount() -> None:
    with pytest.raises(ValueError, match="gross_amount must be non-negative"):
        calculate_cost(
            side="buy",
            trade_date=date(2026, 7, 29),
            market="SH",
            gross_amount=Decimal("-1"),
            model=load_model(),
        )


def test_zero_amount_has_zero_cost() -> None:
    result = calculate_cost(
        side="buy",
        trade_date=date(2026, 7, 29),
        market="SH",
        gross_amount=Decimal("0"),
        model=load_model(),
    )
    assert result.total == Decimal("0.00")


@pytest.mark.parametrize(
    ("trade_date", "market"),
    [
        (date(2022, 4, 28), "SH"),
        (date(2026, 7, 29), "BJ"),
    ],
)
def test_cost_fails_when_date_or_market_is_unsupported(
    trade_date: date,
    market: str,
) -> None:
    with pytest.raises(ValueError, match="match exactly one segment"):
        calculate_cost(
            side="buy",
            trade_date=trade_date,
            market=market,
            gross_amount=Decimal("100000"),
            model=load_model(),
        )


def test_cost_fails_when_segments_overlap() -> None:
    model = load_model()
    segments = model["segments"]
    assert isinstance(segments, list)
    overlapping = dict(segments[0])
    overlapping["segment_id"] = "overlapping-segment"
    overlapping["effective_from"] = "2023-08-27"
    overlapping["effective_to"] = None
    segments.append(overlapping)

    with pytest.raises(ValueError, match="matched=2"):
        select_cost_segment(
            trade_date=date(2023, 8, 27),
            market="SH",
            model=model,
        )


def test_high_turnover_accumulates_all_cost_components() -> None:
    model = load_model()
    gross_amount = Decimal("100000")
    round_trips = 100

    buy = calculate_cost(
        side="buy",
        trade_date=date(2026, 7, 29),
        market="SH",
        gross_amount=gross_amount,
        model=model,
    )
    sell = calculate_cost(
        side="sell",
        trade_date=date(2026, 7, 29),
        market="SH",
        gross_amount=gross_amount,
        model=model,
    )

    assert (buy.total + sell.total) * round_trips == Decimal("11200.00")
