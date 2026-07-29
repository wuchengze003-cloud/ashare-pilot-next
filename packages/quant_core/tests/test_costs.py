import json
from decimal import Decimal
from pathlib import Path

from ashare_quant_core import calculate_cost

ROOT = Path(__file__).resolve().parents[3]


def test_cost_golden_cases() -> None:
    fixture_path = ROOT / "contracts" / "golden_fixtures" / "cost-cases.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    model_path = (fixture_path.parent / fixture["model"]).resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))

    for case in fixture["cases"]:
        result = calculate_cost(
            side=case["side"],
            gross_amount=Decimal(str(case["gross_amount"])),
            model=model,
        )
        assert result.commission == Decimal(str(case["expected_commission"]))
        assert result.stamp_duty == Decimal(str(case["expected_stamp_duty"]))
        assert result.transfer_fee == Decimal(str(case["expected_transfer_fee"]))
        assert result.total == Decimal(str(case["expected_total_cost"]))


def test_cost_rejects_negative_amount() -> None:
    model = {
        "commission_rate": 0,
        "minimum_commission": 0,
        "stamp_duty_rate_sell": 0,
        "transfer_fee_rate": 0,
        "rounding": {"scale": 2, "mode": "ROUND_HALF_UP", "component_level": True},
    }
    try:
        calculate_cost(side="buy", gross_amount=Decimal("-1"), model=model)
    except ValueError as exc:
        assert str(exc) == "gross_amount must be non-negative"
    else:
        raise AssertionError("negative gross amount must fail")


def test_zero_amount_has_zero_cost() -> None:
    model = {
        "commission_rate": 0.0003,
        "minimum_commission": 5,
        "stamp_duty_rate_sell": 0.0005,
        "transfer_fee_rate": 0.00001,
        "rounding": {"scale": 2, "mode": "ROUND_HALF_UP", "component_level": True},
    }
    result = calculate_cost(side="buy", gross_amount=Decimal("0"), model=model)
    assert result.total == Decimal("0.00")
