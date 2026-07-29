"""Deterministic transaction-cost calculation."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


@dataclass(frozen=True)
class CostBreakdown:
    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal

    @property
    def total(self) -> Decimal:
        return self.commission + self.stamp_duty + self.transfer_fee


def calculate_cost(
    *,
    side: Literal["buy", "sell"],
    gross_amount: Decimal,
    model: Mapping[str, object],
) -> CostBreakdown:
    """Calculate costs using the contract's component-level rounding."""
    if gross_amount < 0:
        raise ValueError("gross_amount must be non-negative")
    if side not in {"buy", "sell"}:
        raise ValueError(f"unsupported side: {side}")
    if gross_amount == 0:
        zero = Decimal("0.00")
        return CostBreakdown(commission=zero, stamp_duty=zero, transfer_fee=zero)

    rounding = model["rounding"]
    if not isinstance(rounding, Mapping):
        raise ValueError("rounding contract must be a mapping")
    if rounding["mode"] != "ROUND_HALF_UP" or rounding["component_level"] is not True:
        raise ValueError("unsupported rounding contract")
    quantum = Decimal(1).scaleb(-int(rounding["scale"]))

    commission_raw = max(
        gross_amount * _decimal(model["commission_rate"]),
        _decimal(model["minimum_commission"]),
    )
    stamp_duty_raw = (
        gross_amount * _decimal(model["stamp_duty_rate_sell"])
        if side == "sell"
        else Decimal("0")
    )
    transfer_fee_raw = gross_amount * _decimal(model["transfer_fee_rate"])
    return CostBreakdown(
        commission=commission_raw.quantize(quantum, rounding=ROUND_HALF_UP),
        stamp_duty=stamp_duty_raw.quantize(quantum, rounding=ROUND_HALF_UP),
        transfer_fee=transfer_fee_raw.quantize(quantum, rounding=ROUND_HALF_UP),
    )
