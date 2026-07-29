"""Deterministic transaction-cost calculation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

Market = Literal["SH", "SZ", "BJ"]
Side = Literal["buy", "sell"]


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


def select_cost_segment(
    *,
    trade_date: date,
    market: Market,
    model: Mapping[str, object],
) -> Mapping[str, object]:
    """Select exactly one market/date segment or fail closed."""
    raw_segments = model.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        raise ValueError("cost model segments must be a sequence")

    matches: list[Mapping[str, object]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            raise ValueError("cost model segment must be a mapping")
        markets = raw_segment.get("markets")
        if not isinstance(markets, Sequence) or isinstance(markets, (str, bytes)):
            raise ValueError("cost model segment markets must be a sequence")
        effective_from = date.fromisoformat(str(raw_segment["effective_from"]))
        raw_effective_to = raw_segment.get("effective_to")
        effective_to = (
            date.fromisoformat(str(raw_effective_to))
            if raw_effective_to is not None
            else None
        )
        if market in markets and effective_from <= trade_date and (
            effective_to is None or trade_date <= effective_to
        ):
            matches.append(raw_segment)

    if len(matches) != 1:
        raise ValueError(
            "cost model must match exactly one segment for "
            f"market={market} trade_date={trade_date.isoformat()}; matched={len(matches)}"
        )
    return matches[0]


def calculate_cost(
    *,
    side: Side,
    trade_date: date,
    market: Market,
    gross_amount: Decimal,
    model: Mapping[str, object],
) -> CostBreakdown:
    """Calculate date-aware costs using component-level rounding."""
    if gross_amount < 0:
        raise ValueError("gross_amount must be non-negative")
    if side not in {"buy", "sell"}:
        raise ValueError(f"unsupported side: {side}")

    rounding = model["rounding"]
    if not isinstance(rounding, Mapping):
        raise ValueError("rounding contract must be a mapping")
    if rounding["mode"] != "ROUND_HALF_UP" or rounding["component_level"] is not True:
        raise ValueError("unsupported rounding contract")
    quantum = Decimal(1).scaleb(-int(rounding["scale"]))

    commission = model["commission"]
    if not isinstance(commission, Mapping):
        raise ValueError("commission contract must be a mapping")
    segment = select_cost_segment(trade_date=trade_date, market=market, model=model)
    if gross_amount == 0:
        zero = Decimal("0.00")
        return CostBreakdown(commission=zero, stamp_duty=zero, transfer_fee=zero)

    commission_raw = max(
        gross_amount * _decimal(commission["rate"]),
        _decimal(commission["minimum"]),
    )
    stamp_duty_raw = gross_amount * _decimal(segment[f"stamp_duty_rate_{side}"])
    transfer_fee_raw = gross_amount * _decimal(segment[f"transfer_fee_rate_{side}"])
    return CostBreakdown(
        commission=commission_raw.quantize(quantum, rounding=ROUND_HALF_UP),
        stamp_duty=stamp_duty_raw.quantize(quantum, rounding=ROUND_HALF_UP),
        transfer_fee=transfer_fee_raw.quantize(quantum, rounding=ROUND_HALF_UP),
    )
