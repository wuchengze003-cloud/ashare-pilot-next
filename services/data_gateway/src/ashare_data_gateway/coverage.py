"""Point-in-time historical universe coverage audit."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

SYMBOL_PATTERN = re.compile(r"^[0-9]{6}\.(SZ|SH|BJ)$")


def _require_symbol(symbol: str) -> None:
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"invalid A-share symbol: {symbol}")


@dataclass(frozen=True, order=True)
class UniverseMembership:
    symbol: str
    valid_from: date
    valid_to: date | None

    def __post_init__(self) -> None:
        _require_symbol(self.symbol)
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("membership valid_to cannot precede valid_from")

    def contains(self, trade_date: date) -> bool:
        return self.valid_from <= trade_date and (
            self.valid_to is None or trade_date <= self.valid_to
        )


@dataclass(frozen=True, order=True)
class SecurityLifecycle:
    symbol: str
    listed_on: date
    delisted_on: date | None

    def __post_init__(self) -> None:
        _require_symbol(self.symbol)
        if self.delisted_on is not None and self.delisted_on < self.listed_on:
            raise ValueError("delisted_on cannot precede listed_on")

    def contains(self, trade_date: date) -> bool:
        return self.listed_on <= trade_date and (
            self.delisted_on is None or trade_date <= self.delisted_on
        )


@dataclass(frozen=True, order=True)
class CoverageGap:
    symbol: str
    trade_date: date


@dataclass(frozen=True, order=True)
class MemberCountAnomaly:
    trade_date: date
    actual_count: int
    expected_count: int


@dataclass(frozen=True)
class CoverageAudit:
    audit_id: str
    universe_policy_id: str
    universe_policy_version: str
    window_start: date
    window_end: date
    generated_at: datetime
    expected_member_count: int | None
    open_trade_days: int
    expected_member_days: int
    bar_member_days: int
    suspended_member_days: int
    expected_delisted_member_day_count: int
    expected_delisted_member_days: tuple[CoverageGap, ...]
    missing_member_days: tuple[CoverageGap, ...]
    silent_skip_symbols: tuple[str, ...]
    delisted_symbols_in_scope: tuple[str, ...]
    member_count_anomalies: tuple[MemberCountAnomaly, ...]
    provenance_warnings: tuple[str, ...]
    passed: bool
    reason_codes: tuple[str, ...]

    @property
    def document(self) -> dict[str, object]:
        return {
            "contract_id": "coverage-audit",
            "schema_version": "2.0.0",
            "audit_id": self.audit_id,
            "universe_policy_id": self.universe_policy_id,
            "universe_policy_version": self.universe_policy_version,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "generated_at": self.generated_at.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "expected_member_count": self.expected_member_count,
            "open_trade_days": self.open_trade_days,
            "expected_member_days": self.expected_member_days,
            "bar_member_days": self.bar_member_days,
            "suspended_member_days": self.suspended_member_days,
            "expected_delisted_member_day_count": (
                self.expected_delisted_member_day_count
            ),
            "expected_delisted_member_days": [
                {
                    **asdict(member_day),
                    "trade_date": member_day.trade_date.isoformat(),
                }
                for member_day in self.expected_delisted_member_days
            ],
            "missing_member_days": [
                {
                    **asdict(gap),
                    "trade_date": gap.trade_date.isoformat(),
                }
                for gap in self.missing_member_days
            ],
            "silent_skip_symbols": list(self.silent_skip_symbols),
            "delisted_symbols_in_scope": list(self.delisted_symbols_in_scope),
            "member_count_anomalies": [
                {
                    **asdict(anomaly),
                    "trade_date": anomaly.trade_date.isoformat(),
                }
                for anomaly in self.member_count_anomalies
            ],
            "provenance_warnings": list(self.provenance_warnings),
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
        }


def _active_members(
    *,
    trade_date: date,
    memberships: tuple[UniverseMembership, ...],
    lifecycles: dict[str, SecurityLifecycle],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            membership.symbol
            for membership in memberships
            if membership.contains(trade_date)
            and lifecycles[membership.symbol].contains(trade_date)
        )
    )


def audit_historical_coverage(
    *,
    audit_id: str,
    universe_policy_id: str,
    universe_policy_version: str,
    window_start: date,
    window_end: date,
    generated_at: datetime,
    trading_days: tuple[date, ...],
    memberships: tuple[UniverseMembership, ...],
    lifecycles: tuple[SecurityLifecycle, ...],
    bar_keys: frozenset[tuple[str, date]],
    suspension_keys: frozenset[tuple[str, date]],
    expected_member_count: int | None,
    provenance_warnings: tuple[str, ...] = (),
) -> CoverageAudit:
    """Audit every expected historical member-day without silently skipping symbols.

    ``suspension_keys`` must contain only supplier records already verified as
    ``suspend_type == "S"``. A member-day is classified exactly once, in bar,
    suspension, expected-delist, then missing precedence order.
    """
    if not audit_id or not universe_policy_id or not universe_policy_version:
        raise ValueError("audit and universe policy identities cannot be empty")
    if window_end < window_start:
        raise ValueError("coverage window_end cannot precede window_start")
    if generated_at.tzinfo is None or generated_at.utcoffset() != timedelta(0):
        raise ValueError("generated_at must be timezone-aware UTC")
    if expected_member_count is not None and expected_member_count < 1:
        raise ValueError("expected_member_count must be positive")

    ordered_days = tuple(sorted(set(trading_days)))
    if len(ordered_days) != len(trading_days):
        raise ValueError("trading_days must be unique")
    if not ordered_days:
        raise ValueError("coverage audit requires at least one trading day")
    if any(day < window_start or day > window_end for day in ordered_days):
        raise ValueError("trading_days must stay inside the audit window")

    lifecycle_by_symbol: dict[str, SecurityLifecycle] = {}
    for lifecycle in lifecycles:
        if lifecycle.symbol in lifecycle_by_symbol:
            raise ValueError(f"duplicate lifecycle for {lifecycle.symbol}")
        lifecycle_by_symbol[lifecycle.symbol] = lifecycle
    missing_lifecycles = sorted(
        {membership.symbol for membership in memberships} - lifecycle_by_symbol.keys()
    )
    if missing_lifecycles:
        raise ValueError(f"memberships lack lifecycle records: {missing_lifecycles}")

    ordered_memberships = tuple(sorted(memberships))
    for index, membership in enumerate(ordered_memberships):
        for other in ordered_memberships[index + 1 :]:
            if other.symbol != membership.symbol:
                continue
            first_end = membership.valid_to or date.max
            second_end = other.valid_to or date.max
            if membership.valid_from <= second_end and other.valid_from <= first_end:
                raise ValueError(f"overlapping memberships for {membership.symbol}")

    for symbol, trade_date in bar_keys | suspension_keys:
        _require_symbol(symbol)
        if trade_date < window_start or trade_date > window_end:
            raise ValueError("availability key is outside the audit window")

    for warning in provenance_warnings:
        if not warning or not re.fullmatch(r"[A-Z0-9_]+", warning):
            raise ValueError("provenance warnings must be uppercase reason codes")

    expected_keys: set[tuple[str, date]] = set()
    anomalies: list[MemberCountAnomaly] = []
    for trade_date in ordered_days:
        active = _active_members(
            trade_date=trade_date,
            memberships=ordered_memberships,
            lifecycles=lifecycle_by_symbol,
        )
        expected_keys.update((symbol, trade_date) for symbol in active)
        if expected_member_count is not None and len(active) != expected_member_count:
            anomalies.append(
                MemberCountAnomaly(
                    trade_date=trade_date,
                    actual_count=len(active),
                    expected_count=expected_member_count,
                )
            )

    covered_by_bar = expected_keys & bar_keys
    covered_by_suspension = (expected_keys - covered_by_bar) & suspension_keys
    remaining_keys = expected_keys - covered_by_bar - covered_by_suspension
    covered_by_expected_delist = {
        (symbol, trade_date)
        for symbol, trade_date in remaining_keys
        if lifecycle_by_symbol[symbol].delisted_on == trade_date
    }
    missing = tuple(
        CoverageGap(symbol=symbol, trade_date=trade_date)
        for symbol, trade_date in sorted(
            remaining_keys - covered_by_expected_delist
        )
    )
    expected_delisted = tuple(
        CoverageGap(symbol=symbol, trade_date=trade_date)
        for symbol, trade_date in sorted(covered_by_expected_delist)
    )
    expected_symbols = {symbol for symbol, _ in expected_keys}
    observed_symbols = {
        symbol
        for symbol, _trade_date in (
            covered_by_bar | covered_by_suspension | covered_by_expected_delist
        )
    }
    silent_skips = tuple(sorted(expected_symbols - observed_symbols))
    delisted_in_scope = tuple(
        sorted(
            symbol
            for symbol in expected_symbols
            if lifecycle_by_symbol[symbol].delisted_on is not None
            and lifecycle_by_symbol[symbol].delisted_on <= window_end
        )
    )
    reason_codes = tuple(
        code
        for condition, code in (
            (bool(missing), "MISSING_MEMBER_DAYS"),
            (bool(silent_skips), "SILENT_MEMBER_SKIP"),
            (bool(anomalies), "MEMBER_COUNT_ANOMALY"),
        )
        if condition
    )
    return CoverageAudit(
        audit_id=audit_id,
        universe_policy_id=universe_policy_id,
        universe_policy_version=universe_policy_version,
        window_start=window_start,
        window_end=window_end,
        generated_at=generated_at,
        expected_member_count=expected_member_count,
        open_trade_days=len(ordered_days),
        expected_member_days=len(expected_keys),
        bar_member_days=len(covered_by_bar),
        suspended_member_days=len(covered_by_suspension),
        expected_delisted_member_day_count=len(covered_by_expected_delist),
        expected_delisted_member_days=expected_delisted,
        missing_member_days=missing,
        silent_skip_symbols=silent_skips,
        delisted_symbols_in_scope=delisted_in_scope,
        member_count_anomalies=tuple(anomalies),
        provenance_warnings=tuple(sorted(set(provenance_warnings))),
        passed=not reason_codes,
        reason_codes=reason_codes,
    )
