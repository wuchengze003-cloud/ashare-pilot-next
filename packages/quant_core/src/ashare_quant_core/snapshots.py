"""Immutable point-in-time market and universe views."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date

SYMBOL_PATTERN = re.compile(r"^[0-9]{6}\.(SZ|SH|BJ)$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
DAILY_BAR_SCHEMA_ID = "normalized-daily-bar/v1"
DAILY_BAR_SCHEMA_DESCRIPTOR = {
    "schema_id": DAILY_BAR_SCHEMA_ID,
    "primary_key": ["symbol", "trade_date"],
    "fields": [
        ["symbol", "string"],
        ["trade_date", "date"],
        ["open", "finite-positive-number"],
        ["high", "finite-positive-number"],
        ["low", "finite-positive-number"],
        ["close", "finite-positive-number"],
        ["volume", "finite-nonnegative-number"],
        ["amount", "finite-nonnegative-number"],
    ],
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


DAILY_BAR_SCHEMA_SHA256 = _canonical_sha256(DAILY_BAR_SCHEMA_DESCRIPTOR)


def _require_sha256(value: str, *, field_name: str) -> None:
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


def _require_symbol(value: str) -> None:
    if not SYMBOL_PATTERN.fullmatch(value):
        raise ValueError(f"invalid A-share symbol: {value}")


def _require_finite(value: float, *, field_name: str, positive: bool) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    minimum = 0 if not positive else 0.0
    if value < minimum or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} must be {qualifier}")


@dataclass(frozen=True, order=True)
class DailyBar:
    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float

    def __post_init__(self) -> None:
        _require_symbol(self.symbol)
        for field_name in ("open", "high", "low", "close"):
            object.__setattr__(self, field_name, float(getattr(self, field_name)))
            _require_finite(
                getattr(self, field_name),
                field_name=field_name,
                positive=True,
            )
        for field_name in ("volume", "amount"):
            object.__setattr__(self, field_name, float(getattr(self, field_name)))
            _require_finite(
                getattr(self, field_name),
                field_name=field_name,
                positive=False,
            )
        if self.low > min(self.open, self.close):
            raise ValueError("low cannot exceed open or close")
        if self.high < max(self.open, self.close):
            raise ValueError("high cannot be below open or close")
        if self.low > self.high:
            raise ValueError("low cannot exceed high")


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_id: str
    dataset_family_id: str
    manifest_sha256: str
    as_of: date
    data_schema_id: str
    data_schema_sha256: str
    normalization_version: str
    records: tuple[DailyBar, ...]
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_family_id:
            raise ValueError("dataset identities cannot be empty")
        if self.data_schema_id != DAILY_BAR_SCHEMA_ID:
            raise ValueError(f"unsupported data schema: {self.data_schema_id}")
        if self.data_schema_sha256 != DAILY_BAR_SCHEMA_SHA256:
            raise ValueError("data schema hash does not match the normalized record schema")
        if not self.normalization_version:
            raise ValueError("normalization_version cannot be empty")
        _require_sha256(self.manifest_sha256, field_name="manifest_sha256")

        ordered = tuple(sorted(self.records))
        seen: set[tuple[str, date]] = set()
        for record in ordered:
            if record.trade_date > self.as_of:
                raise ValueError("snapshot cannot contain records after as_of")
            key = (record.symbol, record.trade_date)
            if key in seen:
                raise ValueError(
                    f"duplicate daily bar primary key: {record.symbol} {record.trade_date}"
                )
            seen.add(key)
        object.__setattr__(self, "records", ordered)
        visible_identity = {
            "dataset_family_id": self.dataset_family_id,
            "as_of": self.as_of.isoformat(),
            "data_schema_id": self.data_schema_id,
            "data_schema_sha256": self.data_schema_sha256,
            "normalization_version": self.normalization_version,
            "records": [
                {**asdict(record), "trade_date": record.trade_date.isoformat()}
                for record in ordered
            ],
        }
        object.__setattr__(
            self,
            "snapshot_sha256",
            _canonical_sha256(visible_identity),
        )

    @property
    def latest_trade_date(self) -> date | None:
        return max((record.trade_date for record in self.records), default=None)

    def bars(
        self,
        *,
        symbol: str | None = None,
        through: date | None = None,
    ) -> tuple[DailyBar, ...]:
        if symbol is not None:
            _require_symbol(symbol)
        if through is not None and through > self.as_of:
            raise ValueError("strategy cannot request records after snapshot as_of")
        effective_through = through or self.as_of
        return tuple(
            record
            for record in self.records
            if record.trade_date <= effective_through
            and (symbol is None or record.symbol == symbol)
        )


@dataclass(frozen=True, order=True)
class UniverseMember:
    symbol: str
    valid_from: date
    valid_to: date | None
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_symbol(self.symbol)
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("universe member valid_to cannot precede valid_from")
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))


@dataclass(frozen=True)
class UniverseSnapshot:
    universe_id: str
    universe_policy_id: str
    universe_policy_version: str
    source_sha256: str
    as_of: date
    members: tuple[UniverseMember, ...]
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.universe_id or not self.universe_policy_id:
            raise ValueError("universe identities cannot be empty")
        if not self.universe_policy_version:
            raise ValueError("universe_policy_version cannot be empty")
        _require_sha256(self.source_sha256, field_name="source_sha256")

        ordered = tuple(sorted(self.members, key=lambda member: member.symbol))
        seen: set[str] = set()
        for member in ordered:
            if member.symbol in seen:
                raise ValueError(f"universe contains duplicate symbol: {member.symbol}")
            seen.add(member.symbol)
            if member.valid_from > self.as_of or (
                member.valid_to is not None and member.valid_to < self.as_of
            ):
                raise ValueError(
                    f"universe member validity does not cover as_of: {member.symbol}"
                )
        object.__setattr__(self, "members", ordered)
        visible_identity = {
            "universe_policy_id": self.universe_policy_id,
            "universe_policy_version": self.universe_policy_version,
            "as_of": self.as_of.isoformat(),
            "members": [
                {
                    **asdict(member),
                    "valid_from": member.valid_from.isoformat(),
                    "valid_to": (
                        member.valid_to.isoformat()
                        if member.valid_to is not None
                        else None
                    ),
                }
                for member in ordered
            ],
        }
        object.__setattr__(
            self,
            "snapshot_sha256",
            _canonical_sha256(visible_identity),
        )

    @property
    def eligible_symbols(self) -> frozenset[str]:
        return frozenset(member.symbol for member in self.members if member.eligible)
