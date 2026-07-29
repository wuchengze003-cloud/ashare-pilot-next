"""Build and validate deterministic production signals."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ashare_quant_core import (
    ChampionHealth,
    HealthSnapshot,
    RiskAction,
    RuntimeState,
    TargetPosition,
    resolve_state,
    validate_target_transition,
)
from jsonschema import Draft202012Validator, FormatChecker

SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def _require_sha256(value: str, *, field: str) -> None:
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True)
class ChampionRef:
    strategy_id: str
    strategy_version: str
    sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.sha256, field="champion.sha256")


@dataclass(frozen=True)
class ContractSet:
    dataset_sha256: str
    universe_sha256: str
    champion_sha256: str | None
    cost_model_sha256: str
    market_rules_sha256: str
    execution_policy_sha256: str
    portfolio_risk_sha256: str
    code_sha256: str
    config_sha256: str
    lockfile_sha256: str

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if value is not None:
                _require_sha256(value, field=f"contract_set.{field}")


@dataclass(frozen=True)
class SignalInputs:
    signal_id: str
    as_of: date
    latest_complete_date: date
    generated_at: datetime
    health: HealthSnapshot
    contract_set: ContractSet
    target_positions: tuple[TargetPosition, ...]
    reason_codes: tuple[str, ...]
    champion: ChampionRef | None = None
    previous_signal_sha256: str | None = None
    previous_signal: Mapping[str, Any] | None = None


def canonical_signal_sha256(document: Mapping[str, Any]) -> str:
    """Hash a signal using one stable JSON representation."""
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positions_from_document(document: Mapping[str, Any]) -> tuple[TargetPosition, ...]:
    raw_positions = document.get("target_positions")
    if not isinstance(raw_positions, list):
        raise ValueError("previous signal target_positions must be a list")
    return tuple(
        TargetPosition(
            symbol=str(item["symbol"]),
            target_weight=float(item["target_weight"]),
        )
        for item in raw_positions
    )


def _validate_dates(inputs: SignalInputs) -> None:
    if inputs.generated_at.tzinfo is None or inputs.generated_at.utcoffset() != timedelta(0):
        raise ValueError("generated_at must be timezone-aware UTC")
    if inputs.latest_complete_date > inputs.as_of:
        raise ValueError("latest_complete_date cannot be later than as_of")


def _validate_champion(inputs: SignalInputs, state: RuntimeState) -> None:
    if inputs.health.champion is ChampionHealth.NEVER_ACTIVATED:
        if inputs.champion is not None:
            raise ValueError("never-activated state cannot reference a champion")
    elif inputs.champion is None:
        raise ValueError(f"{inputs.health.champion.value} requires a champion reference")

    expected_hash = inputs.champion.sha256 if inputs.champion else None
    if inputs.contract_set.champion_sha256 != expected_hash:
        raise ValueError("contract_set champion hash does not match champion reference")
    if state is RuntimeState.ACTIVE and inputs.champion is None:
        raise ValueError("ACTIVE requires a champion")


def _validate_schema(
    document: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.validate(document)


def _validate_document_semantics(document: Mapping[str, Any]) -> None:
    as_of = date.fromisoformat(str(document["as_of"]))
    latest_complete_date = date.fromisoformat(str(document["latest_complete_date"]))
    if latest_complete_date > as_of:
        raise ValueError("previous signal latest_complete_date is later than as_of")
    generated_at = datetime.fromisoformat(
        str(document["generated_at"]).replace("Z", "+00:00")
    )
    if generated_at.tzinfo is None or generated_at.utcoffset() != timedelta(0):
        raise ValueError("previous signal generated_at must use UTC")
    champion = document["champion"]
    champion_hash = document["contract_set"]["champion_sha256"]
    if champion is None and champion_hash is not None:
        raise ValueError("previous signal champion hash requires a champion")
    if champion is not None and champion_hash != champion["sha256"]:
        raise ValueError("previous signal champion hash does not match champion")


def build_production_signal(
    inputs: SignalInputs,
    *,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a signal only after validating state, history, hashes, and schema."""
    _validate_dates(inputs)
    if not inputs.reason_codes:
        raise ValueError("at least one reason code is required")

    state = resolve_state(inputs.health)
    _validate_champion(inputs, state)

    previous_positions: Sequence[TargetPosition] | None = None
    previous_signal_sha256: str | None = None
    if (inputs.previous_signal is None) != (inputs.previous_signal_sha256 is None):
        raise ValueError("previous signal document and expected hash must be provided together")
    if inputs.previous_signal_sha256 is not None:
        _require_sha256(
            inputs.previous_signal_sha256,
            field="previous_signal_sha256",
        )
    if inputs.previous_signal is not None and inputs.previous_signal_sha256 is not None:
        _validate_schema(inputs.previous_signal, schema)
        _validate_document_semantics(inputs.previous_signal)
        previous_positions = _positions_from_document(inputs.previous_signal)
        actual_previous_hash = canonical_signal_sha256(inputs.previous_signal)
        if actual_previous_hash != inputs.previous_signal_sha256:
            raise ValueError("previous signal hash does not match expected hash")
        previous_signal_sha256 = inputs.previous_signal_sha256

    validate_target_transition(
        state=state,
        current=inputs.target_positions,
        previous=previous_positions,
    )

    signal = {
        "contract_id": "production-signal",
        "schema_version": "2.0.0",
        "signal_id": inputs.signal_id,
        "state": state.value,
        "as_of": inputs.as_of.isoformat(),
        "latest_complete_date": inputs.latest_complete_date.isoformat(),
        "generated_at": inputs.generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "champion": asdict(inputs.champion) if inputs.champion else None,
        "previous_signal_sha256": previous_signal_sha256,
        "target_positions": [
            asdict(position)
            for position in sorted(inputs.target_positions, key=lambda item: item.symbol)
        ],
        "reason_codes": list(inputs.reason_codes),
        "contract_set": asdict(inputs.contract_set),
    }
    _validate_schema(signal, schema)
    return signal


def build_initial_flat_signal(
    inputs: SignalInputs,
    *,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the canonical no-champion state and no other kind of FLAT."""
    if (
        inputs.health.champion is not ChampionHealth.NEVER_ACTIVATED
        or inputs.health.risk_action is not RiskAction.NONE
        or inputs.champion is not None
        or inputs.previous_signal_sha256 is not None
        or inputs.previous_signal is not None
        or inputs.target_positions
    ):
        raise ValueError("initial FLAT is only valid before any champion is activated")
    return build_production_signal(inputs, schema=schema)
