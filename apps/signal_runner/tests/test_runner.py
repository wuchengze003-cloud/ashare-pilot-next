import json
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from ashare_quant_core import ChampionHealth, HealthSnapshot, RiskAction, TargetPosition
from ashare_signal_runner.runner import (
    ChampionRef,
    ContractSet,
    SignalInputs,
    build_initial_flat_signal,
    build_production_signal,
    canonical_signal_sha256,
)
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads(
    (ROOT / "contracts/schemas/production-signal.schema.json").read_text(encoding="utf-8")
)
CHAMPION_HASH = "9" * 64


def contract_set(*, champion: bool) -> ContractSet:
    return ContractSet(
        dataset_manifest_sha256="a" * 64,
        dataset_snapshot_sha256="b" * 64,
        universe_snapshot_sha256="8" * 64,
        champion_sha256=CHAMPION_HASH if champion else None,
        cost_model_sha256="c" * 64,
        market_rules_sha256="d" * 64,
        execution_policy_sha256="e" * 64,
        portfolio_risk_sha256="f" * 64,
        code_sha256="1" * 64,
        config_sha256="2" * 64,
        lockfile_sha256="3" * 64,
    )


def health(
    *,
    champion: ChampionHealth = ChampionHealth.HEALTHY,
    decision_data_valid: bool = True,
    risk_action: RiskAction = RiskAction.NONE,
) -> HealthSnapshot:
    return HealthSnapshot(
        execution_data_valid=True,
        previous_target_known=True,
        decision_data_valid=decision_data_valid,
        contracts_valid=True,
        hashes_valid=True,
        champion=champion,
        risk_action=risk_action,
    )


def champion() -> ChampionRef:
    return ChampionRef(
        strategy_id="fixture-reference",
        strategy_version="v1",
        sha256=CHAMPION_HASH,
    )


def active_signal() -> dict[str, object]:
    return build_production_signal(
        SignalInputs(
            signal_id="fixture-active",
            as_of=date(2026, 7, 29),
            latest_complete_date=date(2026, 7, 29),
            generated_at=datetime(2026, 7, 29, 8, tzinfo=UTC),
            health=health(),
            contract_set=contract_set(champion=True),
            target_positions=(TargetPosition("600000.SH", 0.3),),
            reason_codes=("CHAMPION_ACTIVE",),
            champion=champion(),
        ),
        schema=SCHEMA,
    )


def test_runner_publishes_explicit_initial_flat() -> None:
    signal = build_initial_flat_signal(
        SignalInputs(
            signal_id="fixture-flat",
            as_of=date(2026, 7, 29),
            latest_complete_date=date(2026, 7, 29),
            generated_at=datetime(2026, 7, 29, 8, tzinfo=UTC),
            health=health(champion=ChampionHealth.NEVER_ACTIVATED),
            contract_set=contract_set(champion=False),
            target_positions=(),
            reason_codes=("NO_ACTIVE_CHAMPION",),
        ),
        schema=SCHEMA,
    )

    assert signal["state"] == "FLAT"
    assert signal["champion"] is None
    assert signal["target_positions"] == []
    assert "filled" not in signal
    assert "executed" not in signal


def test_runner_publishes_active_signal() -> None:
    signal = active_signal()

    assert signal["state"] == "ACTIVE"
    assert signal["champion"] == {
        "strategy_id": "fixture-reference",
        "strategy_version": "v1",
        "sha256": CHAMPION_HASH,
    }


def test_runner_publishes_hold_with_identical_target_and_previous_hash() -> None:
    previous = active_signal()
    previous_hash = canonical_signal_sha256(previous)
    signal = build_production_signal(
        SignalInputs(
            signal_id="fixture-hold",
            as_of=date(2026, 7, 30),
            latest_complete_date=date(2026, 7, 29),
            generated_at=datetime(2026, 7, 30, 8, tzinfo=UTC),
            health=health(decision_data_valid=False),
            contract_set=contract_set(champion=True),
            target_positions=(TargetPosition("600000.SH", 0.3),),
            reason_codes=("DECISION_DATA_INVALID",),
            champion=champion(),
            sequence=2,
            previous_head_sha256="5" * 64,
            previous_signal_sha256=previous_hash,
            previous_signal=previous,
        ),
        schema=SCHEMA,
    )

    assert signal["state"] == "HOLD"
    assert signal["target_positions"] == previous["target_positions"]
    assert signal["previous_signal_sha256"] == previous_hash


def test_runner_rejects_hold_target_change() -> None:
    with pytest.raises(ValueError, match="preserve"):
        build_production_signal(
            SignalInputs(
                signal_id="fixture-bad-hold",
                as_of=date(2026, 7, 30),
                latest_complete_date=date(2026, 7, 29),
                generated_at=datetime(2026, 7, 30, 8, tzinfo=UTC),
                health=health(decision_data_valid=False),
                contract_set=contract_set(champion=True),
                target_positions=(TargetPosition("600000.SH", 0.2),),
                reason_codes=("DECISION_DATA_INVALID",),
                champion=champion(),
                sequence=2,
                previous_head_sha256="5" * 64,
                previous_signal_sha256=canonical_signal_sha256(active_signal()),
                previous_signal=active_signal(),
            ),
            schema=SCHEMA,
        )


def test_runner_publishes_reduce_only() -> None:
    previous = active_signal()
    signal = build_production_signal(
        SignalInputs(
            signal_id="fixture-reduce",
            as_of=date(2026, 7, 30),
            latest_complete_date=date(2026, 7, 30),
            generated_at=datetime(2026, 7, 30, 8, tzinfo=UTC),
            health=health(risk_action=RiskAction.REDUCE),
            contract_set=contract_set(champion=True),
            target_positions=(TargetPosition("600000.SH", 0.1),),
            reason_codes=("RISK_REDUCE_ONLY",),
            champion=champion(),
            sequence=2,
            previous_head_sha256="5" * 64,
            previous_signal_sha256=canonical_signal_sha256(previous),
            previous_signal=previous,
        ),
        schema=SCHEMA,
    )

    assert signal["state"] == "REDUCE_ONLY"
    assert signal["target_positions"] == [
        {"symbol": "600000.SH", "target_weight": 0.1}
    ]


def test_runner_rejects_reduce_only_increase() -> None:
    previous = active_signal()
    with pytest.raises(ValueError, match="cannot increase"):
        build_production_signal(
            SignalInputs(
                signal_id="fixture-bad-reduce",
                as_of=date(2026, 7, 30),
                latest_complete_date=date(2026, 7, 30),
                generated_at=datetime(2026, 7, 30, 8, tzinfo=UTC),
                health=health(risk_action=RiskAction.REDUCE),
                contract_set=contract_set(champion=True),
                target_positions=(TargetPosition("600000.SH", 0.4),),
                reason_codes=("RISK_REDUCE_ONLY",),
                champion=champion(),
                sequence=2,
                previous_head_sha256="5" * 64,
                previous_signal_sha256=canonical_signal_sha256(previous),
                previous_signal=previous,
            ),
            schema=SCHEMA,
        )


def test_risk_flat_keeps_champion_and_correct_reason() -> None:
    previous = active_signal()
    signal = build_production_signal(
        SignalInputs(
            signal_id="fixture-risk-flat",
            as_of=date(2026, 7, 30),
            latest_complete_date=date(2026, 7, 30),
            generated_at=datetime(2026, 7, 30, 8, tzinfo=UTC),
            health=health(risk_action=RiskAction.FLAT),
            contract_set=contract_set(champion=True),
            target_positions=(),
            reason_codes=("RISK_FLAT",),
            champion=champion(),
            sequence=2,
            previous_head_sha256="5" * 64,
            previous_signal_sha256=canonical_signal_sha256(previous),
            previous_signal=previous,
        ),
        schema=SCHEMA,
    )

    assert signal["state"] == "FLAT"
    assert signal["champion"] is not None
    assert signal["reason_codes"] == ["RISK_FLAT"]


def test_initial_flat_helper_rejects_risk_flat() -> None:
    with pytest.raises(ValueError, match="before any champion"):
        build_initial_flat_signal(
            SignalInputs(
                signal_id="fixture-wrong-flat",
                as_of=date(2026, 7, 30),
                latest_complete_date=date(2026, 7, 30),
                generated_at=datetime(2026, 7, 30, 8, tzinfo=UTC),
                health=health(risk_action=RiskAction.FLAT),
                contract_set=contract_set(champion=True),
                target_positions=(),
                reason_codes=("NO_ACTIVE_CHAMPION",),
                champion=champion(),
            ),
            schema=SCHEMA,
        )


@pytest.mark.parametrize(
    "generated_at",
    [
        datetime(2026, 7, 29, 8),
        datetime(2026, 7, 29, 8, tzinfo=timezone(timedelta(hours=8))),
    ],
)
def test_runner_requires_utc(generated_at: datetime) -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        build_initial_flat_signal(
            SignalInputs(
                signal_id="fixture-invalid-time",
                as_of=date(2026, 7, 29),
                latest_complete_date=date(2026, 7, 29),
                generated_at=generated_at,
                health=health(champion=ChampionHealth.NEVER_ACTIVATED),
                contract_set=contract_set(champion=False),
                target_positions=(),
                reason_codes=("NO_ACTIVE_CHAMPION",),
            ),
            schema=SCHEMA,
        )


def test_runner_rejects_future_complete_date() -> None:
    with pytest.raises(ValueError, match="cannot be later"):
        build_initial_flat_signal(
            SignalInputs(
                signal_id="fixture-future-data",
                as_of=date(2026, 7, 29),
                latest_complete_date=date(2026, 7, 30),
                generated_at=datetime(2026, 7, 29, 8, tzinfo=UTC),
                health=health(champion=ChampionHealth.NEVER_ACTIVATED),
                contract_set=contract_set(champion=False),
                target_positions=(),
                reason_codes=("NO_ACTIVE_CHAMPION",),
            ),
            schema=SCHEMA,
        )


def test_contract_set_rejects_invalid_hash() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ContractSet(
            dataset_manifest_sha256="not-a-hash",
            dataset_snapshot_sha256="b" * 64,
            universe_snapshot_sha256="8" * 64,
            champion_sha256=None,
            cost_model_sha256="c" * 64,
            market_rules_sha256="d" * 64,
            execution_policy_sha256="e" * 64,
            portfolio_risk_sha256="f" * 64,
            code_sha256="1" * 64,
            config_sha256="2" * 64,
            lockfile_sha256="3" * 64,
        )


def test_runner_rejects_previous_signal_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match expected"):
        build_production_signal(
            SignalInputs(
                signal_id="fixture-wrong-previous",
                as_of=date(2026, 7, 30),
                latest_complete_date=date(2026, 7, 29),
                generated_at=datetime(2026, 7, 30, 8, tzinfo=UTC),
                health=health(decision_data_valid=False),
                contract_set=contract_set(champion=True),
                target_positions=(TargetPosition("600000.SH", 0.3),),
                reason_codes=("DECISION_DATA_INVALID",),
                champion=champion(),
                sequence=2,
                previous_head_sha256="5" * 64,
                previous_signal_sha256="4" * 64,
                previous_signal=active_signal(),
            ),
            schema=SCHEMA,
        )


def test_runner_validates_output_against_schema() -> None:
    invalid_schema = dict(SCHEMA)
    invalid_schema["required"] = [*SCHEMA["required"], "missing_by_design"]

    with pytest.raises(ValidationError):
        build_initial_flat_signal(
            SignalInputs(
                signal_id="fixture-schema-check",
                as_of=date(2026, 7, 29),
                latest_complete_date=date(2026, 7, 29),
                generated_at=datetime(2026, 7, 29, 8, tzinfo=UTC),
                health=health(champion=ChampionHealth.NEVER_ACTIVATED),
                contract_set=contract_set(champion=False),
                target_positions=(),
                reason_codes=("NO_ACTIVE_CHAMPION",),
            ),
            schema=invalid_schema,
        )
