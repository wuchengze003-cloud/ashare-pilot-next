from datetime import UTC, date, datetime

import pytest
from ashare_quant_core import ChampionHealth, HealthSnapshot
from ashare_signal_runner import SignalInputs, build_safe_signal


def test_runner_publishes_explicit_flat_without_champion() -> None:
    signal = build_safe_signal(
        SignalInputs(
            signal_id="fixture-flat",
            as_of=date(2026, 7, 29),
            latest_complete_date=date(2026, 7, 29),
            generated_at=datetime(2026, 7, 29, 8, tzinfo=UTC),
            health=HealthSnapshot(
                execution_data_valid=True,
                previous_target_known=True,
                decision_data_valid=True,
                contracts_valid=True,
                hashes_valid=True,
                champion=ChampionHealth.NEVER_ACTIVATED,
            ),
            input_hashes={"dataset": "a" * 64},
        )
    )
    assert signal["state"] == "FLAT"
    assert signal["target_positions"] == []
    assert "filled" not in signal
    assert "executed" not in signal


def test_runner_refuses_non_flat_without_adapter() -> None:
    with pytest.raises(ValueError, match="ACTIVE requires"):
        build_safe_signal(
            SignalInputs(
                signal_id="fixture-invalid-active",
                as_of=date(2026, 7, 29),
                latest_complete_date=date(2026, 7, 29),
                generated_at=datetime(2026, 7, 29, 8, tzinfo=UTC),
                health=HealthSnapshot(
                    execution_data_valid=True,
                    previous_target_known=True,
                    decision_data_valid=True,
                    contracts_valid=True,
                    hashes_valid=True,
                    champion=ChampionHealth.HEALTHY,
                ),
                input_hashes={"dataset": "a" * 64},
            )
        )


def test_runner_refuses_ambiguous_hold_without_previous_target() -> None:
    with pytest.raises(ValueError, match="HOLD requires"):
        build_safe_signal(
            SignalInputs(
                signal_id="fixture-invalid-hold",
                as_of=date(2026, 7, 29),
                latest_complete_date=date(2026, 7, 28),
                generated_at=datetime(2026, 7, 29, 8, tzinfo=UTC),
                health=HealthSnapshot(
                    execution_data_valid=True,
                    previous_target_known=True,
                    decision_data_valid=False,
                    contracts_valid=True,
                    hashes_valid=True,
                    champion=ChampionHealth.HEALTHY,
                ),
                input_hashes={"dataset": "a" * 64},
            )
        )
