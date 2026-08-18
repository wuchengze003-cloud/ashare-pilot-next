import pytest
from ashare_quant_core import (
    ChampionHealth,
    HealthSnapshot,
    RiskAction,
    RuntimeState,
    constrain_execution_targets,
    resolve_state,
)


def snapshot(**overrides: object) -> HealthSnapshot:
    values: dict[str, object] = {
        "execution_data_valid": True,
        "previous_target_known": True,
        "decision_data_valid": True,
        "contracts_valid": True,
        "hashes_valid": True,
        "champion": ChampionHealth.HEALTHY,
        "risk_action": RiskAction.NONE,
    }
    values.update(overrides)
    return HealthSnapshot(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("health", "expected"),
    [
        (snapshot(), RuntimeState.ACTIVE),
        (snapshot(execution_data_valid=False, risk_action=RiskAction.FLAT), RuntimeState.HOLD),
        (snapshot(previous_target_known=False, risk_action=RiskAction.REDUCE), RuntimeState.HOLD),
        (snapshot(decision_data_valid=False), RuntimeState.HOLD),
        (snapshot(contracts_valid=False), RuntimeState.HOLD),
        (snapshot(hashes_valid=False), RuntimeState.HOLD),
        (snapshot(risk_action=RiskAction.FLAT), RuntimeState.FLAT),
        (snapshot(risk_action=RiskAction.REDUCE), RuntimeState.REDUCE_ONLY),
        (snapshot(champion=ChampionHealth.WITHDRAWN), RuntimeState.REDUCE_ONLY),
        (snapshot(champion=ChampionHealth.NEVER_ACTIVATED), RuntimeState.FLAT),
    ],
)
def test_state_priority(health: HealthSnapshot, expected: RuntimeState) -> None:
    assert resolve_state(health) is expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (RuntimeState.ACTIVE, {"000001.SZ": 1000, "600000.SH": 500}),
        (RuntimeState.HOLD, {"000001.SZ": 300, "600000.SH": 700}),
        (RuntimeState.REDUCE_ONLY, {"000001.SZ": 300, "600000.SH": 500}),
        (RuntimeState.FLAT, {"000001.SZ": 0, "600000.SH": 0}),
    ],
)
def test_execution_targets_follow_runtime_state(
    state: RuntimeState,
    expected: dict[str, int],
) -> None:
    assert (
        constrain_execution_targets(
            state=state,
            desired_shares={"000001.SZ": 1000, "600000.SH": 500},
            current_shares={"000001.SZ": 300, "600000.SH": 700},
        )
        == expected
    )
