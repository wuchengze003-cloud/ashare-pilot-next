import pytest
from ashare_quant_core import (
    ChampionHealth,
    HealthSnapshot,
    RiskAction,
    RuntimeState,
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
