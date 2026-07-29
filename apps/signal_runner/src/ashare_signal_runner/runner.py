"""Build explicit safe-state signals without claiming execution."""

from dataclasses import dataclass
from datetime import date, datetime

from ashare_quant_core import HealthSnapshot, RuntimeState, resolve_state


@dataclass(frozen=True)
class SignalInputs:
    signal_id: str
    as_of: date
    latest_complete_date: date
    generated_at: datetime
    health: HealthSnapshot
    input_hashes: dict[str, str]


def build_safe_signal(inputs: SignalInputs) -> dict[str, object]:
    """Build a no-champion scaffold signal with explicit state."""
    state = resolve_state(inputs.health)
    if state is not RuntimeState.FLAT:
        raise ValueError(f"{state.value} requires a previous target or champion adapter")
    return {
        "contract_id": "production-signal",
        "schema_version": "1.0.0",
        "signal_id": inputs.signal_id,
        "state": state.value,
        "as_of": inputs.as_of.isoformat(),
        "latest_complete_date": inputs.latest_complete_date.isoformat(),
        "generated_at": inputs.generated_at.isoformat().replace("+00:00", "Z"),
        "champion": None,
        "previous_signal_sha256": None,
        "target_positions": [],
        "reason_codes": ["NO_ACTIVE_CHAMPION"],
        "input_hashes": inputs.input_hashes,
    }
