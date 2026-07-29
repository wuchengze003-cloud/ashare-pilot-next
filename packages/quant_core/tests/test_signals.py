import pytest
from ashare_quant_core import RuntimeState, TargetPosition, validate_target_transition

PREVIOUS = (
    TargetPosition("600000.SH", 0.3),
    TargetPosition("000001.SZ", 0.2),
)


def test_active_accepts_new_target_without_previous() -> None:
    validate_target_transition(
        state=RuntimeState.ACTIVE,
        current=(TargetPosition("600000.SH", 0.4),),
        previous=None,
    )


def test_hold_preserves_previous_target() -> None:
    validate_target_transition(
        state=RuntimeState.HOLD,
        current=PREVIOUS,
        previous=PREVIOUS,
    )


@pytest.mark.parametrize(
    "current",
    [
        (TargetPosition("600000.SH", 0.2),),
        (*PREVIOUS, TargetPosition("300001.SZ", 0.1)),
    ],
)
def test_hold_rejects_changed_target(current: tuple[TargetPosition, ...]) -> None:
    with pytest.raises(ValueError, match="preserve"):
        validate_target_transition(
            state=RuntimeState.HOLD,
            current=current,
            previous=PREVIOUS,
        )


def test_hold_requires_previous_target() -> None:
    with pytest.raises(ValueError, match="requires a previous"):
        validate_target_transition(
            state=RuntimeState.HOLD,
            current=PREVIOUS,
            previous=None,
        )


def test_reduce_only_accepts_reduction_and_removal() -> None:
    validate_target_transition(
        state=RuntimeState.REDUCE_ONLY,
        current=(TargetPosition("600000.SH", 0.1),),
        previous=PREVIOUS,
    )


def test_reduce_only_rejects_new_symbol() -> None:
    with pytest.raises(ValueError, match="cannot add"):
        validate_target_transition(
            state=RuntimeState.REDUCE_ONLY,
            current=(*PREVIOUS, TargetPosition("300001.SZ", 0.1)),
            previous=PREVIOUS,
        )


def test_reduce_only_rejects_weight_increase() -> None:
    with pytest.raises(ValueError, match="cannot increase"):
        validate_target_transition(
            state=RuntimeState.REDUCE_ONLY,
            current=(TargetPosition("600000.SH", 0.4),),
            previous=PREVIOUS,
        )


def test_reduce_only_requires_previous_target() -> None:
    with pytest.raises(ValueError, match="requires a previous"):
        validate_target_transition(
            state=RuntimeState.REDUCE_ONLY,
            current=(),
            previous=None,
        )


def test_flat_requires_empty_target() -> None:
    with pytest.raises(ValueError, match="empty target"):
        validate_target_transition(
            state=RuntimeState.FLAT,
            current=(TargetPosition("600000.SH", 0.1),),
            previous=PREVIOUS,
        )
