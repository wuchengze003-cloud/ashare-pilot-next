import pytest
from ashare_quant_core import TargetPosition, validate_portfolio_targets


def validate(targets: tuple[TargetPosition, ...]) -> None:
    validate_portfolio_targets(
        targets,
        eligible_symbols={"000001.SZ", "600000.SH"},
        max_positions=2,
        max_single_weight=0.4,
        max_gross_exposure=0.6,
    )


def test_portfolio_accepts_targets_within_all_limits() -> None:
    validate(
        (
            TargetPosition("000001.SZ", 0.3),
            TargetPosition("600000.SH", 0.2),
        )
    )


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (
            (
                TargetPosition("000001.SZ", 0.2),
                TargetPosition("000001.SZ", 0.1),
            ),
            "unique",
        ),
        ((TargetPosition("300308.SZ", 0.2),), "ineligible"),
        ((TargetPosition("000001.SZ", 0.5),), "single-position"),
        (
            (
                TargetPosition("000001.SZ", 0.4),
                TargetPosition("600000.SH", 0.3),
            ),
            "gross-exposure",
        ),
    ],
)
def test_portfolio_rejects_invalid_targets(
    targets: tuple[TargetPosition, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate(targets)
