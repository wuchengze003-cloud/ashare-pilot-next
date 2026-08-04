"""Research application boundary.

Experiment evidence lives here: point-in-time feature replay, the
baseline ML model, and the walk-forward backtest. Research cannot
publish production signals; promotion artifacts are consumed by the
Signal Runner through immutable contracts.
"""

from .backtest import BacktestReport, LeakCheck, PilotConfig, run_walk_forward
from .baseline_model import (
    DEFAULT_HORIZONS,
    MODEL_CODE_VERSION,
    MultiHorizonModel,
    TrainingWindow,
)
from .datasets import load_manifest, load_snapshot
from .features import (
    FEATURE_NAMES,
    MIN_OBSERVATIONS,
    FeatureRow,
    build_feature_panel,
    compute_feature_row,
    forward_return_label,
)
from .replay import FeatureReplay, ReplayFeature, build_feature_replay

__all__ = [
    "DEFAULT_HORIZONS",
    "FEATURE_NAMES",
    "MIN_OBSERVATIONS",
    "MODEL_CODE_VERSION",
    "BacktestReport",
    "FeatureReplay",
    "FeatureRow",
    "LeakCheck",
    "MultiHorizonModel",
    "PilotConfig",
    "ReplayFeature",
    "TrainingWindow",
    "build_feature_panel",
    "build_feature_replay",
    "compute_feature_row",
    "forward_return_label",
    "load_manifest",
    "load_snapshot",
    "run_walk_forward",
]
