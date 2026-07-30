"""Research application boundary.

Real experiments and promotion logic are intentionally absent from the
architecture scaffold.
"""

from .replay import FeatureReplay, ReplayFeature, build_feature_replay

__all__ = ["FeatureReplay", "ReplayFeature", "build_feature_replay"]
