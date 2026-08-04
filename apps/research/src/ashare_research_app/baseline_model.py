"""Deterministic baseline ML model for the live pilot.

Uses scikit-learn HistGradientBoostingRegressor per prediction horizon.
Training rows are always in time order; no shuffling. The serialized
bundle is content-hashed so a Champion can bind it immutably.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from datetime import date

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5)
MODEL_CODE_VERSION = "baseline-hgb/v1"
RANDOM_STATE = 20260804


@dataclass(frozen=True)
class TrainingWindow:
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date

    def __post_init__(self) -> None:
        if not (
            self.train_end
            < self.validation_start
            <= self.validation_end
            < self.test_start
        ):
            raise ValueError("training windows must be ordered and disjoint")


class MultiHorizonModel:
    """One gradient-boosting model per horizon; score is the mean prediction."""

    def __init__(self, *, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> None:
        if not horizons:
            raise ValueError("at least one horizon is required")
        self.horizons = tuple(sorted(horizons))
        self.models: dict[int, HistGradientBoostingRegressor] = {}
        self.training_cutoff: date | None = None

    def fit(
        self,
        feature_matrix: np.ndarray,
        labels_by_horizon: dict[int, np.ndarray],
        *,
        training_cutoff: date,
    ) -> None:
        if feature_matrix.ndim != 2:
            raise ValueError("feature matrix must be 2-dimensional")
        for horizon in self.horizons:
            labels = labels_by_horizon.get(horizon)
            if labels is None:
                raise ValueError(f"missing labels for horizon {horizon}")
            valid = ~np.isnan(labels)
            if int(valid.sum()) < 20:
                raise ValueError(f"horizon {horizon} has fewer than 20 training rows")
            model = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
            model.fit(feature_matrix[valid], labels[valid])
            self.models[horizon] = model
        self.training_cutoff = training_cutoff

    def score(self, feature_matrix: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError("model is not trained")
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        predictions = np.column_stack(
            [self.models[horizon].predict(feature_matrix) for horizon in self.horizons]
        )
        return predictions.mean(axis=1)

    def bundle_bytes(self) -> bytes:
        if not self.models or self.training_cutoff is None:
            raise ValueError("cannot serialize an untrained model")
        bundle = {
            "code_version": MODEL_CODE_VERSION,
            "horizons": list(self.horizons),
            "training_cutoff": self.training_cutoff.isoformat(),
            "models": {horizon: self.models[horizon] for horizon in self.horizons},
        }
        return pickle.dumps(bundle, protocol=5)

    @staticmethod
    def from_bundle_bytes(content: bytes) -> MultiHorizonModel:
        bundle = pickle.loads(content)
        if bundle.get("code_version") != MODEL_CODE_VERSION:
            raise ValueError("unsupported model bundle version")
        model = MultiHorizonModel(horizons=tuple(bundle["horizons"]))
        model.models = dict(bundle["models"])
        model.training_cutoff = date.fromisoformat(bundle["training_cutoff"])
        return model

    @staticmethod
    def bundle_sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
