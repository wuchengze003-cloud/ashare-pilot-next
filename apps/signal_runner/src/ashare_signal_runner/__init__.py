"""Deterministic production target publisher."""

from .pipeline import RunArtifacts, build_active_run, publish_run
from .runner import (
    ChampionRef,
    ContractSet,
    SignalInputs,
    build_initial_flat_signal,
    build_production_signal,
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_signal_sha256,
)

__all__ = [
    "ChampionRef",
    "ContractSet",
    "RunArtifacts",
    "SignalInputs",
    "build_active_run",
    "build_initial_flat_signal",
    "build_production_signal",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_signal_sha256",
    "publish_run",
]
