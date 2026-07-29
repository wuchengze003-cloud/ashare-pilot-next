"""Deterministic production target publisher."""

from .runner import (
    ChampionRef,
    ContractSet,
    SignalInputs,
    build_initial_flat_signal,
    build_production_signal,
    canonical_signal_sha256,
)

__all__ = [
    "ChampionRef",
    "ContractSet",
    "SignalInputs",
    "build_initial_flat_signal",
    "build_production_signal",
    "canonical_signal_sha256",
]
