"""Verified immutable inputs to atomic production-signal artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ashare_quant_core import (
    ChampionHealth,
    DatasetSnapshot,
    HealthSnapshot,
    RiskAction,
    RuntimeState,
    TargetPosition,
    UniverseSnapshot,
    resolve_state,
    select_cost_segment,
    validate_portfolio_targets,
)
from jsonschema import Draft202012Validator, FormatChecker

from .adapters import (
    AdapterVerificationError,
    VerifiedAdapter,
    load_verified_adapter,
)
from .environment import resolve_runtime_environment
from .runner import (
    ChampionRef,
    ContractSet,
    SignalInputs,
    build_production_signal,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .snapshots import build_universe_snapshot, load_dataset_snapshot

BASE_INPUT_CONTRACTS = {
    "cost-model",
    "dataset-manifest",
    "execution-policy",
    "market-rules",
    "portfolio-risk",
    "universe",
}
OPTIONAL_INPUT_CONTRACTS = {"champion"}
NO_STRATEGY_SHA256 = hashlib.sha256(b"ashare-pilot/no-strategy/v1").hexdigest()
FIXED_CONTRACT_FIELDS = {
    "cost-model": "cost_model_sha256",
    "execution-policy": "execution_policy_sha256",
    "market-rules": "market_rules_sha256",
    "portfolio-risk": "portfolio_risk_sha256",
}


@dataclass(frozen=True)
class RunArtifacts:
    production_signal: dict[str, Any]
    runtime_manifest: dict[str, Any]


def _validate_contract(
    *,
    contract_id: str,
    document: Mapping[str, Any],
    schemas: Mapping[str, Mapping[str, Any]],
) -> None:
    schema = schemas.get(contract_id)
    if schema is None:
        raise ValueError(f"missing schema for {contract_id}")
    if document.get("contract_id") != contract_id:
        raise ValueError(f"document contract_id does not match {contract_id}")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)


def _champion_binding_issues(
    *,
    champion: Mapping[str, Any],
    contract_hashes: Mapping[str, str],
    dataset_manifest: Mapping[str, Any],
    universe: Mapping[str, Any],
    lockfile_sha256: str,
    adapter: VerifiedAdapter | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    contract_issues: list[str] = []
    hash_issues: list[str] = []
    fixed_contract_set = champion["fixed_contract_set"]
    for contract_id, champion_field in FIXED_CONTRACT_FIELDS.items():
        if fixed_contract_set[champion_field] != contract_hashes[contract_id]:
            contract_issues.append(
                f"CHAMPION_{contract_id.replace('-', '_').upper()}_MISMATCH"
            )
    compatibility = champion["promotion_compatibility"]
    if compatibility["dataset_family_id"] != dataset_manifest["dataset_family_id"]:
        contract_issues.append("CHAMPION_DATASET_FAMILY_MISMATCH")
    if compatibility["data_schema_sha256"] != dataset_manifest["data_schema_sha256"]:
        contract_issues.append("CHAMPION_DATA_SCHEMA_MISMATCH")
    if (
        compatibility["normalization_version"]
        != dataset_manifest["normalization_version"]
    ):
        contract_issues.append("CHAMPION_NORMALIZATION_VERSION_MISMATCH")
    if compatibility["universe_policy_id"] != universe["universe_policy_id"]:
        contract_issues.append("CHAMPION_UNIVERSE_POLICY_MISMATCH")
    if (
        compatibility["universe_policy_version"]
        != universe["universe_policy_version"]
    ):
        contract_issues.append("CHAMPION_UNIVERSE_POLICY_VERSION_MISMATCH")
    if fixed_contract_set["lockfile_sha256"] != lockfile_sha256:
        hash_issues.append("CHAMPION_LOCKFILE_MISMATCH")
    if adapter is None:
        hash_issues.append("ADAPTER_VERIFICATION_FAILED")
    return tuple(contract_issues), tuple(hash_issues)


def _previous_signal_context(
    *,
    previous_signal: Mapping[str, Any] | None,
    previous_signal_sha256: str | None,
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[
    tuple[TargetPosition, ...] | None,
    ChampionRef | None,
    ContractSet | None,
]:
    if (previous_signal is None) != (previous_signal_sha256 is None):
        raise ValueError("previous signal document and expected hash must be provided together")
    if previous_signal is None or previous_signal_sha256 is None:
        return None, None, None
    _validate_contract(
        contract_id="production-signal",
        document=previous_signal,
        schemas=schemas,
    )
    if canonical_json_sha256(previous_signal) != previous_signal_sha256:
        raise ValueError("previous signal hash does not match expected hash")
    positions = tuple(
        TargetPosition(
            symbol=str(item["symbol"]),
            target_weight=float(item["target_weight"]),
        )
        for item in previous_signal["target_positions"]
    )
    raw_champion = previous_signal["champion"]
    champion = (
        ChampionRef(
            strategy_id=str(raw_champion["strategy_id"]),
            strategy_version=str(raw_champion["strategy_version"]),
            sha256=str(raw_champion["sha256"]),
        )
        if raw_champion is not None
        else None
    )
    raw_contract_set = previous_signal["contract_set"]
    contract_set = ContractSet(
        dataset_manifest_sha256=str(
            raw_contract_set["dataset_manifest_sha256"]
        ),
        dataset_snapshot_sha256=str(
            raw_contract_set["dataset_snapshot_sha256"]
        ),
        universe_snapshot_sha256=str(
            raw_contract_set["universe_snapshot_sha256"]
        ),
        champion_sha256=raw_contract_set["champion_sha256"],
        cost_model_sha256=str(raw_contract_set["cost_model_sha256"]),
        market_rules_sha256=str(raw_contract_set["market_rules_sha256"]),
        execution_policy_sha256=str(raw_contract_set["execution_policy_sha256"]),
        portfolio_risk_sha256=str(raw_contract_set["portfolio_risk_sha256"]),
        code_sha256=str(raw_contract_set["code_sha256"]),
        config_sha256=str(raw_contract_set["config_sha256"]),
        lockfile_sha256=str(raw_contract_set["lockfile_sha256"]),
    )
    return positions, champion, contract_set


def _deduplicate(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def build_run(
    *,
    as_of: date,
    generated_at: datetime,
    signal_id: str,
    run_id: str,
    deployment_git_sha: str,
    repository_root: Path,
    adapter_root: Path,
    dataset_root: Path,
    documents: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
    previous_signal: Mapping[str, Any] | None = None,
    previous_signal_sha256: str | None = None,
    risk_action: RiskAction = RiskAction.NONE,
) -> RunArtifacts:
    """Build one production state from verified inputs and internal health checks."""
    environment = resolve_runtime_environment(
        repository_root=repository_root,
        deployment_git_sha=deployment_git_sha,
    )
    if generated_at.tzinfo is None or generated_at.utcoffset() != timedelta(0):
        raise ValueError("generated_at must be timezone-aware UTC")
    unknown_documents = sorted(
        documents.keys() - BASE_INPUT_CONTRACTS - OPTIONAL_INPUT_CONTRACTS
    )
    if unknown_documents:
        raise ValueError(f"unknown input contracts: {unknown_documents}")
    missing_documents = sorted(BASE_INPUT_CONTRACTS - documents.keys())
    if missing_documents:
        raise ValueError(f"missing input contracts: {missing_documents}")

    for contract_id in sorted(BASE_INPUT_CONTRACTS):
        _validate_contract(
            contract_id=contract_id,
            document=documents[contract_id],
            schemas=schemas,
        )
    champion = documents.get("champion")
    if champion is not None:
        _validate_contract(
            contract_id="champion",
            document=champion,
            schemas=schemas,
        )

    dataset_manifest = documents["dataset-manifest"]
    universe = documents["universe"]
    portfolio_risk = documents["portfolio-risk"]
    contract_hashes = {
        contract_id: canonical_json_sha256(document)
        for contract_id, document in documents.items()
        if contract_id in BASE_INPUT_CONTRACTS | OPTIONAL_INPUT_CONTRACTS
    }
    previous_positions, previous_champion, previous_contract_set = (
        _previous_signal_context(
            previous_signal=previous_signal,
            previous_signal_sha256=previous_signal_sha256,
            schemas=schemas,
        )
    )

    reason_codes: list[str] = []
    decision_data_valid = True
    contracts_valid = True
    hashes_valid = True
    dataset_as_of = date.fromisoformat(str(dataset_manifest["as_of"]))
    universe_as_of = date.fromisoformat(str(universe["as_of"]))
    if universe_as_of > as_of:
        raise ValueError("universe cannot be later than the run as_of")
    if dataset_manifest["quality_status"] != "pass":
        decision_data_valid = False
        reason_codes.append("DATASET_QUALITY_FAILED")
    if universe["quality_status"] != "pass":
        decision_data_valid = False
        reason_codes.append("UNIVERSE_QUALITY_FAILED")
    if dataset_as_of < as_of:
        decision_data_valid = False
        reason_codes.append("DATASET_STALE")
    if universe_as_of != as_of:
        decision_data_valid = False
        reason_codes.append("UNIVERSE_STALE")

    dataset_snapshot: DatasetSnapshot | None = None
    universe_snapshot: UniverseSnapshot | None = None
    snapshot_error: OSError | ValueError | None = None
    try:
        dataset_snapshot = load_dataset_snapshot(
            dataset_manifest=dataset_manifest,
            dataset_root=dataset_root,
            as_of=as_of,
        )
    except (OSError, ValueError) as exc:
        snapshot_error = exc
        hashes_valid = False
        reason_codes.append("DATASET_SNAPSHOT_INVALID")
    if (
        dataset_snapshot is not None
        and dataset_snapshot.latest_trade_date != as_of
    ):
        decision_data_valid = False
        reason_codes.append("DATASET_STALE")
    try:
        universe_snapshot = build_universe_snapshot(
            universe=universe,
            as_of=universe_as_of,
        )
    except ValueError as exc:
        snapshot_error = snapshot_error or exc
        hashes_valid = False
        reason_codes.append("UNIVERSE_SNAPSHOT_INVALID")

    current_champion: ChampionRef | None = None
    verified_adapter: VerifiedAdapter | None = None
    adapter_error: AdapterVerificationError | None = None
    champion_health = ChampionHealth.NEVER_ACTIVATED
    code_sha256 = NO_STRATEGY_SHA256
    config_sha256 = NO_STRATEGY_SHA256
    champion_hash: str | None = None
    if champion is None:
        if previous_signal is not None:
            champion_health = ChampionHealth.HEALTHY
            current_champion = previous_champion
            contracts_valid = False
            reason_codes.append("CHAMPION_MISSING")
            if previous_contract_set is None:
                raise ValueError("previous signal contract set is missing")
            code_sha256 = previous_contract_set.code_sha256
            config_sha256 = previous_contract_set.config_sha256
            champion_hash = previous_contract_set.champion_sha256
        else:
            reason_codes.append("CHAMPION_NEVER_ACTIVATED")
    else:
        champion_health = ChampionHealth.HEALTHY
        champion_hash = contract_hashes["champion"]
        current_champion = ChampionRef(
            strategy_id=str(champion["strategy_id"]),
            strategy_version=str(champion["strategy_version"]),
            sha256=champion_hash,
        )
        fixed_contract_set = champion["fixed_contract_set"]
        code_sha256 = str(fixed_contract_set["code_sha256"])
        config_sha256 = str(fixed_contract_set["config_sha256"])
        promoted_at = datetime.fromisoformat(
            str(champion["promoted_at"]).replace("Z", "+00:00")
        )
        if promoted_at > generated_at:
            raise ValueError("champion cannot be promoted after signal generation")
        try:
            verified_adapter = load_verified_adapter(
                champion=champion,
                adapter_root=adapter_root,
                schema=schemas["strategy-adapter"],
            )
        except AdapterVerificationError as exc:
            adapter_error = exc
        contract_issues, hash_issues = _champion_binding_issues(
            champion=champion,
            contract_hashes=contract_hashes,
            dataset_manifest=dataset_manifest,
            universe=universe,
            lockfile_sha256=environment.lockfile_sha256,
            adapter=verified_adapter,
        )
        if contract_issues:
            contracts_valid = False
            reason_codes.extend(contract_issues)
        if hash_issues:
            hashes_valid = False
            reason_codes.extend(hash_issues)

    health = HealthSnapshot(
        execution_data_valid=True,
        previous_target_known=True,
        decision_data_valid=decision_data_valid,
        contracts_valid=contracts_valid,
        hashes_valid=hashes_valid,
        champion=champion_health,
        risk_action=risk_action,
    )
    state = resolve_state(health)
    if state in {RuntimeState.HOLD, RuntimeState.REDUCE_ONLY} and previous_positions is None:
        verification_error = snapshot_error or adapter_error
        if verification_error is not None:
            raise ValueError(
                "cannot publish a degraded state without a previous signal: "
                f"{verification_error}"
            ) from verification_error
        raise ValueError(f"{state.value} requires a verified previous signal")

    dataset_snapshot_sha256 = (
        dataset_snapshot.snapshot_sha256
        if dataset_snapshot is not None
        else (
            previous_contract_set.dataset_snapshot_sha256
            if previous_contract_set is not None
            else None
        )
    )
    universe_snapshot_sha256 = (
        universe_snapshot.snapshot_sha256
        if universe_snapshot is not None
        else (
            previous_contract_set.universe_snapshot_sha256
            if previous_contract_set is not None
            else None
        )
    )
    if dataset_snapshot_sha256 is None or universe_snapshot_sha256 is None:
        raise ValueError("verified snapshot hashes are required to publish a signal")
    contract_set = ContractSet(
        dataset_manifest_sha256=contract_hashes["dataset-manifest"],
        dataset_snapshot_sha256=dataset_snapshot_sha256,
        universe_snapshot_sha256=universe_snapshot_sha256,
        champion_sha256=champion_hash,
        cost_model_sha256=contract_hashes["cost-model"],
        market_rules_sha256=contract_hashes["market-rules"],
        execution_policy_sha256=contract_hashes["execution-policy"],
        portfolio_risk_sha256=contract_hashes["portfolio-risk"],
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        lockfile_sha256=environment.lockfile_sha256,
    )

    targets: tuple[TargetPosition, ...]
    if state is RuntimeState.FLAT:
        targets = ()
        if risk_action is RiskAction.FLAT:
            reason_codes.append("RISK_FLAT")
    elif state is RuntimeState.HOLD:
        assert previous_positions is not None
        targets = previous_positions
    else:
        if verified_adapter is None:
            raise ValueError(f"{state.value} requires a verified strategy adapter")
        if dataset_snapshot is None or universe_snapshot is None:
            raise ValueError(f"{state.value} requires verified current snapshots")
        targets = tuple(
            verified_adapter.strategy.target_positions(
                as_of=as_of,
                dataset_snapshot=dataset_snapshot,
                universe_snapshot=universe_snapshot,
            )
        )
        validate_portfolio_targets(
            targets,
            eligible_symbols=set(universe_snapshot.eligible_symbols),
            max_positions=int(portfolio_risk["max_positions"]),
            max_single_weight=float(portfolio_risk["max_single_weight"]),
            max_gross_exposure=float(portfolio_risk["max_gross_exposure"]),
        )
        for target in targets:
            market = target.symbol.rsplit(".", 1)[-1]
            select_cost_segment(
                trade_date=as_of,
                market=market,
                model=documents["cost-model"],
            )
        if state is RuntimeState.ACTIVE:
            reason_codes.append("CHAMPION_ACTIVE")
        else:
            reason_codes.append("RISK_REDUCE")

    latest_complete_date = (
        dataset_snapshot.latest_trade_date
        if dataset_snapshot is not None
        else (
            date.fromisoformat(str(previous_signal["latest_complete_date"]))
            if previous_signal is not None
            else None
        )
    )
    if latest_complete_date is None:
        raise ValueError("latest complete trade date is unavailable")
    production_signal = build_production_signal(
        SignalInputs(
            signal_id=signal_id,
            as_of=as_of,
            latest_complete_date=latest_complete_date,
            generated_at=generated_at,
            health=health,
            contract_set=contract_set,
            target_positions=targets,
            reason_codes=_deduplicate(reason_codes),
            champion=current_champion,
            previous_signal=previous_signal,
            previous_signal_sha256=previous_signal_sha256,
        ),
        schema=schemas["production-signal"],
    )
    signal_hash = canonical_json_sha256(production_signal)
    runtime_manifest = {
        "contract_id": "runtime-manifest",
        "schema_version": "1.0.0",
        "run_id": run_id,
        "git_sha": environment.git_sha,
        "as_of": as_of.isoformat(),
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "inputs": [
            {"artifact_id": contract_id, "sha256": contract_hashes[contract_id]}
            for contract_id in sorted(contract_hashes)
        ]
        + [
            {
                "artifact_id": "lockfile",
                "sha256": environment.lockfile_sha256,
            }
        ]
        + (
            [
                {
                    "artifact_id": "strategy-adapter-manifest",
                    "sha256": verified_adapter.manifest_sha256,
                },
                {
                    "artifact_id": "strategy-adapter-package",
                    "sha256": verified_adapter.package_sha256,
                },
            ]
            if verified_adapter is not None
            else []
        )
        + [
            {
                "artifact_id": "dataset-snapshot",
                "sha256": dataset_snapshot_sha256,
            },
            {
                "artifact_id": "universe-snapshot",
                "sha256": universe_snapshot_sha256,
            },
        ]
        + (
            [{"artifact_id": "previous-production-signal", "sha256": previous_signal_sha256}]
            if previous_signal_sha256 is not None
            else []
        ),
        "outputs": [
            {"artifact_id": "production-signal", "sha256": signal_hash},
        ],
    }
    _validate_contract(
        contract_id="runtime-manifest",
        document=runtime_manifest,
        schemas=schemas,
    )
    return RunArtifacts(
        production_signal=production_signal,
        runtime_manifest=runtime_manifest,
    )


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_run(*, output_dir: Path, artifacts: RunArtifacts) -> None:
    """Publish an immutable run directory, exposing the manifest last."""
    if output_dir.exists():
        raise FileExistsError(f"run output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary_dir.exists():
        raise FileExistsError(f"temporary run output already exists: {temporary_dir}")

    temporary_dir.mkdir()
    try:
        _write_fsynced(
            temporary_dir / "production-signal.json",
            canonical_json_bytes(artifacts.production_signal),
        )
        _write_fsynced(
            temporary_dir / "runtime-manifest.json",
            canonical_json_bytes(artifacts.runtime_manifest),
        )
        _fsync_directory(temporary_dir)
        os.replace(temporary_dir, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
