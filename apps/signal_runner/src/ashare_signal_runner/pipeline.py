"""Verified immutable inputs to atomic production-signal artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ashare_quant_core import (
    ChampionHealth,
    HealthSnapshot,
    RiskAction,
    RuntimeState,
    Strategy,
    TargetPosition,
    resolve_state,
    select_cost_segment,
    validate_portfolio_targets,
)
from jsonschema import Draft202012Validator, FormatChecker

from .runner import (
    ChampionRef,
    ContractSet,
    SignalInputs,
    build_production_signal,
    canonical_json_bytes,
    canonical_json_sha256,
)

BASE_INPUT_CONTRACTS = {
    "cost-model",
    "dataset-manifest",
    "execution-policy",
    "market-rules",
    "portfolio-risk",
    "universe",
}
OPTIONAL_INPUT_CONTRACTS = {"champion"}
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[a-f0-9]{40}$")
NO_STRATEGY_SHA256 = hashlib.sha256(b"ashare-pilot/no-strategy/v1").hexdigest()
PROMOTION_CONTRACT_FIELDS = {
    "cost-model": "cost_model_sha256",
    "dataset-manifest": "dataset_manifest_sha256",
    "execution-policy": "execution_policy_sha256",
    "market-rules": "market_rules_sha256",
    "portfolio-risk": "portfolio_risk_sha256",
    "universe": "universe_sha256",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_row_count(path: Path) -> int | None:
    if path.suffix.lower() != ".json":
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, list):
        return len(document)
    if isinstance(document, Mapping) and isinstance(document.get("rows"), list):
        return len(document["rows"])
    return None


def _verify_dataset_files(
    dataset_manifest: Mapping[str, Any],
    *,
    dataset_root: Path,
) -> None:
    resolved_root = dataset_root.resolve(strict=True)
    for artifact in dataset_manifest["files"]:
        relative_path = Path(str(artifact["path"]))
        candidate = dataset_root / relative_path
        if candidate.is_symlink():
            raise ValueError(f"dataset artifact cannot be a symlink: {relative_path}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"dataset artifact escapes dataset root: {relative_path}")
        actual_sha256 = _sha256_file(resolved)
        if actual_sha256 != artifact["sha256"]:
            raise ValueError(f"dataset artifact hash mismatch: {relative_path}")
        actual_row_count = _json_row_count(resolved)
        if actual_row_count is not None and actual_row_count != artifact["row_count"]:
            raise ValueError(f"dataset artifact row count mismatch: {relative_path}")


def _eligible_symbols(universe: Mapping[str, Any], *, as_of: date) -> set[str]:
    eligible: set[str] = set()
    seen: set[str] = set()
    for member in universe["members"]:
        symbol = str(member["symbol"])
        if symbol in seen:
            raise ValueError(f"universe contains duplicate symbol: {symbol}")
        seen.add(symbol)
        valid_from = date.fromisoformat(str(member["valid_from"]))
        raw_valid_to = member.get("valid_to")
        valid_to = date.fromisoformat(str(raw_valid_to)) if raw_valid_to else None
        if not valid_from <= as_of or (valid_to is not None and as_of > valid_to):
            raise ValueError(f"universe member validity does not cover as_of: {symbol}")
        if member["eligible"]:
            eligible.add(symbol)
    return eligible


def _validate_request_hashes(*, git_sha: str, lockfile_sha256: str) -> None:
    if not GIT_SHA_PATTERN.fullmatch(git_sha):
        raise ValueError("git_sha must be a lowercase 40-character Git SHA")
    if not SHA256_PATTERN.fullmatch(lockfile_sha256):
        raise ValueError("lockfile_sha256 must be a lowercase SHA-256")


def _champion_binding_issues(
    *,
    champion: Mapping[str, Any],
    contract_hashes: Mapping[str, str],
    lockfile_sha256: str,
    strategy: Strategy | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    contract_issues: list[str] = []
    hash_issues: list[str] = []
    promotion_contract_set = champion["promotion_contract_set"]
    for contract_id, champion_field in PROMOTION_CONTRACT_FIELDS.items():
        if promotion_contract_set[champion_field] != contract_hashes[contract_id]:
            contract_issues.append(
                f"CHAMPION_{contract_id.replace('-', '_').upper()}_MISMATCH"
            )
    if promotion_contract_set["lockfile_sha256"] != lockfile_sha256:
        hash_issues.append("CHAMPION_LOCKFILE_MISMATCH")
    if strategy is None:
        contract_issues.append("STRATEGY_ADAPTER_MISSING")
    else:
        if champion["strategy_id"] != strategy.strategy_id:
            contract_issues.append("CHAMPION_STRATEGY_ID_MISMATCH")
        if champion["strategy_version"] != strategy.strategy_version:
            contract_issues.append("CHAMPION_STRATEGY_VERSION_MISMATCH")
        if champion["adapter_id"] != getattr(strategy, "adapter_id", None):
            contract_issues.append("CHAMPION_ADAPTER_ID_MISMATCH")
        if champion["adapter_sha256"] != getattr(strategy, "adapter_sha256", None):
            hash_issues.append("CHAMPION_ADAPTER_HASH_MISMATCH")
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
        dataset_sha256=str(raw_contract_set["dataset_sha256"]),
        universe_sha256=str(raw_contract_set["universe_sha256"]),
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
    strategy: Strategy | None,
    as_of: date,
    generated_at: datetime,
    signal_id: str,
    run_id: str,
    git_sha: str,
    lockfile_sha256: str,
    dataset_root: Path,
    documents: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
    previous_signal: Mapping[str, Any] | None = None,
    previous_signal_sha256: str | None = None,
    risk_action: RiskAction = RiskAction.NONE,
) -> RunArtifacts:
    """Build one production state from verified inputs and internal health checks."""
    _validate_request_hashes(git_sha=git_sha, lockfile_sha256=lockfile_sha256)
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
    if dataset_as_of > as_of:
        raise ValueError("dataset manifest cannot be later than the run as_of")
    if universe_as_of > as_of:
        raise ValueError("universe cannot be later than the run as_of")
    if dataset_manifest["dataset_kind"] != "normalized":
        decision_data_valid = False
        reason_codes.append("DATASET_NOT_NORMALIZED")
    if dataset_manifest["quality_status"] != "pass":
        decision_data_valid = False
        reason_codes.append("DATASET_QUALITY_FAILED")
    if universe["quality_status"] != "pass":
        decision_data_valid = False
        reason_codes.append("UNIVERSE_QUALITY_FAILED")
    if dataset_as_of != as_of:
        decision_data_valid = False
        reason_codes.append("DATASET_STALE")
    if universe_as_of != as_of:
        decision_data_valid = False
        reason_codes.append("UNIVERSE_STALE")
    dataset_file_error: OSError | ValueError | json.JSONDecodeError | None = None
    try:
        _verify_dataset_files(dataset_manifest, dataset_root=dataset_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        dataset_file_error = exc
        hashes_valid = False
        reason_codes.append("DATASET_FILES_INVALID")

    current_champion: ChampionRef | None = None
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
        promotion_contract_set = champion["promotion_contract_set"]
        code_sha256 = str(promotion_contract_set["code_sha256"])
        config_sha256 = str(promotion_contract_set["config_sha256"])
        promoted_at = datetime.fromisoformat(
            str(champion["promoted_at"]).replace("Z", "+00:00")
        )
        if promoted_at > generated_at:
            raise ValueError("champion cannot be promoted after signal generation")
        contract_issues, hash_issues = _champion_binding_issues(
            champion=champion,
            contract_hashes=contract_hashes,
            lockfile_sha256=lockfile_sha256,
            strategy=strategy,
        )
        if champion["dataset_id"] != dataset_manifest["dataset_id"]:
            contract_issues = (*contract_issues, "CHAMPION_DATASET_ID_MISMATCH")
        if contract_issues:
            contracts_valid = False
            reason_codes.extend(contract_issues)
        if hash_issues:
            hashes_valid = False
            reason_codes.extend(hash_issues)

    contract_set = ContractSet(
        dataset_sha256=contract_hashes["dataset-manifest"],
        universe_sha256=contract_hashes["universe"],
        champion_sha256=champion_hash,
        cost_model_sha256=contract_hashes["cost-model"],
        market_rules_sha256=contract_hashes["market-rules"],
        execution_policy_sha256=contract_hashes["execution-policy"],
        portfolio_risk_sha256=contract_hashes["portfolio-risk"],
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        lockfile_sha256=lockfile_sha256,
    )

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
        if dataset_file_error is not None:
            raise ValueError(
                "cannot publish a degraded state without a previous signal: "
                f"{dataset_file_error}"
            ) from dataset_file_error
        raise ValueError(f"{state.value} requires a verified previous signal")

    targets: tuple[TargetPosition, ...]
    if state is RuntimeState.FLAT:
        targets = ()
        if risk_action is RiskAction.FLAT:
            reason_codes.append("RISK_FLAT")
    elif state is RuntimeState.HOLD:
        assert previous_positions is not None
        targets = previous_positions
    else:
        if strategy is None:
            raise ValueError(f"{state.value} requires a strategy adapter")
        targets = tuple(
            strategy.target_positions(
                as_of=as_of,
                dataset_id=str(dataset_manifest["dataset_id"]),
                universe_id=str(universe["universe_id"]),
            )
        )
        validate_portfolio_targets(
            targets,
            eligible_symbols=_eligible_symbols(universe, as_of=as_of),
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

    production_signal = build_production_signal(
        SignalInputs(
            signal_id=signal_id,
            as_of=as_of,
            latest_complete_date=dataset_as_of,
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
        "git_sha": git_sha,
        "as_of": as_of.isoformat(),
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "inputs": [
            {"artifact_id": contract_id, "sha256": contract_hashes[contract_id]}
            for contract_id in sorted(contract_hashes)
        ]
        + [{"artifact_id": "lockfile", "sha256": lockfile_sha256}]
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
