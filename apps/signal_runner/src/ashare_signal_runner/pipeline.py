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
    Strategy,
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

REQUIRED_INPUT_CONTRACTS = {
    "champion",
    "cost-model",
    "dataset-manifest",
    "execution-policy",
    "market-rules",
    "portfolio-risk",
    "universe",
}
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[a-f0-9]{40}$")
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


def _verify_champion_bindings(
    *,
    champion: Mapping[str, Any],
    contract_hashes: Mapping[str, str],
    lockfile_sha256: str,
    strategy: Strategy,
) -> None:
    promotion_contract_set = champion["promotion_contract_set"]
    for contract_id, champion_field in PROMOTION_CONTRACT_FIELDS.items():
        if promotion_contract_set[champion_field] != contract_hashes[contract_id]:
            raise ValueError(
                f"champion {champion_field} does not match current {contract_id}"
            )
    if promotion_contract_set["lockfile_sha256"] != lockfile_sha256:
        raise ValueError("champion lockfile_sha256 does not match current lockfile")
    if champion["adapter_id"] != getattr(strategy, "adapter_id", None):
        raise ValueError("champion adapter_id does not match strategy adapter")
    if champion["adapter_sha256"] != getattr(strategy, "adapter_sha256", None):
        raise ValueError("champion adapter_sha256 does not match strategy adapter")


def build_active_run(
    *,
    strategy: Strategy,
    as_of: date,
    generated_at: datetime,
    signal_id: str,
    run_id: str,
    git_sha: str,
    lockfile_sha256: str,
    dataset_root: Path,
    documents: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
) -> RunArtifacts:
    """Build one ACTIVE signal and runtime manifest from verified contracts."""
    _validate_request_hashes(git_sha=git_sha, lockfile_sha256=lockfile_sha256)
    if generated_at.tzinfo is None or generated_at.utcoffset() != timedelta(0):
        raise ValueError("generated_at must be timezone-aware UTC")
    missing_documents = sorted(REQUIRED_INPUT_CONTRACTS - documents.keys())
    if missing_documents:
        raise ValueError(f"missing input contracts: {missing_documents}")

    for contract_id in sorted(REQUIRED_INPUT_CONTRACTS):
        _validate_contract(
            contract_id=contract_id,
            document=documents[contract_id],
            schemas=schemas,
        )

    dataset_manifest = documents["dataset-manifest"]
    universe = documents["universe"]
    champion = documents["champion"]
    portfolio_risk = documents["portfolio-risk"]
    contract_hashes = {
        contract_id: canonical_json_sha256(document)
        for contract_id, document in documents.items()
        if contract_id in REQUIRED_INPUT_CONTRACTS
    }
    if dataset_manifest["dataset_kind"] != "normalized":
        raise ValueError("production inference requires a normalized dataset")
    if dataset_manifest["quality_status"] != "pass":
        raise ValueError("dataset quality must pass")
    if universe["quality_status"] != "pass":
        raise ValueError("universe quality must pass")
    if date.fromisoformat(str(dataset_manifest["as_of"])) != as_of:
        raise ValueError("dataset manifest as_of must match the run as_of")
    if date.fromisoformat(str(universe["as_of"])) != as_of:
        raise ValueError("universe as_of must match the run as_of")
    if champion["dataset_id"] != dataset_manifest["dataset_id"]:
        raise ValueError("champion dataset_id does not match dataset manifest")
    if champion["strategy_id"] != strategy.strategy_id:
        raise ValueError("champion strategy_id does not match strategy adapter")
    if champion["strategy_version"] != strategy.strategy_version:
        raise ValueError("champion strategy_version does not match strategy adapter")
    _verify_champion_bindings(
        champion=champion,
        contract_hashes=contract_hashes,
        lockfile_sha256=lockfile_sha256,
        strategy=strategy,
    )
    promoted_at = datetime.fromisoformat(str(champion["promoted_at"]).replace("Z", "+00:00"))
    if promoted_at > generated_at:
        raise ValueError("champion cannot be promoted after signal generation")

    _verify_dataset_files(dataset_manifest, dataset_root=dataset_root)

    champion_hash = contract_hashes["champion"]
    promotion_contract_set = champion["promotion_contract_set"]
    contract_set = ContractSet(
        dataset_sha256=contract_hashes["dataset-manifest"],
        universe_sha256=contract_hashes["universe"],
        champion_sha256=champion_hash,
        cost_model_sha256=contract_hashes["cost-model"],
        market_rules_sha256=contract_hashes["market-rules"],
        execution_policy_sha256=contract_hashes["execution-policy"],
        portfolio_risk_sha256=contract_hashes["portfolio-risk"],
        code_sha256=str(promotion_contract_set["code_sha256"]),
        config_sha256=str(promotion_contract_set["config_sha256"]),
        lockfile_sha256=lockfile_sha256,
    )

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
    production_signal = build_production_signal(
        SignalInputs(
            signal_id=signal_id,
            as_of=as_of,
            latest_complete_date=as_of,
            generated_at=generated_at,
            health=HealthSnapshot(
                execution_data_valid=True,
                previous_target_known=True,
                decision_data_valid=True,
                contracts_valid=True,
                hashes_valid=True,
                champion=ChampionHealth.HEALTHY,
                risk_action=RiskAction.NONE,
            ),
            contract_set=contract_set,
            target_positions=targets,
            reason_codes=("CHAMPION_ACTIVE",),
            champion=ChampionRef(
                strategy_id=str(champion["strategy_id"]),
                strategy_version=str(champion["strategy_version"]),
                sha256=champion_hash,
            ),
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
        + [{"artifact_id": "lockfile", "sha256": lockfile_sha256}],
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
