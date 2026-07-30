import hashlib
import inspect
import json
import shutil
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from ashare_quant_core import RiskAction
from ashare_signal_runner import (
    build_run,
    canonical_json_bytes,
    canonical_json_sha256,
    publish_run,
)
from ashare_signal_runner.snapshots import load_dataset_snapshot
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = ROOT / "contracts/golden_fixtures/vertical-slice"
EXAMPLES = ROOT / "contracts/examples"
SCHEMAS = ROOT / "contracts/schemas"
ADAPTER_ROOT = (
    ROOT / "apps/signal_runner/tests/fixtures/approved_adapters"
)
DEPLOYMENT_GIT_SHA = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
LOCKFILE_SHA256 = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
FIXED_CONTRACT_FIELDS = {
    "cost-model": "cost_model_sha256",
    "execution-policy": "execution_policy_sha256",
    "market-rules": "market_rules_sha256",
    "portfolio-risk": "portfolio_risk_sha256",
}


def load_json(path: Path) -> dict[str, object] | list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_schemas() -> dict[str, dict[str, object]]:
    registry = load_json(ROOT / "contracts/registry.json")
    assert isinstance(registry, dict)
    schemas: dict[str, dict[str, object]] = {}
    for entry in registry["contracts"]:
        schema = load_json(ROOT / "contracts" / entry["schema"])
        assert isinstance(schema, dict)
        schemas[entry["contract_id"]] = schema
    return schemas


def load_documents() -> dict[str, dict[str, object]]:
    paths = {
        "champion": FIXTURE_ROOT / "champion.json",
        "cost-model": EXAMPLES / "cost-model.example.json",
        "dataset-manifest": FIXTURE_ROOT / "dataset-manifest.json",
        "execution-policy": EXAMPLES / "execution-policy.example.json",
        "market-rules": EXAMPLES / "market-rules.example.json",
        "portfolio-risk": EXAMPLES / "portfolio-risk.example.json",
        "universe": FIXTURE_ROOT / "universe.json",
    }
    documents: dict[str, dict[str, object]] = {}
    for contract_id, path in paths.items():
        document = load_json(path)
        assert isinstance(document, dict)
        documents[contract_id] = document
    return documents


def bind_champion_to_documents(
    documents: dict[str, dict[str, object]],
) -> None:
    champion = documents["champion"]
    fixed_contract_set = champion["fixed_contract_set"]
    assert isinstance(fixed_contract_set, dict)
    for contract_id, champion_field in FIXED_CONTRACT_FIELDS.items():
        fixed_contract_set[champion_field] = canonical_json_sha256(
            documents[contract_id]
        )
    fixed_contract_set["lockfile_sha256"] = LOCKFILE_SHA256


def build_fixture_run(
    *,
    dataset_root: Path = FIXTURE_ROOT,
    documents: dict[str, dict[str, object]] | None = None,
    adapter_root: Path = ADAPTER_ROOT,
    deployment_git_sha: str = DEPLOYMENT_GIT_SHA,
    as_of: date = date(2026, 7, 29),
    generated_at: datetime = datetime(2026, 7, 29, 8, 10, tzinfo=UTC),
    previous_signal: dict[str, object] | None = None,
    previous_signal_sha256: str | None = None,
    risk_action: RiskAction = RiskAction.NONE,
):
    return build_run(
        as_of=as_of,
        generated_at=generated_at,
        signal_id=f"fixture-signal-{as_of.isoformat()}",
        run_id=f"fixture-run-{as_of.isoformat()}",
        deployment_git_sha=deployment_git_sha,
        repository_root=ROOT,
        adapter_root=adapter_root,
        dataset_root=dataset_root,
        documents=documents or load_documents(),
        schemas=load_schemas(),
        previous_signal=previous_signal,
        previous_signal_sha256=previous_signal_sha256,
        risk_action=risk_action,
    )


def write_fixture_dataset(
    *,
    dataset_root: Path,
    rows: list[dict[str, object]],
    documents: dict[str, dict[str, object]],
    manifest_as_of: date,
) -> None:
    content = (json.dumps(rows, ensure_ascii=True, indent=2) + "\n").encode()
    (dataset_root / "daily-bars.json").write_bytes(content)
    manifest = documents["dataset-manifest"]
    manifest["dataset_id"] = (
        f"fixture-daily-{manifest_as_of.isoformat()}-{hashlib.sha256(content).hexdigest()[:12]}"
    )
    manifest["as_of"] = manifest_as_of.isoformat()
    manifest["generated_at"] = (
        f"{manifest_as_of.isoformat()}T08:00:00Z"
    )
    files = manifest["files"]
    assert isinstance(files, list)
    artifact = files[0]
    assert isinstance(artifact, dict)
    trade_dates = [str(row["trade_date"]) for row in rows]
    artifact.update(
        {
            "sha256": hashlib.sha256(content).hexdigest(),
            "row_count": len(rows),
            "file_size_bytes": len(content),
            "min_trade_date": min(trade_dates),
            "max_trade_date": max(trade_dates),
        }
    )


def adapter_package_sha256(manifest: dict[str, object]) -> str:
    return canonical_json_sha256(
        {
            "adapter_id": manifest["adapter_id"],
            "strategy_id": manifest["strategy_id"],
            "strategy_version": manifest["strategy_version"],
            "entrypoint": manifest["entrypoint"],
            "code_sha256": manifest["code_sha256"],
            "config_sha256": manifest["config_sha256"],
        }
    )


def rewrite_adapter_config(
    *,
    adapter_root: Path,
    documents: dict[str, dict[str, object]],
    forced_symbol: str | None,
) -> None:
    package_root = adapter_root / "fixture-reference-adapter/v1"
    config_bytes = (
        json.dumps(
            {"target_weight": 0.15, "forced_symbol": forced_symbol},
            ensure_ascii=True,
            indent=2,
        )
        + "\n"
    ).encode()
    (package_root / "config.json").write_bytes(config_bytes)
    manifest = load_json(package_root / "adapter.json")
    assert isinstance(manifest, dict)
    manifest["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    manifest["package_sha256"] = adapter_package_sha256(manifest)
    (package_root / "adapter.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    champion = documents["champion"]
    champion["adapter_sha256"] = manifest["package_sha256"]
    fixed_contract_set = champion["fixed_contract_set"]
    assert isinstance(fixed_contract_set, dict)
    fixed_contract_set["config_sha256"] = manifest["config_sha256"]


def test_vertical_slice_is_byte_deterministic_and_manifest_binds_output(
    tmp_path: Path,
) -> None:
    first = build_fixture_run()
    second = build_fixture_run()

    assert canonical_json_bytes(first.production_signal) == canonical_json_bytes(
        second.production_signal
    )
    assert canonical_json_bytes(first.runtime_manifest) == canonical_json_bytes(
        second.runtime_manifest
    )
    assert first.production_signal["state"] == "ACTIVE"
    assert first.production_signal["target_positions"] == [
        {"symbol": "000001.SZ", "target_weight": 0.15}
    ]
    fixed_contract_set = load_documents()["champion"]["fixed_contract_set"]
    assert isinstance(fixed_contract_set, dict)
    promotion_evidence = load_documents()["champion"]["promotion_evidence"]
    assert isinstance(promotion_evidence, dict)
    assert first.production_signal["contract_set"]["code_sha256"] == (
        fixed_contract_set["code_sha256"]
    )
    assert first.production_signal["contract_set"]["config_sha256"] == (
        fixed_contract_set["config_sha256"]
    )
    adapter_package = ADAPTER_ROOT / "fixture-reference-adapter/v1"
    assert first.production_signal["contract_set"]["code_sha256"] == hashlib.sha256(
        (adapter_package / "adapter.py").read_bytes()
    ).hexdigest()
    assert first.production_signal["contract_set"]["config_sha256"] == hashlib.sha256(
        (adapter_package / "config.json").read_bytes()
    ).hexdigest()
    assert (
        first.production_signal["contract_set"]["dataset_manifest_sha256"]
        == promotion_evidence["dataset_manifest_sha256"]
    )
    assert (
        first.production_signal["contract_set"]["dataset_snapshot_sha256"]
        == promotion_evidence["dataset_snapshot_sha256"]
    )
    assert (
        first.production_signal["contract_set"]["universe_snapshot_sha256"]
        == promotion_evidence["universe_snapshot_sha256"]
    )
    runtime_inputs = {
        item["artifact_id"]: item["sha256"]
        for item in first.runtime_manifest["inputs"]
    }
    assert (
        first.production_signal["contract_set"]["dataset_manifest_sha256"]
        == runtime_inputs["dataset-manifest"]
    )
    assert (
        first.production_signal["contract_set"]["dataset_snapshot_sha256"]
        == runtime_inputs["dataset-snapshot"]
    )
    assert (
        first.production_signal["contract_set"]["universe_snapshot_sha256"]
        == runtime_inputs["universe-snapshot"]
    )
    assert (
        runtime_inputs["strategy-adapter-package"]
        == load_documents()["champion"]["adapter_sha256"]
    )
    assert runtime_inputs["lockfile"] == LOCKFILE_SHA256
    assert first.runtime_manifest["git_sha"] == DEPLOYMENT_GIT_SHA

    output_dir = tmp_path / "fixture-run"
    publish_run(output_dir=output_dir, artifacts=first)
    signal_bytes = (output_dir / "production-signal.json").read_bytes()
    manifest = json.loads((output_dir / "runtime-manifest.json").read_text())
    assert hashlib.sha256(signal_bytes).hexdigest() == manifest["outputs"][0]["sha256"]
    assert not (tmp_path / ".fixture-run.tmp").exists()

    with pytest.raises(FileExistsError, match="already exists"):
        publish_run(output_dir=output_dir, artifacts=first)


def test_deployment_git_sha_must_match_checked_out_repository() -> None:
    with pytest.raises(ValueError, match="does not match"):
        build_fixture_run(deployment_git_sha="0" * 40)


def test_dataset_content_hash_is_verified_before_strategy_runs(tmp_path: Path) -> None:
    copied_fixture = tmp_path / "vertical-slice"
    shutil.copytree(FIXTURE_ROOT, copied_fixture)
    (copied_fixture / "daily-bars.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_fixture_run(dataset_root=copied_fixture)


def test_dataset_row_count_is_verified_before_strategy_runs() -> None:
    documents = load_documents()
    manifest = documents["dataset-manifest"]
    files = manifest["files"]
    assert isinstance(files, list)
    files[0]["row_count"] = 999
    bind_champion_to_documents(documents)

    with pytest.raises(ValueError, match="row count mismatch"):
        build_fixture_run(documents=documents)


def test_duplicate_daily_bar_primary_key_is_rejected(tmp_path: Path) -> None:
    raw_bars = load_json(FIXTURE_ROOT / "daily-bars.json")
    assert isinstance(raw_bars, list)
    copied_fixture = tmp_path / "vertical-slice"
    shutil.copytree(FIXTURE_ROOT, copied_fixture)
    documents = load_documents()
    write_fixture_dataset(
        dataset_root=copied_fixture,
        rows=[*raw_bars, dict(raw_bars[0])],
        documents=documents,
        manifest_as_of=date(2026, 7, 29),
    )

    with pytest.raises(ValueError, match="duplicate daily bar primary key"):
        build_fixture_run(
            dataset_root=copied_fixture,
            documents=documents,
        )


def test_dataset_path_escape_is_rejected(tmp_path: Path) -> None:
    copied_fixture = tmp_path / "vertical-slice"
    shutil.copytree(FIXTURE_ROOT, copied_fixture)
    documents = load_documents()
    manifest = documents["dataset-manifest"]
    files = manifest["files"]
    assert isinstance(files, list)
    files[0]["path"] = "../outside.json"
    (tmp_path / "outside.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes dataset root"):
        load_dataset_snapshot(
            dataset_manifest=manifest,
            dataset_root=copied_fixture,
            as_of=date(2026, 7, 29),
        )


def test_dataset_symlink_is_rejected(tmp_path: Path) -> None:
    copied_fixture = tmp_path / "vertical-slice"
    shutil.copytree(FIXTURE_ROOT, copied_fixture)
    target = tmp_path / "outside.json"
    target.write_bytes((copied_fixture / "daily-bars.json").read_bytes())
    (copied_fixture / "daily-bars.json").unlink()
    (copied_fixture / "daily-bars.json").symlink_to(target)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        build_fixture_run(dataset_root=copied_fixture)


def test_production_entry_cannot_accept_an_arbitrary_strategy_object() -> None:
    parameters = inspect.signature(build_run).parameters

    assert "strategy" not in parameters
    assert "git_sha" not in parameters
    assert "lockfile_sha256" not in parameters


def test_portfolio_guard_rejects_loaded_target_outside_pit_universe(
    tmp_path: Path,
) -> None:
    copied_adapters = tmp_path / "approved-adapters"
    shutil.copytree(ADAPTER_ROOT, copied_adapters)
    documents = load_documents()
    rewrite_adapter_config(
        adapter_root=copied_adapters,
        documents=documents,
        forced_symbol="600000.SH",
    )

    with pytest.raises(ValueError, match="ineligible"):
        build_fixture_run(
            adapter_root=copied_adapters,
            documents=documents,
        )


def test_duplicate_pit_universe_symbol_is_rejected() -> None:
    documents = load_documents()
    universe = documents["universe"]
    members = universe["members"]
    assert isinstance(members, list)
    duplicate = dict(members[0])
    duplicate["reason_codes"] = ["DUPLICATE_FIXTURE"]
    members.append(duplicate)
    bind_champion_to_documents(documents)

    with pytest.raises(ValueError, match="duplicate symbol"):
        build_fixture_run(documents=documents)


def test_target_requires_cost_segment_for_signal_date() -> None:
    documents = load_documents()
    cost_model = documents["cost-model"]
    segments = cost_model["segments"]
    assert isinstance(segments, list)
    segments[-1]["effective_to"] = "2024-01-01"
    bind_champion_to_documents(documents)

    with pytest.raises(ValueError, match="match exactly one segment"):
        build_fixture_run(documents=documents)


def mutate_bound_contract(
    documents: dict[str, dict[str, object]],
    *,
    contract_id: str,
) -> None:
    document = documents[contract_id]
    if contract_id == "cost-model":
        commission = document["commission"]
        assert isinstance(commission, dict)
        commission["rate"] = 0.00031
    elif contract_id == "market-rules":
        segments = document["segments"]
        assert isinstance(segments, list)
        segments[0]["lot_size"] = 200
    elif contract_id == "execution-policy":
        document["slippage_bps"] = 11
    elif contract_id == "portfolio-risk":
        document["max_positions"] = 9
    else:
        raise AssertionError(f"unsupported contract fixture: {contract_id}")


@pytest.mark.parametrize("contract_id", sorted(FIXED_CONTRACT_FIELDS))
def test_champion_contract_drift_holds_verified_previous_target(
    contract_id: str,
) -> None:
    previous = build_fixture_run().production_signal
    documents = load_documents()
    mutate_bound_contract(documents, contract_id=contract_id)

    result = build_fixture_run(
        documents=documents,
        previous_signal=previous,
        previous_signal_sha256=canonical_json_sha256(previous),
    )

    assert result.production_signal["state"] == "HOLD"
    assert result.production_signal["target_positions"] == previous["target_positions"]
    expected_reason = f"CHAMPION_{contract_id.replace('-', '_').upper()}_MISMATCH"
    assert expected_reason in result.production_signal["reason_codes"]


def test_champion_lockfile_drift_holds_verified_previous_target() -> None:
    previous = build_fixture_run().production_signal
    documents = load_documents()
    fixed_contract_set = documents["champion"]["fixed_contract_set"]
    assert isinstance(fixed_contract_set, dict)
    fixed_contract_set["lockfile_sha256"] = "4" * 64

    result = build_fixture_run(
        documents=documents,
        previous_signal=previous,
        previous_signal_sha256=canonical_json_sha256(previous),
    )

    assert result.production_signal["state"] == "HOLD"
    assert result.production_signal["target_positions"] == previous["target_positions"]
    assert "CHAMPION_LOCKFILE_MISMATCH" in result.production_signal["reason_codes"]


def test_champion_requires_complete_fixed_contract_set() -> None:
    documents = load_documents()
    fixed_contract_set = documents["champion"]["fixed_contract_set"]
    assert isinstance(fixed_contract_set, dict)
    fixed_contract_set.pop("portfolio_risk_sha256")

    with pytest.raises(ValidationError, match="required property"):
        build_fixture_run(documents=documents)


@pytest.mark.parametrize(
    ("adapter_field", "adapter_value"),
    [
        ("adapter_id", "different-adapter/v1"),
        ("adapter_sha256", "e" * 64),
    ],
)
def test_champion_rejects_declared_adapter_drift(
    adapter_field: str,
    adapter_value: str,
) -> None:
    previous = build_fixture_run().production_signal
    documents = load_documents()
    documents["champion"][adapter_field] = adapter_value

    result = build_fixture_run(
        documents=documents,
        previous_signal=previous,
        previous_signal_sha256=canonical_json_sha256(previous),
    )

    assert result.production_signal["state"] == "HOLD"
    assert result.production_signal["target_positions"] == previous["target_positions"]
    assert "ADAPTER_VERIFICATION_FAILED" in result.production_signal["reason_codes"]


def test_adapter_code_tampering_holds_without_execution(tmp_path: Path) -> None:
    previous = build_fixture_run().production_signal
    copied_adapters = tmp_path / "approved-adapters"
    shutil.copytree(ADAPTER_ROOT, copied_adapters)
    code_path = (
        copied_adapters
        / "fixture-reference-adapter/v1/adapter.py"
    )
    code_path.write_text(
        code_path.read_text(encoding="utf-8")
        + "\nraise AssertionError('tampered code executed')\n",
        encoding="utf-8",
    )

    result = build_fixture_run(
        adapter_root=copied_adapters,
        previous_signal=previous,
        previous_signal_sha256=canonical_json_sha256(previous),
    )

    assert result.production_signal["state"] == "HOLD"
    assert "ADAPTER_VERIFICATION_FAILED" in result.production_signal["reason_codes"]


def test_no_champion_builds_initial_flat_without_loading_adapter(
    tmp_path: Path,
) -> None:
    documents = load_documents()
    documents.pop("champion")
    result = build_fixture_run(
        documents=documents,
        adapter_root=tmp_path / "missing-adapters",
    )

    assert result.production_signal["state"] == "FLAT"
    assert result.production_signal["champion"] is None
    assert result.production_signal["target_positions"] == []
    assert result.production_signal["reason_codes"] == [
        "CHAMPION_NEVER_ACTIVATED"
    ]


def test_missing_current_champion_holds_last_verified_target() -> None:
    previous = build_fixture_run().production_signal
    documents = load_documents()
    documents.pop("champion")

    result = build_fixture_run(
        documents=documents,
        previous_signal=previous,
        previous_signal_sha256=canonical_json_sha256(previous),
    )

    assert result.production_signal["state"] == "HOLD"
    assert result.production_signal["champion"] == previous["champion"]
    assert result.production_signal["target_positions"] == previous["target_positions"]
    assert result.production_signal["reason_codes"] == ["CHAMPION_MISSING"]


def test_stale_decision_data_holds_last_verified_target() -> None:
    previous = build_fixture_run().production_signal

    result = build_fixture_run(
        as_of=date(2026, 7, 30),
        generated_at=datetime(2026, 7, 30, 8, 10, tzinfo=UTC),
        previous_signal=previous,
        previous_signal_sha256=canonical_json_sha256(previous),
    )

    assert result.production_signal["state"] == "HOLD"
    assert result.production_signal["latest_complete_date"] == "2026-07-29"
    assert result.production_signal["target_positions"] == previous["target_positions"]
    assert "DATASET_STALE" in result.production_signal["reason_codes"]
    assert "UNIVERSE_STALE" in result.production_signal["reason_codes"]


def test_degraded_run_without_previous_signal_fails_closed() -> None:
    with pytest.raises(ValueError, match="HOLD requires a verified previous signal"):
        build_fixture_run(
            as_of=date(2026, 7, 30),
            generated_at=datetime(2026, 7, 30, 8, 10, tzinfo=UTC),
        )


def test_risk_actions_flow_through_production_pipeline() -> None:
    previous = build_fixture_run().production_signal
    previous_hash = canonical_json_sha256(previous)

    reduce_result = build_fixture_run(
        previous_signal=previous,
        previous_signal_sha256=previous_hash,
        risk_action=RiskAction.REDUCE,
    )
    flat_result = build_fixture_run(
        previous_signal=previous,
        previous_signal_sha256=previous_hash,
        risk_action=RiskAction.FLAT,
    )

    assert reduce_result.production_signal["state"] == "REDUCE_ONLY"
    assert reduce_result.production_signal["reason_codes"] == ["RISK_REDUCE"]
    assert flat_result.production_signal["state"] == "FLAT"
    assert flat_result.production_signal["target_positions"] == []
    assert flat_result.production_signal["reason_codes"] == ["RISK_FLAT"]


def test_future_rows_do_not_change_historical_snapshot_or_target(
    tmp_path: Path,
) -> None:
    raw_bars = load_json(FIXTURE_ROOT / "daily-bars.json")
    assert isinstance(raw_bars, list)
    baseline = build_fixture_run()
    copied_fixture = tmp_path / "vertical-slice"
    shutil.copytree(FIXTURE_ROOT, copied_fixture)
    documents = load_documents()
    with_future = [
        *raw_bars,
        {
            "symbol": "000001.SZ",
            "trade_date": "2026-07-30",
            "open": 1.1,
            "high": 1.2,
            "low": 0.9,
            "close": 1.0,
            "volume": 200000.0,
            "amount": 200000.0,
        },
    ]
    write_fixture_dataset(
        dataset_root=copied_fixture,
        rows=with_future,
        documents=documents,
        manifest_as_of=date(2026, 7, 30),
    )
    replayed = build_fixture_run(
        dataset_root=copied_fixture,
        documents=documents,
        as_of=date(2026, 7, 29),
    )
    baseline_snapshot = load_dataset_snapshot(
        dataset_manifest=load_documents()["dataset-manifest"],
        dataset_root=FIXTURE_ROOT,
        as_of=date(2026, 7, 29),
    )
    replayed_snapshot = load_dataset_snapshot(
        dataset_manifest=documents["dataset-manifest"],
        dataset_root=copied_fixture,
        as_of=date(2026, 7, 29),
    )

    assert replayed.production_signal["state"] == "ACTIVE"
    assert replayed_snapshot.records == baseline_snapshot.records
    assert (
        replayed.production_signal["target_positions"]
        == baseline.production_signal["target_positions"]
    )
    assert (
        replayed.production_signal["contract_set"]["dataset_snapshot_sha256"]
        == baseline.production_signal["contract_set"]["dataset_snapshot_sha256"]
    )
    assert (
        replayed.production_signal["contract_set"]["dataset_manifest_sha256"]
        != baseline.production_signal["contract_set"]["dataset_manifest_sha256"]
    )


def test_record_order_does_not_change_snapshot_or_target(tmp_path: Path) -> None:
    raw_bars = load_json(FIXTURE_ROOT / "daily-bars.json")
    assert isinstance(raw_bars, list)
    baseline = build_fixture_run()
    copied_fixture = tmp_path / "vertical-slice"
    shutil.copytree(FIXTURE_ROOT, copied_fixture)
    documents = load_documents()
    write_fixture_dataset(
        dataset_root=copied_fixture,
        rows=list(reversed(raw_bars)),
        documents=documents,
        manifest_as_of=date(2026, 7, 29),
    )

    replayed = build_fixture_run(
        dataset_root=copied_fixture,
        documents=documents,
    )

    assert (
        replayed.production_signal["contract_set"]["dataset_snapshot_sha256"]
        == baseline.production_signal["contract_set"]["dataset_snapshot_sha256"]
    )
    assert (
        replayed.production_signal["target_positions"]
        == baseline.production_signal["target_positions"]
    )


def test_loaded_snapshot_is_unchanged_after_source_file_mutation(
    tmp_path: Path,
) -> None:
    copied_fixture = tmp_path / "vertical-slice"
    shutil.copytree(FIXTURE_ROOT, copied_fixture)
    documents = load_documents()
    snapshot = load_dataset_snapshot(
        dataset_manifest=documents["dataset-manifest"],
        dataset_root=copied_fixture,
        as_of=date(2026, 7, 29),
    )
    expected_records = snapshot.records
    expected_hash = snapshot.snapshot_sha256

    (copied_fixture / "daily-bars.json").write_text("[]", encoding="utf-8")

    assert snapshot.records == expected_records
    assert snapshot.snapshot_sha256 == expected_hash


def test_dataset_artifact_is_read_once_for_hash_and_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes
    calls = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal calls
        if path.name == "daily-bars.json":
            calls += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    documents = load_documents()

    load_dataset_snapshot(
        dataset_manifest=documents["dataset-manifest"],
        dataset_root=FIXTURE_ROOT,
        as_of=date(2026, 7, 29),
    )

    assert calls == 1


@pytest.mark.parametrize(
    ("field_path", "value", "expected_reason"),
    [
        (
            ("dataset-manifest", "dataset_family_id"),
            "different-family/v1",
            "CHAMPION_DATASET_FAMILY_MISMATCH",
        ),
        (
            ("dataset-manifest", "normalization_version"),
            "different-normalizer/v1",
            "CHAMPION_NORMALIZATION_VERSION_MISMATCH",
        ),
        (
            ("dataset-manifest", "data_schema_sha256"),
            "e" * 64,
            "CHAMPION_DATA_SCHEMA_MISMATCH",
        ),
        (
            ("universe", "universe_policy_id"),
            "different-policy",
            "CHAMPION_UNIVERSE_POLICY_MISMATCH",
        ),
        (
            ("universe", "universe_policy_version"),
            "v2",
            "CHAMPION_UNIVERSE_POLICY_VERSION_MISMATCH",
        ),
    ],
)
def test_champion_rejects_incompatible_snapshot_policy(
    field_path: tuple[str, str],
    value: str,
    expected_reason: str,
) -> None:
    previous = build_fixture_run().production_signal
    documents = load_documents()
    documents[field_path[0]][field_path[1]] = value

    result = build_fixture_run(
        documents=documents,
        previous_signal=previous,
        previous_signal_sha256=canonical_json_sha256(previous),
    )

    assert result.production_signal["state"] == "HOLD"
    assert expected_reason in result.production_signal["reason_codes"]
