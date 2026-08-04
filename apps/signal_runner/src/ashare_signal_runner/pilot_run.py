"""Public command: run the pilot production chain with explicit arguments.

Loads champion, contracts, and the immutable dataset; builds one
production signal through the standard pipeline; publishes it under an
explicit runs root and advances the signal head atomically.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from .pipeline import build_run, load_current_run, publish_run

ROOT = Path(__file__).resolve().parents[4]


def _load_json_object(path: Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"contract document must be an object: {path}")
    return document


def load_schemas() -> dict[str, dict]:
    registry = _load_json_object(ROOT / "contracts/registry.json")
    schemas: dict[str, dict] = {}
    for entry in registry["contracts"]:
        schema = _load_json_object(ROOT / "contracts" / entry["schema"])
        schemas[str(entry["contract_id"])] = schema
    return schemas


def run_pilot(
    *,
    contracts_dir: Path,
    adapter_root: Path,
    dataset_root: Path,
    runs_root: Path,
    head_path: Path,
    as_of: date,
    generated_at: datetime,
    signal_id: str,
    run_id: str,
    deployment_git_sha: str,
) -> dict:
    contracts_dir = Path(contracts_dir)
    documents = {
        contract_id: _load_json_object(contracts_dir / f"{contract_id}.json")
        for contract_id in (
            "cost-model",
            "execution-policy",
            "market-rules",
            "portfolio-risk",
            "universe",
            "dataset-manifest",
        )
    }
    champion = _load_json_object(contracts_dir / "champion.json")
    schemas = load_schemas()

    previous_signal = None
    previous_head = None
    current = load_current_run(
        runs_root=Path(runs_root),
        head_path=Path(head_path),
        required_as_of=as_of,
        schemas=schemas,
    )
    if current is not None:
        previous_signal = current.production_signal
        previous_head = current.signal_head

    artifacts = build_run(
        as_of=as_of,
        generated_at=generated_at,
        signal_id=signal_id,
        run_id=run_id,
        deployment_git_sha=deployment_git_sha,
        repository_root=ROOT,
        adapter_root=Path(adapter_root),
        dataset_root=Path(dataset_root),
        documents={**documents, "champion": champion},
        schemas=schemas,
        previous_signal=previous_signal,
        previous_head=previous_head,
    )
    publish_run(
        output_dir=Path(runs_root) / run_id,
        head_path=Path(head_path),
        artifacts=artifacts,
        schemas=schemas,
    )
    return artifacts.production_signal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pilot production chain")
    parser.add_argument("--contracts-dir", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--head-path", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--signal-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args(argv)

    generated_at = datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    signal = run_pilot(
        contracts_dir=Path(args.contracts_dir),
        adapter_root=Path(args.adapter_root),
        dataset_root=Path(args.dataset_root),
        runs_root=Path(args.runs_root),
        head_path=Path(args.head_path),
        as_of=date.fromisoformat(args.as_of),
        generated_at=generated_at,
        signal_id=args.signal_id,
        run_id=args.run_id,
        deployment_git_sha=args.git_sha,
    )
    json.dump(
        {
            "state": signal["state"],
            "sequence": signal["sequence"],
            "target_positions": signal["target_positions"],
            "reason_codes": signal["reason_codes"],
        },
        sys.stdout,
        ensure_ascii=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
