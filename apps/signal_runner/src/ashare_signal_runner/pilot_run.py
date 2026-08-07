"""Public command: run the pilot production chain.

Two invocation modes:

* ``--runtime-root`` (official live path): the active frozen champion is
  resolved exclusively through ``active-champion.json``; contracts and
  adapter are loaded from the immutable champion package. If no active
  champion exists the command fails closed with NO_ACTIVE_CHAMPION and
  never trains, promotes, or falls back to a test strategy.
* ``--contracts-dir``/``--adapter-root`` (internal/tests only): explicit
  paths, kept for the research-side integration tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from .pipeline import build_run, load_current_run, publish_run

ROOT = Path(__file__).resolve().parents[4]

EXIT_NO_ACTIVE_CHAMPION = 4


def _canonical_sha256(document: dict) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def resolve_active_champion(runtime_root: Path) -> tuple[Path, Path, dict, str]:
    """Return (contracts_dir, adapter_root, champion, champion_id) from the pointer."""
    pointer_path = Path(runtime_root) / "active-champion.json"
    if not pointer_path.is_file():
        raise LookupError("NO_ACTIVE_CHAMPION")
    pointer = _load_json_object(pointer_path)
    champion_id = str(pointer["champion_id"])
    package_dir = Path(runtime_root) / "champions" / champion_id
    champion = _load_json_object(package_dir / "champion.json")
    if _canonical_sha256(champion) != str(pointer["champion_sha256"]):
        raise ValueError("active pointer champion hash does not match package")
    return package_dir / "contracts", package_dir / "adapter", champion, champion_id


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
    champion_path: Path | None = None,
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
    champion = _load_json_object(
        Path(champion_path) if champion_path is not None else contracts_dir / "champion.json"
    )
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
    parser.add_argument("--runtime-root", default="")
    parser.add_argument("--contracts-dir", default="")
    parser.add_argument("--adapter-root", default="")
    parser.add_argument("--champion", default="")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--head-path", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--signal-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args(argv)

    champion_id = None
    if args.runtime_root:
        if args.contracts_dir or args.adapter_root:
            print("runtime-root cannot be combined with explicit champion paths")
            return 2
        try:
            contracts_dir, adapter_root, _champion, champion_id = resolve_active_champion(
                Path(args.runtime_root)
            )
        except LookupError:
            json.dump({"error": "NO_ACTIVE_CHAMPION"}, sys.stdout)
            sys.stdout.write("\n")
            return EXIT_NO_ACTIVE_CHAMPION
        champion_path = (
            Path(args.runtime_root) / "champions" / str(champion_id) / "champion.json"
        )
    elif args.contracts_dir and args.adapter_root:
        contracts_dir = Path(args.contracts_dir)
        adapter_root = Path(args.adapter_root)
        champion_path = (
            Path(args.champion) if args.champion else contracts_dir / "champion.json"
        )
    else:
        print("provide --runtime-root or both --contracts-dir and --adapter-root")
        return 2

    generated_at = datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    signal = run_pilot(
        contracts_dir=contracts_dir,
        adapter_root=adapter_root,
        dataset_root=Path(args.dataset_root),
        runs_root=Path(args.runs_root),
        head_path=Path(args.head_path),
        as_of=date.fromisoformat(args.as_of),
        generated_at=generated_at,
        signal_id=args.signal_id,
        run_id=args.run_id,
        deployment_git_sha=args.git_sha,
        champion_path=champion_path,
    )
    json.dump(
        {
            "state": signal["state"],
            "sequence": signal["sequence"],
            "as_of": signal["as_of"],
            "champion_id": champion_id,
            "champion_sha256": signal["contract_set"]["champion_sha256"],
            "dataset_snapshot_sha256": signal["contract_set"]["dataset_snapshot_sha256"],
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
