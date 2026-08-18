"""Explicit human-approved Champion activation command."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .promotion import activate_champion, canonical_json_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Activate one reviewed Champion package")
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--champion-id", required=True)
    parser.add_argument("--expected-champion-sha256", required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--activated-at", required=True)
    args = parser.parse_args(argv)
    if not args.approval_id.strip() or len(args.approval_id) > 120:
        raise ValueError("approval-id must contain 1 to 120 characters")
    activated_at = datetime.fromisoformat(args.activated_at.replace("Z", "+00:00"))
    if activated_at.tzinfo is None or activated_at.utcoffset() is None:
        raise ValueError("activated-at must be timezone-aware")
    package_dir = args.runtime_root / "champions" / args.champion_id
    champion_path = package_dir / "champion.json"
    if champion_path.is_symlink() or not champion_path.is_file():
        raise ValueError("reviewed Champion document is unavailable")
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    actual_sha256 = canonical_json_sha256(champion)
    if actual_sha256 != args.expected_champion_sha256:
        raise ValueError("reviewed Champion hash does not match the package")
    pointer = activate_champion(
        runtime_root=args.runtime_root,
        champion_id=args.champion_id,
        activated_at=activated_at.astimezone(UTC),
        approval_id=args.approval_id,
    )
    activation_id = str(pointer["activation_id"])
    receipt_path = args.runtime_root / "activation-receipts" / f"{activation_id}.json"
    print(receipt_path.read_text(encoding="ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
