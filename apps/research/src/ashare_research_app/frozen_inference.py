"""Public command: inference-only scoring with the active frozen champion.

Loads the active champion's model bundle and scores the current
point-in-time snapshot, and derives the current-cycle universe from the
current dataset. This command never trains, fits, tunes, promotes, or
creates champions; it is the live-side ranking producer.

Strategy parameters (top_k / per_weight) are read from the frozen
champion's adapter config so a live caller cannot drift an activated
champion's semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .baseline_model import MultiHorizonModel
from .datasets import load_snapshot
from .features import FEATURE_NAMES, build_feature_panel
from .promotion import build_universe_document

EXIT_NO_ACTIVE_CHAMPION = 4
EXIT_TOP_K_OVERRIDE_REJECTED = 5


def _load_json_object(path: Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"document must be an object: {path}")
    return document


def _canonical_sha256(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _rank_correlation(x: list[float], y: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: (values[i], i))
        ranked = [0.0] * len(values)
        for position, index in enumerate(order):
            ranked[index] = float(position)
        return ranked

    rx, ry = ranks(x), ranks(y)
    n = len(rx)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return 0.0
    return covariance / (var_x * var_y) ** 0.5


def _load_frozen_package(runtime_root: Path) -> tuple[dict, dict, dict, bytes]:
    """Return (pointer, champion, config, bundle_bytes) after integrity checks.

    The trust chain flows from the activation pointer (outside the
    package) through receipt_sha256 → champion_sha256 → model bundle
    SHA → adapter config SHA.  Tampering any asset or hash that is
    anchored in the pointer is detected.
    """
    pointer_path = Path(runtime_root) / "active-champion.json"
    if not pointer_path.is_file():
        raise LookupError("NO_ACTIVE_CHAMPION")
    pointer = _load_json_object(pointer_path)
    package_dir = Path(runtime_root) / "champions" / str(pointer["champion_id"])
    champion = _load_json_object(package_dir / "champion.json")
    if _canonical_sha256(champion) != str(pointer["champion_sha256"]):
        raise ValueError("active pointer champion hash does not match package champion")
    receipt = _load_json_object(package_dir / "promotion-receipt.json")
    if _canonical_sha256(receipt) != str(pointer.get("receipt_sha256")):
        raise ValueError("active pointer receipt hash does not match package receipt")
    if str(receipt["champion_sha256"]) != _canonical_sha256(champion):
        raise ValueError("promotion receipt champion hash does not match champion")
    adapter_id = str(champion["adapter_id"])  # read from champion, not hardcoded
    config = _load_json_object(
        package_dir / "adapter" / Path(*adapter_id.split("/")) / "config.json"
    )
    bundle_bytes = (package_dir / "model" / "model.bundle").read_bytes()
    bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
    if bundle_sha != str(receipt["model_bundle_sha256"]):
        raise ValueError("model bundle hash does not match promotion receipt")
    if bundle_sha != str(config["model_bundle_sha256"]):
        raise ValueError("model bundle hash does not match adapter config")
    return pointer, champion, config, bundle_bytes


def score_frozen_champion(
    *,
    runtime_root: Path,
    manifest: dict,
    dataset_root: Path,
) -> dict:
    pointer, champion, config, bundle_bytes = _load_frozen_package(runtime_root)
    model = MultiHorizonModel.from_bundle_bytes(bundle_bytes)
    top_k = int(config["top_k"])

    as_of_text = str(manifest["as_of"])
    snapshot = load_snapshot(
        manifest=manifest,
        dataset_root=Path(dataset_root),
        as_of=datetime.strptime(as_of_text, "%Y-%m-%d").date(),
    )
    bars_by_symbol: dict[str, list] = {}
    for bar in snapshot.records:
        bars_by_symbol.setdefault(bar.symbol, []).append(bar)
    for bars in bars_by_symbol.values():
        bars.sort(key=lambda item: item.trade_date)
    signal_date = max(bar.trade_date for bar in snapshot.records)

    panel = build_feature_panel(snapshot, as_of=signal_date)
    rows = [row for row in panel if row.trade_date == signal_date]
    matrix = np.asarray([row.values for row in rows], dtype=float)
    scores = model.score(matrix)
    ranked = sorted(
        ((row.symbol, float(score)) for row, score in zip(rows, scores, strict=True)),
        key=lambda item: (-item[1], item[0]),
    )
    targets = [symbol for symbol, _ in ranked[:top_k]]
    number_of_symbols = len(ranked)

    recommendations = []
    for rank, (symbol, score) in enumerate(ranked, start=1):
        bars = bars_by_symbol[symbol]
        last_close = bars[-1].close
        risk_notes: list[str] = []
        if bars[-1].trade_date < signal_date:
            risk_notes.append("NO_BAR_ON_SIGNAL_DATE")
        recommendation = "BUY" if symbol in targets else "WATCH"
        recommendations.append(
            {
                "symbol": symbol,
                "rank": rank,
                "score": round(score, 6),
                "recommendation": recommendation,
                "rank_strength": round(1.0 - (rank - 1) / max(number_of_symbols, 1), 4),
                "price_band": (
                    {"low": round(last_close * 0.98, 4), "high": round(last_close, 4)}
                    if recommendation == "BUY"
                    else None
                ),
                "risk_notes": risk_notes,
            }
        )

    feature_weights = []
    if matrix.size:
        for column, name in enumerate(FEATURE_NAMES):
            weight = abs(
                _rank_correlation([float(v) for v in matrix[:, column]], [float(s) for s in scores])
            )
            feature_weights.append({"name": name, "weight": round(weight, 6)})
        feature_weights.sort(key=lambda item: (-item["weight"], item["name"]))

    return {
        "report_id": "pilot-frozen-inference/v1",
        "as_of": as_of_text,
        "dataset_id": snapshot.dataset_id,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "champion_id": str(pointer["champion_id"]),
        "model_bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "latest_signal_date": signal_date.isoformat(),
        "recommendations": recommendations,
        "feature_weights": feature_weights[:3],
        "top_k": top_k,
        "per_weight": float(config["per_weight"]),
    }


def derive_current_universe(
    *,
    runtime_root: Path,
    manifest: dict,
    dataset_root: Path,
    generated_at: str,
) -> dict:
    """Build the current-cycle universe from the current dataset only."""
    _pointer, _champion, _config, _bundle = _load_frozen_package(runtime_root)
    as_of_text = str(manifest["as_of"])
    snapshot = load_snapshot(
        manifest=manifest,
        dataset_root=Path(dataset_root),
        as_of=datetime.strptime(as_of_text, "%Y-%m-%d").date(),
    )
    return build_universe_document(
        symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        as_of=as_of_text,
        generated_at=generated_at,
        dataset_manifest=manifest,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inference-only frozen champion scoring")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--universe-out", default="")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "DO NOT USE. Strategy parameters are governed by the frozen "
            "champion's adapter config. Providing this flag causes a hard "
            "failure to prevent live override of an activated champion."
        ),
    )
    args = parser.parse_args(argv)
    if args.top_k is not None:
        json.dump(
            {"error": "TOP_K_OVERRIDE_REJECTED", "detail": "top_k is frozen in the champion"},
            sys.stdout,
        )
        sys.stdout.write("\n")
        return EXIT_TOP_K_OVERRIDE_REJECTED

    manifest = _load_json_object(Path(args.dataset_manifest))
    try:
        report = score_frozen_champion(
            runtime_root=Path(args.runtime_root),
            manifest=manifest,
            dataset_root=Path(args.dataset_root),
        )
        if args.universe_out:
            universe = derive_current_universe(
                runtime_root=Path(args.runtime_root),
                manifest=manifest,
                dataset_root=Path(args.dataset_root),
                generated_at=args.generated_at,
            )
            universe_path = Path(args.universe_out)
            universe_path.parent.mkdir(parents=True, exist_ok=True)
            universe_path.write_text(json.dumps(universe, ensure_ascii=True, indent=2) + "\n")
    except LookupError:
        json.dump({"error": "NO_ACTIVE_CHAMPION"}, sys.stdout)
        sys.stdout.write("\n")
        return EXIT_NO_ACTIVE_CHAMPION

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    temporary.replace(out_path)
    json.dump({"out": str(out_path), "as_of": report["as_of"]}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
