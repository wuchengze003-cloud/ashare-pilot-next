"""Public command: inference-only scoring with the active frozen champion.

Loads the active champion's model bundle and scores the current
point-in-time snapshot. This command never trains, fits, tunes,
promotes, or creates champions; it is the live-side ranking producer.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .baseline_model import MultiHorizonModel
from .datasets import load_snapshot
from .features import FEATURE_NAMES, build_feature_panel

EXIT_NO_ACTIVE_CHAMPION = 4
ADAPTER_ID = "ml-baseline-adapter/v1"


def _load_json_object(path: Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"document must be an object: {path}")
    return document


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


def score_frozen_champion(
    *,
    runtime_root: Path,
    manifest: dict,
    dataset_root: Path,
    top_k: int,
) -> dict:
    pointer_path = Path(runtime_root) / "active-champion.json"
    if not pointer_path.is_file():
        raise LookupError("NO_ACTIVE_CHAMPION")
    pointer = _load_json_object(pointer_path)
    package_dir = Path(runtime_root) / "champions" / str(pointer["champion_id"])
    receipt = _load_json_object(package_dir / "promotion-receipt.json")
    config = _load_json_object(
        package_dir / "adapter" / Path(*ADAPTER_ID.split("/")) / "config.json"
    )
    bundle_bytes = (package_dir / "model" / "model.bundle").read_bytes()
    model = MultiHorizonModel.from_bundle_bytes(bundle_bytes)

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
                "rank_strength": round(
                    1.0 - (rank - 1) / max(number_of_symbols, 1), 4
                ),
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
                _rank_correlation(
                    [float(v) for v in matrix[:, column]], [float(s) for s in scores]
                )
            )
            feature_weights.append({"name": name, "weight": round(weight, 6)})
        feature_weights.sort(key=lambda item: (-item["weight"], item["name"]))

    return {
        "report_id": "pilot-frozen-inference/v1",
        "as_of": as_of_text,
        "dataset_id": snapshot.dataset_id,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "champion_id": str(pointer["champion_id"]),
        "model_bundle_sha256": str(receipt["model_bundle_sha256"]),
        "latest_signal_date": signal_date.isoformat(),
        "recommendations": recommendations,
        "feature_weights": feature_weights[:3],
        "top_k": top_k,
        "per_weight": float(config["per_weight"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inference-only frozen champion scoring")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        report = score_frozen_champion(
            runtime_root=Path(args.runtime_root),
            manifest=_load_json_object(Path(args.dataset_manifest)),
            dataset_root=Path(args.dataset_root),
            top_k=args.top_k,
        )
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
