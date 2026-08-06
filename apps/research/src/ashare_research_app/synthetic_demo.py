"""Public command: write a deterministic synthetic dataset for fresh clones.

The generated dataset is explicitly labeled synthetic and is only for
demonstrating the vertical pipeline without credentials or runtime
artifacts. Fixed calendar and seeds keep it byte-reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from ashare_quant_core import DAILY_BAR_SCHEMA_ID, DAILY_BAR_SCHEMA_SHA256

SYMBOLS = (
    "000001.SZ",
    "000333.SZ",
    "000858.SZ",
    "002415.SZ",
    "002594.SZ",
    "300014.SZ",
    "300059.SZ",
    "300750.SZ",
    "600036.SH",
    "600519.SH",
    "601318.SH",
    "601398.SH",
    "688008.SH",
    "688111.SH",
    "688981.SH",
    "600466.SH",
    "000732.SZ",
    "000666.SZ",
    "600530.SH",
    "002564.SZ",
)
FIXED_END = date(2023, 12, 29)


def trading_days(end: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(days))


def generate_rows(*, days: int, seed: int) -> list[dict[str, object]]:
    calendar = trading_days(FIXED_END, days)
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        generator = random.Random(f"{seed}:{symbol}")
        price = 8.0 + generator.random() * 60.0
        halted_until: date | None = None
        for index, day in enumerate(calendar):
            if symbol == "600530.SH" and 40 <= index <= 70:
                halted_until = day
                continue
            if halted_until is not None and day <= halted_until:
                continue
            drift = generator.uniform(-0.035, 0.036)
            open_price = price
            close_price = max(1.0, price * (1 + drift))
            high = max(open_price, close_price) * (1 + generator.uniform(0, 0.012))
            low = min(open_price, close_price) * (1 - generator.uniform(0, 0.012))
            volume_lots = generator.uniform(20000, 200000)
            amount_thousand = volume_lots * 100 * (open_price + close_price) / 2 / 1000
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": day.isoformat(),
                    "open": round(open_price, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close_price, 4),
                    "volume": round(volume_lots * 100, 0),
                    "amount": round(amount_thousand * 1000, 0),
                }
            )
            price = close_price
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write deterministic synthetic dataset")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--days", type=int, default=170)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    dataset_id = "synthetic-demo-v1"
    dataset_dir = out_dir / "datasets" / dataset_id
    manifest_dir = out_dir / "manifests"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(days=args.days, seed=args.seed)
    content = (json.dumps(rows, ensure_ascii=True, indent=2) + "\n").encode()
    (dataset_dir / "daily-bars.json").write_bytes(content)
    trade_dates = [str(row["trade_date"]) for row in rows]
    manifest = {
        "contract_id": "dataset-manifest",
        "schema_version": "2.0.0",
        "dataset_id": dataset_id,
        "dataset_family_id": "synthetic-demo/v1",
        "dataset_kind": "normalized",
        "data_schema_id": DAILY_BAR_SCHEMA_ID,
        "data_schema_sha256": DAILY_BAR_SCHEMA_SHA256,
        "normalization_version": "synthetic-normalizer/v1",
        "as_of": max(trade_dates),
        "generated_at": f"{max(trade_dates)}T00:30:00Z",
        "source": "synthetic-demo",
        "source_version": "v1",
        "parent_manifest_sha256": None,
        "quality_status": "pass",
        "quality_reasons": [],
        "files": [
            {
                "path": "daily-bars.json",
                "sha256": hashlib.sha256(content).hexdigest(),
                "row_count": len(rows),
                "file_size_bytes": len(content),
                "min_trade_date": min(trade_dates),
                "max_trade_date": max(trade_dates),
            }
        ],
    }
    manifest_path = manifest_dir / f"{dataset_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    json.dump(
        {
            "manifest_path": str(manifest_path),
            "dataset_dir": str(dataset_dir),
            "as_of": max(trade_dates),
        },
        sys.stdout,
        ensure_ascii=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
