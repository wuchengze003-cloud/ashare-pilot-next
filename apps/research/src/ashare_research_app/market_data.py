"""Load the full-market daily-bar dataset collected from mootdx.

Each symbol is stored as one JSON object (``{code}.json``) whose ``data`` array
holds the daily bars. This module flattens them into a single pandas DataFrame
for cross-sectional research. It is a research convenience loader, not a
Dataset Manifest consumer — the collected data is raw vendor export under
``runtime/``, outside the immutable dataset pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COLUMNS = ("symbol", "trade_date", "open", "high", "low", "close", "volume", "amount")


def load_full_market(root: Path) -> pd.DataFrame:
    """Flatten all ``{code}.json`` files under ``root`` into one DataFrame.

    Returns columns ``symbol`` (e.g. "600519.SH") and the bar fields, sorted by
    ``(trade_date, symbol)`` with ``trade_date`` parsed as ``datetime64[ns]``.
    """
    root = Path(root)
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("[0-9]*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or not doc.get("data"):
            continue
        symbol = str(doc["symbol"])
        frame = pd.DataFrame(doc["data"])
        frame.insert(0, "symbol", symbol)
        frames.append(frame)
    if not frames:
        raise ValueError(f"no full-market data found under {root}")
    result = pd.concat(frames, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result[list(COLUMNS)]
    return result.sort_values(["trade_date", "symbol"], ignore_index=True)
