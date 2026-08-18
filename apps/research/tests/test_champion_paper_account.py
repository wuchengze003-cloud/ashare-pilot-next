from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ashare_research_app.champion import Fit, paper_backtest

ROOT = Path(__file__).resolve().parents[3]


def _contract(name: str) -> dict:
    return json.loads(
        (ROOT / "contracts" / "examples" / f"{name}.example.json").read_text(encoding="utf-8")
    )


def test_suspended_holding_keeps_its_last_observable_value() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    rows: list[dict[str, object]] = []
    for trade_date in dates:
        symbols = ["600000.SH", "000001.SZ"]
        if trade_date == pd.Timestamp("2024-01-04"):
            symbols = ["000001.SZ"]
        for symbol in symbols:
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "open": 10.0,
                    "close": 10.0,
                    "circ_mv": 3_000_000.0,
                }
            )
    panel = pd.DataFrame(rows)
    fit = Fit(
        predictions={
            dates[0]: (
                np.array(["600000.SH", "000001.SZ"]),
                np.array([2.0, 1.0]),
            )
        },
        last_model=None,
        model_evolution=[],
    )

    nav, turnover, _ = paper_backtest(
        panel,
        fit,
        pd.Series(dtype=float),
        cost_model=_contract("cost-model"),
        market_rules=_contract("market-rules"),
        execution_policy=_contract("execution-policy"),
    )

    assert len(nav) == 2
    assert len(turnover) == 2
    assert nav.iloc[1] == pytest.approx(nav.iloc[0])
