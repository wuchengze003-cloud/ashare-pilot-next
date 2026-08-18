"""Champion strategy: GBDT cross-sectional model, concentrated portfolio.

A data-driven multi-factor model (NOT a hand-picked factor set). A gradient-
boosted tree learns from a 27-feature cross-section, retrained every 20 trading
days on the trailing 120 days, and picks the top-10 names for a human-sized
concentrated book.

Current research configuration:
    - slow alpha: label horizon 60 days (survives the 2023 bear via timing)
    - concentrated: 10 names, ~10% each, ~1 trade/day (human-executable)
    - realistic fills: next-open + slippage, hold-to-exit (stop/profit/signal)
    - point-in-time inputs: delayed fields use their availability dates
    - mature-label rolling fit: every refit excludes labels ending after the
      refit date, while the latest feature row can still be scored
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from ashare_quant_core import (
    DailyBarView,
    ExecutionDay,
    SimulatedPortfolioState,
    classify_board,
    execute_buy,
    execute_sell,
    mark_to_market,
    parse_market_rules,
    settle_t_plus_one,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from ashare_research_app.factors import add_factors
from ashare_research_app.market_data import load_full_market

MAIN_BOARD = ("000", "001", "002", "600", "601", "603", "605")
TRADING_DAYS = 252

# Wide feature pool (27 features). Data speaks: the tree picks what matters.
FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_60_20",
    "ma5_20",
    "ma20_60",
    "vol_20d",
    "vol_60d",
    "high_low",
    "amt_20d",
    "vol_avg_20d",
    "size_log",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "ep",
    "turnover_rate",
    "volume_ratio",
    "main_flow_prev",
    "lg_net_prev",
    "margin_chg_prev",
    "winner_rate",
    "holder_chg",
]

# Research configuration. Any production activation still requires a separate,
# immutable evaluation and promotion decision.
LABEL_HORIZON = 60
# Kept as an independently evaluated research horizon. It must never be spliced
# into the 60-day model's recent history merely to fill unavailable labels.
TAIL_LABEL = 20
WINDOW = 120
REFIT_EVERY = 20
TOP_K = 10
EXIT_BUFFER = 4
MIN_HOLD = 20
MAX_BUY = 2
MAX_SELL = 2
STOP_LOSS = 0.06
TAKE_PROFIT = 0.25
TIMING_BAND = 0.03
SEGMENTS = [
    ("train", "2023-01-01", "2024-06-30"),
    ("valid", "2024-07-01", "2025-06-30"),
    ("test", "2025-07-01", "2026-08-31"),
]

# Human-readable Chinese labels for each feature (shown in the dashboard).
FEATURE_LABELS = {
    "ret_1d": "1日涨幅",
    "ret_5d": "5日涨幅",
    "ret_10d": "10日涨幅",
    "ret_20d": "20日涨幅",
    "ret_60d": "60日涨幅",
    "ret_120d": "120日涨幅",
    "ret_60_20": "60/20日动量",
    "ma5_20": "5/20均线",
    "ma20_60": "20/60均线",
    "vol_20d": "20日波动",
    "vol_60d": "60日波动",
    "high_low": "日内振幅",
    "amt_20d": "20日成交额",
    "vol_avg_20d": "20日量能",
    "size_log": "市值规模",
    "pe_ttm": "市盈率",
    "pb": "市净率",
    "ps_ttm": "市销率",
    "dv_ttm": "股息率",
    "ep": "盈利收益率",
    "turnover_rate": "换手率",
    "volume_ratio": "量比",
    "main_flow_prev": "主力净流入",
    "lg_net_prev": "大单净额",
    "margin_chg_prev": "融资环比",
    "winner_rate": "获利盘比例",
    "holder_chg": "股东户数环比",
}

# Key explainable features shown per stock (attribution), with a friendly unit.
# Unit semantics: "pct" = decimal -> percent (*100); "pct_raw" = already percent;
# "num" = raw number; "mktcap" = ten-thousand CNY -> hundred-million CNY (*1e-4).
KEY_FEATURES = [
    ("ret_60d", "60日涨幅", "pct"),
    ("ret_20d", "20日涨幅", "pct"),
    ("turnover_rate", "换手率", "pct_raw"),
    ("volume_ratio", "量比", "num"),
    ("circ_mv", "流通市值", "mktcap"),
    ("pe_ttm", "市盈率", "num"),
    ("dv_ttm", "股息率", "pct_raw"),
    ("winner_rate", "获利盘", "pct_raw"),
]


@dataclass
class Fit:
    predictions: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]
    last_model: HistGradientBoostingRegressor | None
    model_evolution: list[dict] = None  # type: ignore[assignment]
    last_refit_date: pd.Timestamp | None = None


def _cross_sectional_ic(
    seg: pd.DataFrame, model: HistGradientBoostingRegressor, feats: list[str]
) -> float:
    """Mean daily cross-sectional IC (Spearman rank corr of model score vs label).

    IC measures the model's *ranking* skill per day and is unaffected by trade
    frequency, so it is the right per-refit "is this model any good" score —
    unlike a short-window Sharpe which is pure noise on a few dozen days.
    """
    ics: list[float] = []
    for _, day in seg.groupby("trade_date", sort=False):
        pred = model.predict(day[feats].to_numpy())
        pr = pd.Series(pred).rank()
        lr = pd.Series(day["_label"].to_numpy()).rank()
        ic = pr.corr(lr)
        if pd.notna(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def _load_by_date(dirpath: Path) -> pd.DataFrame:
    """Load a per-trade-date interface (one JSON per date) into one frame."""
    frames = []
    for f in sorted(dirpath.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame(j["items"], columns=j["fields"]))
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    return out


def _load_per_symbol(dirpath: Path) -> pd.DataFrame:
    """Load a per-symbol interface (one JSON per ts_code) into one frame."""
    frames = []
    for f in sorted(dirpath.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame(j["items"], columns=j["fields"]))
    out = pd.concat(frames, ignore_index=True)
    for col in ("trade_date", "end_date", "ann_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _load_symbol_names(root: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for p in sorted(root.glob("[0-9]*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        names[str(doc.get("symbol"))] = str(doc.get("name", ""))
    return names


def _load_industry(root: Path) -> dict[str, str]:
    """Shenwan Level-1 industry mapping (ts_code -> industry name).

    Sourced from index_member_all.
    """
    p = root / "runtime/sw_industry.json"
    if not p.exists():
        return {}
    return {k: str(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}


def _merge_holder_number_as_of(
    panel: pd.DataFrame,
    holder_numbers: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest holder count that was public on each signal date.

    ``end_date`` is the reporting cut-off, not the publication date.  A holder
    count therefore becomes observable only on ``ann_date``.  Rows without a
    publication date, or rows claiming publication before their own reporting
    cut-off, are rejected rather than backfilled from ``end_date``.

    The vendor source has a publication date but no publication timestamp.  To
    avoid assuming an intraday availability time, a disclosure becomes usable
    only on the first signal date strictly after ``ann_date``.  The selected
    report and announcement dates stay in the panel as audit fields.
    """
    required = ("ts_code", "ann_date", "end_date", "holder_num")
    missing = [field for field in required if field not in holder_numbers.columns]
    if missing:
        raise ValueError("stk_holdernumber is missing point-in-time fields: " + ", ".join(missing))

    events = holder_numbers[list(required)].rename(columns={"ts_code": "symbol"}).copy()
    events["ann_date"] = pd.to_datetime(events["ann_date"], errors="coerce")
    events["end_date"] = pd.to_datetime(events["end_date"], errors="coerce")
    events["holder_num"] = pd.to_numeric(events["holder_num"], errors="coerce")
    events = events.dropna(subset=["symbol", "ann_date", "end_date", "holder_num"])
    events = events[events["ann_date"] >= events["end_date"]]
    events = events.sort_values(["symbol", "ann_date", "end_date"]).drop_duplicates(
        ["symbol", "ann_date"], keep="last"
    )

    parts: list[pd.DataFrame] = []
    for symbol, group in panel.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=False):
        history = events[events["symbol"] == symbol].sort_values("ann_date")
        group = group.copy()
        group["holder_num"] = np.nan
        group["holder_report_end_date"] = pd.NaT
        group["holder_ann_date"] = pd.NaT
        if not history.empty:
            positions = (
                history["ann_date"]
                .to_numpy()
                .searchsorted(group["trade_date"].to_numpy(), side="left")
                - 1
            )
            visible = positions >= 0
            if visible.any():
                selected = history.iloc[positions[visible]]
                group.loc[visible, "holder_num"] = selected["holder_num"].to_numpy()
                group.loc[visible, "holder_report_end_date"] = selected["end_date"].to_numpy()
                group.loc[visible, "holder_ann_date"] = selected["ann_date"].to_numpy()
        parts.append(group)

    if not parts:
        result = panel.copy()
        result["holder_num"] = np.nan
        result["holder_report_end_date"] = pd.NaT
        result["holder_ann_date"] = pd.NaT
        return result
    return pd.concat(parts, ignore_index=True)


def _load_index(root: Path) -> pd.Series:
    """Load Shanghai Composite (000001.SH) daily closes, normalized to 1.0."""
    p = root / "runtime/index/000001.SH.json"
    if not p.exists():
        return pd.Series(dtype=float)
    doc = json.loads(p.read_text(encoding="utf-8"))
    df = pd.DataFrame(doc["items"], columns=doc["fields"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")
    close = df.set_index("trade_date")["close"].astype(float)
    return close / close.iloc[0]


def build_wide_panel(root: Path) -> pd.DataFrame:
    """Build the 27-feature panel from full-market bars + alternative data."""
    alt = root / "runtime/alt-data"
    df = load_full_market(root / "runtime/full-market")
    df = df[df["symbol"].str[:3].isin(MAIN_BOARD)]
    panel = add_factors(df)

    g = panel.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=False)
    close = g["close"]
    panel["ret_10d"] = close.transform(lambda s: s.pct_change(10))
    panel["ret_60d"] = close.transform(lambda s: s.pct_change(60))
    panel["ret_120d"] = close.transform(lambda s: s.pct_change(120))
    panel["ret_60_20"] = close.transform(lambda s: s.shift(20) / s.shift(60) - 1.0)
    panel["ma5"] = close.transform(lambda s: s.rolling(5).mean())
    panel["ma20"] = close.transform(lambda s: s.rolling(20).mean())
    panel["ma60"] = close.transform(lambda s: s.rolling(60).mean())
    panel["ma5_20"] = panel["ma5"] / panel["ma20"] - 1.0
    panel["ma20_60"] = panel["ma20"] / panel["ma60"] - 1.0
    panel["vol_60d"] = g["ret_1d"].transform(lambda s: s.rolling(60).std())
    panel["high_low"] = (panel["high"] - panel["low"]) / panel["close"]

    db = _load_by_date(alt / "daily_basic")[
        [
            "ts_code",
            "trade_date",
            "circ_mv",
            "pe_ttm",
            "pb",
            "ps_ttm",
            "dv_ttm",
            "turnover_rate",
            "volume_ratio",
        ]
    ].rename(columns={"ts_code": "symbol"})
    panel = panel.merge(db, on=["symbol", "trade_date"], how="left")
    panel["size_log"] = -np.log(panel["circ_mv"].where(panel["circ_mv"] > 0))
    panel["ep"] = 1.0 / panel["pe_ttm"].where(panel["pe_ttm"] > 0)

    mf = _load_by_date(alt / "moneyflow")[
        ["ts_code", "trade_date", "net_mf_amount", "buy_lg_amount", "sell_lg_amount"]
    ].rename(columns={"ts_code": "symbol"})
    panel = panel.merge(mf, on=["symbol", "trade_date"], how="left")
    panel["main_flow"] = (
        panel["net_mf_amount"] * 10000.0 / panel["amount"].where(panel["amount"] > 0)
    )
    panel["lg_net"] = (
        (panel["buy_lg_amount"] - panel["sell_lg_amount"])
        * 10000.0
        / panel["amount"].where(panel["amount"] > 0)
    )

    cyq = _load_per_symbol(alt / "cyq_perf")[["ts_code", "trade_date", "winner_rate"]].rename(
        columns={"ts_code": "symbol"}
    )
    panel = panel.merge(cyq, on=["symbol", "trade_date"], how="left")

    mg = (
        _load_by_date(alt / "margin_detail")[["ts_code", "trade_date", "rzye"]]
        .rename(columns={"ts_code": "symbol"})
        .sort_values(["symbol", "trade_date"])
    )
    mg["margin_chg"] = mg.groupby("symbol")["rzye"].pct_change()
    panel = panel.merge(
        mg[["symbol", "trade_date", "margin_chg"]], on=["symbol", "trade_date"], how="left"
    )

    holder_numbers = _load_per_symbol(alt / "stk_holdernumber")
    panel = _merge_holder_number_as_of(panel, holder_numbers)
    panel["holder_chg"] = panel.groupby("symbol")["holder_num"].pct_change()

    # no look-ahead: moneyflow & margin are post-close / T+1, use yesterday
    for c in ("main_flow", "lg_net", "margin_chg"):
        panel[f"{c}_prev"] = panel.groupby("symbol")[c].shift(1)
    for c in (
        "size_log",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "dv_ttm",
        "turnover_rate",
        "volume_ratio",
        "winner_rate",
        "ep",
    ):
        panel[c] = panel.groupby("symbol")[c].ffill()
    return panel.sort_values(["symbol", "trade_date"])


def hysteresis(market_ret: pd.Series, ma_window: int = 20, band: float = TIMING_BAND) -> pd.Series:
    """Market-timing scale with a hysteresis band (no 0/1 oscillation).

    Above +band -> invested (1.0); below -band -> flat (0.0); in between -> keep
    previous state. Replaces the old binary cross that caused 41 cashouts.
    """
    nav = (1.0 + market_ret).cumprod()
    ma = nav.rolling(ma_window).mean()
    dev = nav / ma - 1.0
    scale = np.ones(len(dev))
    state = 1.0
    for i in range(len(dev)):
        d = dev.iloc[i]
        if d > band:
            state = 1.0
        elif d < -band:
            state = 0.0
        scale[i] = state
    return pd.Series(scale, index=nav.index, name="position_scale")


def _prepare_forward_labels(
    panel: pd.DataFrame,
    feats: list[str],
    *,
    horizon: int,
) -> pd.DataFrame:
    """Create forward labels together with the date on which each label matures."""
    if horizon < 1:
        raise ValueError("label horizon must be positive")
    prepared = panel.sort_values(["symbol", "trade_date"]).copy()
    grouped = prepared.groupby("symbol", sort=False)
    prepared["_label_end_date"] = grouped["trade_date"].shift(-horizon)
    prepared["_fwd"] = grouped["close"].shift(-horizon) / prepared["close"] - 1.0
    prepared = prepared.dropna(subset=feats + ["_fwd", "_label_end_date"])
    prepared = prepared[np.isfinite(prepared[feats].to_numpy()).all(axis=1)]
    if (prepared["_label_end_date"] <= prepared["trade_date"]).any():
        raise ValueError("forward label must end after its feature date")
    return prepared


def _mature_relative_labels(
    prepared: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Create cross-sectional labels from peers mature at one refit cutoff."""
    mature = prepared[prepared["_label_end_date"] <= cutoff].copy()
    mature["_label"] = mature.groupby("trade_date")["_fwd"].transform(
        lambda values: values - values.mean()
    )
    return mature


def _new_gbdt() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=42,
    )


def rolling_gbdt(
    prepared: pd.DataFrame,
    full_panel: pd.DataFrame,
    feats: list[str],
    *,
    label_horizon: int = LABEL_HORIZON,
    window: int = WINDOW,
    refit: int = REFIT_EVERY,
) -> Fit:
    """Leak-free rolling fit using only labels mature on each refit date.

    The evaluation model is fitted on the earlier portion of the window and
    scored on an unseen validation tail. A separate production model is then
    fitted on the complete mature-label window and used until the next refit.
    """
    required = {"trade_date", "symbol", "_fwd", "_label_end_date", *feats}
    missing = sorted(required - set(prepared.columns))
    if missing:
        raise ValueError("prepared panel is missing: " + ", ".join(missing))
    if window < 3:
        raise ValueError("rolling window must contain at least three dates")
    if refit < 1:
        raise ValueError("refit interval must be positive")

    full_dates = [pd.Timestamp(day) for day in sorted(full_panel["trade_date"].unique())]
    if not full_dates:
        raise ValueError("full panel contains no trading dates")
    full_by_date = {
        pd.Timestamp(day): group for day, group in full_panel.groupby("trade_date", sort=False)
    }

    refit_dates: list[pd.Timestamp] = []
    last_refit_index: int | None = None
    for index, current in enumerate(full_dates):
        mature = _mature_relative_labels(prepared, cutoff=current)
        if mature["trade_date"].nunique() < window:
            continue
        if last_refit_index is None or index - last_refit_index >= refit:
            refit_dates.append(current)
            last_refit_index = index
    if not refit_dates:
        raise ValueError("no refit date has enough mature labels")

    predictions: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    model_evolution: list[dict] = []
    last_model: HistGradientBoostingRegressor | None = None
    validation_days = min(refit, max(1, window // 5))

    for index, current in enumerate(refit_dates):
        mature = _mature_relative_labels(prepared, cutoff=current)
        mature_dates = [pd.Timestamp(day) for day in sorted(mature["trade_date"].unique())]
        window_dates = mature_dates[-window:]
        window_rows = mature[mature["trade_date"].isin(window_dates)]
        evaluation_train_dates = window_dates[:-validation_days]
        validation_dates = window_dates[-validation_days:]
        evaluation_train = window_rows[window_rows["trade_date"].isin(evaluation_train_dates)]
        validation = window_rows[window_rows["trade_date"].isin(validation_dates)]

        evaluation_model = _new_gbdt()
        evaluation_model.fit(
            evaluation_train[feats].to_numpy(), evaluation_train["_label"].to_numpy()
        )
        train_ic = _cross_sectional_ic(evaluation_train, evaluation_model, feats)
        valid_ic = _cross_sectional_ic(validation, evaluation_model, feats)
        gap = 0.5 * max(0.0, train_ic - valid_ic)

        production_model = _new_gbdt()
        production_model.fit(window_rows[feats].to_numpy(), window_rows["_label"].to_numpy())
        importance_rows = (
            validation.sample(8000, random_state=42) if len(validation) > 8000 else validation
        )
        importance_result = permutation_importance(
            evaluation_model,
            importance_rows[feats].to_numpy(),
            importance_rows["_label"].to_numpy(),
            n_repeats=1,
            random_state=42,
        )
        importance = [
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "importance": round(float(value), 6),
            }
            for feature, value in zip(feats, importance_result.importances_mean, strict=True)
        ]

        next_refit = refit_dates[index + 1] if index + 1 < len(refit_dates) else full_dates[-1]
        max_label_end = pd.Timestamp(window_rows["_label_end_date"].max())
        if max_label_end > current:
            raise ValueError("training label escaped the refit cutoff")
        model_evolution.append(
            {
                "index": index,
                "label_horizon": label_horizon,
                "refit_date": str(current.date()),
                "start_date": str(current.date()),
                "end_date": str(next_refit.date()),
                "train_start": str(evaluation_train_dates[0].date()),
                "train_end": str(evaluation_train_dates[-1].date()),
                "valid_start": str(validation_dates[0].date()),
                "valid_end": str(validation_dates[-1].date()),
                "production_train_start": str(window_dates[0].date()),
                "production_train_end": str(window_dates[-1].date()),
                "max_label_end": str(max_label_end.date()),
                "n_train_days": len(evaluation_train_dates),
                "n_valid_days": len(validation_dates),
                "train_ic": round(train_ic, 4),
                "valid_ic": round(valid_ic, 4),
                "metric_applies_to": "evaluation_model",
                "production_model_fit_scope": "mature_train_plus_validation",
                "gap_penalty": round(gap, 4),
                "score": round(valid_ic - gap, 4),
                "feature_importance": importance,
            }
        )

        start = full_dates.index(current)
        stop = full_dates.index(next_refit) if index + 1 < len(refit_dates) else len(full_dates)
        for day in full_dates[start:stop]:
            day_frame = full_by_date[day]
            valid_features = day_frame[feats].notna().all(axis=1)
            valid_features &= np.isfinite(day_frame[feats].to_numpy()).all(axis=1)
            day_frame = day_frame[valid_features]
            if not day_frame.empty:
                predictions[day] = (
                    day_frame["symbol"].to_numpy(),
                    production_model.predict(day_frame[feats].to_numpy()),
                )
        last_model = production_model

    return Fit(
        predictions=predictions,
        last_model=last_model,
        model_evolution=model_evolution,
        last_refit_date=refit_dates[-1],
    )


def paper_backtest(
    panel: pd.DataFrame,
    fit: Fit,
    scale: pd.Series,
    *,
    cost_model: dict,
    market_rules: dict,
    execution_policy: dict,
    initial_cash: float = 1_000_000.0,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Replay research targets with the authoritative ``quant_core`` semantics."""
    dates = sorted(panel["trade_date"].unique())
    open_p = panel.pivot_table(index="trade_date", columns="symbol", values="open")
    close_p = panel.pivot_table(index="trade_date", columns="symbol", values="close")
    previous_close_p = close_p.ffill().shift(1)
    rules = parse_market_rules(market_rules)
    portfolio = SimulatedPortfolioState(
        cash=Decimal(str(initial_cash)),
        holdings={},
    )
    buy_dates: dict[str, date] = {}
    buy_indices: dict[str, int] = {}
    last_prices: dict[str, float] = {}
    exit_queue: set[str] = set()
    navs: list[float] = []
    turnovers: list[float] = []
    trades: list[dict] = []

    def trade_document(trade, *, reason: str) -> dict:
        return {
            "date": trade.trade_date.isoformat(),
            "symbol": trade.symbol,
            "side": trade.side,
            "shares": trade.shares,
            "price": round(float(trade.price), 4),
            "gross_amount": float(trade.gross_amount),
            "total_cost": float(trade.cost.total),
            "reason": reason,
        }

    for i in range(1, len(dates) - 1):
        sd, buy_day = dates[i - 1], dates[i]
        trade_date = pd.Timestamp(buy_day).date()
        portfolio = settle_t_plus_one(
            portfolio,
            trade_date=trade_date,
            buy_dates=buy_dates,
        )
        sc = 1.0
        if sd in scale.index:
            sc = float(scale.loc[sd])
        in_pool: set[str] = set()
        in_buffer: set[str] = set()
        if sd in fit.predictions:
            syms, pred = fit.predictions[sd]
            rk = pd.Series(pred, index=syms).rank(ascending=False)
            in_pool = set(rk[rk <= TOP_K].index)
            in_buffer = set(rk[rk <= TOP_K + EXIT_BUFFER].index)
        open_row = open_p.loc[buy_day]
        close_row = close_p.loc[buy_day]
        previous_row = previous_close_p.loc[buy_day]
        bars = {
            str(symbol): DailyBarView(open=float(open_price), close=float(close_row[symbol]))
            for symbol, open_price in open_row.items()
            if pd.notna(open_price) and pd.notna(close_row.get(symbol, np.nan))
        }
        previous_closes = {
            str(symbol): float(price) for symbol, price in previous_row.items() if pd.notna(price)
        }
        day = ExecutionDay(
            trade_date=trade_date,
            bars=bars,
            previous_closes=previous_closes,
            slippage_bps=int(execution_policy["slippage_bps"]),
        )
        reference_prices = {
            symbol: previous_closes.get(symbol, last_prices.get(symbol))
            for symbol in portfolio.holdings
        }
        if any(price is None for price in reference_prices.values()):
            raise ValueError("research replay is missing a holding reference price")
        total_assets_before = (
            mark_to_market(
                portfolio,
                prices={symbol: float(price) for symbol, price in reference_prices.items()},
            )
            if portfolio.holdings
            else portfolio.cash
        )
        traded_gross = Decimal("0")
        if sc == 0.0:
            exit_queue |= set(portfolio.holdings)

        order: list[tuple[int, str]] = []
        for sym, holding in portfolio.holdings.items():
            bar = bars.get(sym)
            price = bar.open if bar is not None else np.nan
            if pd.isna(price):
                priority = 9
            elif price <= float(holding.avg_cost) * (1 - STOP_LOSS):
                priority = 0
            elif price >= float(holding.avg_cost) * (1 + TAKE_PROFIT):
                priority = 1
            elif sym in exit_queue:
                priority = 2
            elif i - buy_indices.get(sym, i) >= MIN_HOLD and sym not in in_buffer:
                priority = 3
            else:
                priority = 9
            order.append((priority, sym))

        sells_today = 0
        for priority, sym in sorted(order):
            if priority == 9 or sells_today >= MAX_SELL:
                continue
            holding = portfolio.holdings[sym]
            bar = bars.get(sym)
            assert bar is not None
            if bar.open <= float(holding.avg_cost) * (1 - STOP_LOSS):
                reason = "stop"
            elif bar.open >= float(holding.avg_cost) * (1 + TAKE_PROFIT):
                reason = "profit"
            elif sym in exit_queue:
                reason = "cashout"
            else:
                reason = "signal"
            portfolio, trade, _skip = execute_sell(
                state=portfolio,
                day=day,
                symbol=sym,
                rules=rules,
                cost_model=cost_model,
                reason=reason.upper(),
            )
            if trade is not None:
                trades.append(trade_document(trade, reason=reason))
                traded_gross += trade.gross_amount
                sells_today += 1
                if sym not in portfolio.holdings:
                    buy_dates.pop(sym, None)
                    buy_indices.pop(sym, None)
                    exit_queue.discard(sym)

        if sc > 0.0 and sd in fit.predictions:
            budget = total_assets_before / Decimal(TOP_K)
            syms, pred = fit.predictions[sd]
            buys_today = 0
            for sym in syms[np.argsort(-pred)]:
                sym = str(sym)
                if buys_today >= MAX_BUY or len(portfolio.holdings) >= TOP_K:
                    break
                if sym not in in_pool or sym in portfolio.holdings:
                    continue
                reference = previous_closes.get(sym)
                if reference is None:
                    continue
                lot = rules[classify_board(sym)].lot_size
                requested = int(budget / Decimal(str(reference)) / lot) * lot
                if requested < lot:
                    continue
                portfolio, trade, _skip, bought_on = execute_buy(
                    state=portfolio,
                    day=day,
                    symbol=sym,
                    requested_shares=requested,
                    rules=rules,
                    cost_model=cost_model,
                    reason="SIGNAL",
                )
                if trade is not None and bought_on is not None:
                    trades.append(trade_document(trade, reason="signal"))
                    traded_gross += trade.gross_amount
                    buy_dates[sym] = bought_on
                    buy_indices[sym] = i
                    buys_today += 1

        mark_prices: dict[str, float] = {}
        for sym in portfolio.holdings:
            bar = bars.get(sym)
            if bar is not None:
                last_prices[sym] = bar.close
            elif sym not in last_prices:
                last_prices[sym] = float(reference_prices[sym])
            mark_prices[sym] = last_prices[sym]
        total_assets = (
            mark_to_market(portfolio, prices=mark_prices) if portfolio.holdings else portfolio.cash
        )
        navs.append(float(total_assets))
        turnovers.append(float(traded_gross / total_assets_before) if total_assets_before else 0.0)
    index = pd.DatetimeIndex(dates[1:-1])
    return (
        pd.Series(navs, index=index, name="nav"),
        pd.Series(turnovers, index=index, name="turnover"),
        trades,
    )


def _metrics(nav: pd.Series, turnover: pd.Series) -> dict:
    n = len(nav)
    r = nav.pct_change(fill_method=None).dropna()
    ann = float(nav.iloc[-1] ** (TRADING_DAYS / n) - 1.0)
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    mdd = float((nav / nav.cummax() - 1.0).min())
    underwater = (nav < nav.cummax()).astype(int)
    run = max_run = 0
    for u in underwater:
        run = run + 1 if u else 0
        max_run = max(max_run, run)
    signs = (r < 0).astype(int)
    losing_run = max_losing = 0
    for s in signs:
        losing_run = losing_run + 1 if s else 0
        max_losing = max(max_losing, losing_run)
    return {
        "annual_return": ann,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "avg_turnover": float(turnover.mean()),
        "n_days": n,
        "final_nav": float(nav.iloc[-1]),
        "max_daily_loss": float(r.min()),
        "max_daily_gain": float(r.max()),
        "longest_underwater_days": max_run,
        "longest_losing_streak": max_losing,
    }


def _segment(nav: pd.Series, t0: str, t1: str) -> dict:
    seg = nav[(nav.index >= pd.Timestamp(t0)) & (nav.index <= pd.Timestamp(t1))]
    if len(seg) < 3:
        return {}
    r = seg.pct_change(fill_method=None).dropna()
    ann = float(seg.iloc[-1] / seg.iloc[0]) ** (TRADING_DAYS / len(seg)) - 1.0
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    return {"annual_return": ann, "sharpe": sharpe}


def generate(root: Path) -> dict:
    """Run the champion and return everything the dashboard needs."""
    panel = build_wide_panel(root)
    prepared = _prepare_forward_labels(panel, FEATURES, horizon=LABEL_HORIZON)

    market_ret = panel.groupby("trade_date")["ret_1d"].mean()
    scale = hysteresis(market_ret)
    fit = rolling_gbdt(
        prepared,
        panel,
        FEATURES,
        label_horizon=LABEL_HORIZON,
    )

    contract_root = root / "contracts" / "examples"
    cost_model = json.loads((contract_root / "cost-model.example.json").read_text(encoding="utf-8"))
    market_rules = json.loads(
        (contract_root / "market-rules.example.json").read_text(encoding="utf-8")
    )
    execution_policy = json.loads(
        (contract_root / "execution-policy.example.json").read_text(encoding="utf-8")
    )
    nav, turnover, trades = paper_backtest(
        panel,
        fit,
        scale,
        cost_model=cost_model,
        market_rules=market_rules,
        execution_policy=execution_policy,
    )

    # benchmark: Shanghai Composite index, normalized to 1.0 at the strategy's
    # first NAV date so the two curves start aligned.
    bench = _load_index(root).reindex(nav.index).ffill()
    bench_nav = bench / bench.iloc[0]

    # latest signal: last model on the FULL panel's latest bar (factors-only)
    names = _load_symbol_names(root / "runtime/full-market")
    industry = _load_industry(root)
    latest_date = panel["trade_date"].max()
    latest = panel[panel["trade_date"] == latest_date].copy()
    latest = latest[latest[FEATURES].notna().all(axis=1)]
    signals: list[dict] = []
    if fit.last_model is not None and len(latest):
        pred = fit.last_model.predict(latest[FEATURES].to_numpy())
        order = np.argsort(-pred)
        top_syms = latest["symbol"].to_numpy()[order][:TOP_K]
        top_scores = pred[order][:TOP_K]
        mv = latest.set_index("symbol")["circ_mv"]
        # cross-sectional percentile reference (all names on the latest date)
        ref = panel[panel["trade_date"] == latest_date]
        for s, sc in zip(top_syms, top_scores, strict=True):
            row = ref[ref["symbol"] == s]
            attribution = []
            if len(row):
                r = row.iloc[0]
                for f, label, unit in KEY_FEATURES:
                    val = r.get(f, np.nan)
                    if pd.isna(val):
                        continue
                    col = ref[f]
                    if unit == "pct":
                        val = float(val) * 100.0
                    elif unit == "mktcap":
                        val = float(val) * 1e-4  # ten-thousand CNY -> hundred-million CNY
                    pctl = float((col <= r[f]).mean()) if col.notna().sum() > 0 else np.nan
                    attribution.append(
                        {
                            "feature": f,
                            "label": label,
                            "unit": unit,
                            "value": round(val, 2),
                            "percentile": round(pctl, 3),
                        }
                    )
            close_price = float(r["close"]) if len(row) else float(mv.get(s, np.nan))
            # 1M CNY initial capital, equal-weight across TOP_K names, board-lot 100
            budget = 1_000_000.0 / TOP_K
            shares = max(100, int(budget / close_price / 100) * 100) if close_price > 0 else 0
            signals.append(
                {
                    "symbol": str(s),
                    "name": names.get(str(s), ""),
                    "industry": industry.get(str(s), ""),
                    "score": float(round(float(sc), 4)),
                    # ten-thousand CNY -> hundred-million CNY
                    "market_cap": float(mv.get(s, np.nan)) * 1e-4,
                    "weight": round(1.0 / TOP_K, 4),
                    "close": round(close_price, 2),
                    "shares": shares,
                    "amount": round(shares * close_price, 2),
                    "attribution": attribution,
                }
            )

    feature_importance = sorted(
        (fit.model_evolution or [{}])[-1].get("feature_importance", []),
        key=lambda item: -float(item["importance"]),
    )

    daily = nav.pct_change(fill_method=None).dropna()
    yearly = []
    for yr in sorted({d.year for d in daily.index}):
        seg = daily[daily.index.year == yr]
        if len(seg) >= 5:
            yearly.append({"year": yr, "return": float((1 + seg).prod() - 1.0)})

    # Exit-reason distribution (for the "why do we sell" explanation).
    # The dashboard renderer maps these stable codes to Chinese display labels.
    reason_labels = {"stop": "STOP", "profit": "PROFIT", "signal": "SIGNAL", "cashout": "CASHOUT"}
    reason_counts: dict[str, int] = {}
    for t in trades:
        if t["side"] == "sell":
            reason_counts[t["reason"]] = reason_counts.get(t["reason"], 0) + 1
    exit_reasons = [
        {"reason": k, "label": reason_labels.get(k, k), "count": v}
        for k, v in sorted(reason_counts.items(), key=lambda x: -x[1])
    ]

    # Rolling-refit timestamps + factor IC timeline. Every diagnostic uses the
    # same mature-label cutoff as the model fitted on that date.
    refit_dates = [me["refit_date"] for me in (fit.model_evolution or [])]
    factor_ic_timeline = []
    for me in fit.model_evolution or []:
        cur = pd.Timestamp(me["refit_date"])
        mature = _mature_relative_labels(prepared, cutoff=cur)
        mature_dates = sorted(mature["trade_date"].unique())[-WINDOW:]
        training_rows = mature[mature["trade_date"].isin(mature_dates)]
        ics = []
        for f in FEATURES:
            f_rank = training_rows.groupby("trade_date", sort=False)[f].rank()
            label_rank = training_rows.groupby("trade_date", sort=False)["_label"].rank()
            ic = f_rank.corr(label_rank)
            ics.append(
                {
                    "feature": f,
                    "label": FEATURE_LABELS.get(f, f),
                    "ic": round(float(ic), 4) if pd.notna(ic) else 0.0,
                }
            )
        factor_ic_timeline.append(
            {
                "date": me["refit_date"],
                "factors": sorted(ics, key=lambda x: -abs(x["ic"])),
            }
        )

    # cashout history: merge consecutive cashout days (one flat event drains the
    # book over a few days) into single events, separated by a 5-day gap.
    cashout_days = sorted({t["date"] for t in trades if t["reason"] == "cashout"})
    cashout_history: list[dict] = []
    for d in cashout_days:
        if cashout_history:
            gap = (pd.Timestamp(d) - pd.Timestamp(cashout_history[-1]["date"])).days
            if gap <= 5:
                continue
        cashout_history.append({"date": d, "event": "CASHOUT"})

    # current holdings: rebuild from the paper-trade log (buy minus sell).
    held: dict[str, dict] = {}
    for t in trades:
        if t["side"] == "buy":
            held[t["symbol"]] = {"price": t["price"], "shares": t.get("shares", 100)}
        else:
            held.pop(t["symbol"], None)
    # mark to market at the latest close for a correct position ratio
    latest_close = panel[panel["trade_date"] == latest_date].set_index("symbol")["close"]
    current_holdings = [
        {
            "symbol": s,
            "name": names.get(s, ""),
            "buy_price": round(h["price"], 2),
            "shares": h["shares"],
            "last_close": round(float(latest_close.get(s, h["price"])), 2),
        }
        for s, h in held.items()
    ]
    total_market_value = sum(h["last_close"] * h["shares"] for h in current_holdings)
    total_position = total_market_value / float(nav.iloc[-1])

    # closed trades: pair buy->sell for the per-trade win/loss table
    buy_log: dict[str, dict] = {}
    closed_trades: list[dict] = []
    for t in trades:
        if t["side"] == "buy":
            buy_log[t["symbol"]] = {
                "buy_price": t["price"],
                "buy_date": t["date"],
                "shares": t.get("shares", 0),
            }
        else:
            b = buy_log.pop(t["symbol"], None)
            if b is not None:
                pnl = (t["price"] - b["buy_price"]) / b["buy_price"]
                days = (pd.Timestamp(t["date"]) - pd.Timestamp(b["buy_date"])).days
                closed_trades.append(
                    {
                        "symbol": t["symbol"],
                        "name": names.get(t["symbol"], ""),
                        "buy_date": b["buy_date"],
                        "buy_price": round(b["buy_price"], 2),
                        "sell_date": t["date"],
                        "sell_price": round(t["price"], 2),
                        "pnl": round(pnl, 4),
                        "days": days,
                        "reason": t["reason"],
                    }
                )

    # historical daily signals: top-k for every trading day (history view).
    # A day where the market-timing scale is 0 (flat) shows NO names — the
    # Strategy was out of the market, so the signal must report a flat book,
    # not list 10 stocks it would never have held.
    daily_signals: list[dict] = []
    for d in sorted(fit.predictions.keys()):
        flat = d in scale.index and float(scale.loc[d]) == 0.0
        syms, pred = fit.predictions[d]
        order = np.argsort(-pred)
        k = min(TOP_K, len(order))
        day_frame = panel[panel["trade_date"] == d]
        close_map = dict(zip(day_frame["symbol"], day_frame["close"], strict=False))
        mv_map = dict(zip(day_frame["symbol"], day_frame["circ_mv"], strict=False))
        sig_list = []
        if not flat:
            for j in range(k):
                s = str(syms[order[j]])
                sig_list.append(
                    {
                        "symbol": s,
                        "name": names.get(s, ""),
                        "industry": industry.get(s, ""),
                        "score": round(float(pred[order[j]]), 4),
                        "close": round(float(close_map.get(s, float("nan"))), 2)
                        if pd.notna(close_map.get(s, float("nan")))
                        else None,
                        "market_cap": round(float(mv_map.get(s, float("nan"))) * 1e-4, 1)
                        if pd.notna(mv_map.get(s, float("nan")))
                        else None,
                    }
                )
        daily_signals.append(
            {
                "date": str(d.date()),
                "flat": flat,
                "signals": sig_list,
            }
        )

    return {
        "generated_at": str(latest_date.date()),
        "signal_date": str(latest_date.date()),
        "strategy": "ashare-rolling-gbdt-research-v2",
        "model_note": "mature-label rolling GBDT",
        "history_mode": "research_reconstruction",
        "config": {
            "top_k": TOP_K,
            "label_horizon": LABEL_HORIZON,
            "research_candidate_horizons": [TAIL_LABEL],
            "min_hold": MIN_HOLD,
            "max_buy": MAX_BUY,
            "max_sell": MAX_SELL,
            "stop_loss": STOP_LOSS,
            "take_profit": TAKE_PROFIT,
            "cost_model_id": cost_model["model_id"],
            "execution_policy_id": execution_policy["policy_id"],
            "slippage_bps": execution_policy["slippage_bps"],
            "n_features": len(FEATURES),
            "initial_cash": 1_000_000.0,
        },
        "signals": signals,
        "current_holdings": current_holdings,
        "closed_trades": closed_trades,
        "daily_signals": daily_signals,
        "total_position": round(total_position, 4),
        "refit_dates": refit_dates,
        "factor_ic_timeline": factor_ic_timeline,
        "model_evolution": fit.model_evolution or [],
        "cashout_history": cashout_history,
        "feature_importance": feature_importance,
        "exit_reasons": exit_reasons,
        "nav": [
            {"date": str(d.date()), "nav": round(float(v) / 1_000_000.0, 4)} for d, v in nav.items()
        ],
        "benchmark": [
            {"date": str(d.date()), "nav": round(float(v), 4)} for d, v in bench_nav.items()
        ],
        "metrics": _metrics(nav / 1_000_000.0, turnover),
        "segments": {name: _segment(nav / 1_000_000.0, t0, t1) for name, t0, t1 in SEGMENTS},
        "segment_semantics": "walk_forward_reporting_periods_not_fixed_holdouts",
        "yearly": yearly,
        "trades": trades,
    }


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = generate(root)
    out_dir = root / "runtime/dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "champion.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    m = out["metrics"]
    print(f"signal_date={out['signal_date']}  top_k={len(out['signals'])}")
    print(
        f"annual {m['annual_return']:.2%} sharpe {m['sharpe']:.2f} "
        f"maxdd {m['max_drawdown']:.2%} turnover {m['avg_turnover']:.2%}"
    )
    for seg, v in out["segments"].items():
        print(f"  {seg:6s} annual {v.get('annual_return', 0):.2%} sharpe {v.get('sharpe', 0):.2f}")
    print(f"written -> {out_dir / 'champion.json'}")
