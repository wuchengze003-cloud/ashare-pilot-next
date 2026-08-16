"""Champion strategy: GBDT cross-sectional model, concentrated portfolio.

A data-driven multi-factor model (NOT a hand-picked factor set). A gradient-
boosted tree learns from a 27-feature cross-section, retrained every 20 trading
days on the trailing 120 days, and picks the top-10 names for a human-sized
concentrated book.

Key properties (all verified during the research race):
    - slow alpha: label horizon 60 days (survives the 2023 bear via timing)
    - concentrated: 10 names, ~10% each, ~1 trade/day (human-executable)
    - realistic fills: next-open + slippage, hold-to-exit (stop/profit/signal)
    - no look-ahead: moneyflow/margin shifted 1 day; prediction reuses the last
      trained model on the full panel's latest bar (prediction needs factors
      only, not future labels)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from ashare_research_app.factors import add_factors
from ashare_research_app.market_data import load_full_market

MAIN_BOARD = ("000", "001", "002", "600", "601", "603", "605")
TRADING_DAYS = 252

# Wide feature pool (27 features). Data speaks: the tree picks what matters.
FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    "ret_60_20", "ma5_20", "ma20_60", "vol_20d", "vol_60d", "high_low",
    "amt_20d", "vol_avg_20d", "size_log", "pe_ttm", "pb", "ps_ttm", "dv_ttm",
    "ep", "turnover_rate", "volume_ratio", "main_flow_prev", "lg_net_prev",
    "margin_chg_prev", "winner_rate", "holder_chg",
]

# Frozen champion configuration (from the parallel race, 2026-08-16).
# label60 beats label40/80; top10 has the best Sharpe (2.44) & smallest MDD
# (-14.7%) in the concentrated range, both human-executable (<=10 names).
LABEL_HORIZON = 60
TAIL_LABEL = 20  # tail transition model's horizon (last ~60 days); probed 2026-08-16
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
COST_PER_SIDE = 0.0015

SEGMENTS = [
    ("train", "2023-01-01", "2024-06-30"),
    ("valid", "2024-07-01", "2025-06-30"),
    ("test", "2025-07-01", "2026-08-31"),
]

# Human-readable Chinese labels for each feature (shown in the dashboard).
FEATURE_LABELS = {
    "ret_1d": "1日涨幅", "ret_5d": "5日涨幅", "ret_10d": "10日涨幅",
    "ret_20d": "20日涨幅", "ret_60d": "60日涨幅", "ret_120d": "120日涨幅",
    "ret_60_20": "60/20日动量", "ma5_20": "5/20均线", "ma20_60": "20/60均线",
    "vol_20d": "20日波动", "vol_60d": "60日波动", "high_low": "日内振幅",
    "amt_20d": "20日成交额", "vol_avg_20d": "20日量能",
    "size_log": "市值规模", "pe_ttm": "市盈率", "pb": "市净率",
    "ps_ttm": "市销率", "dv_ttm": "股息率", "ep": "盈利收益率",
    "turnover_rate": "换手率", "volume_ratio": "量比",
    "main_flow_prev": "主力净流入", "lg_net_prev": "大单净额",
    "margin_chg_prev": "融资环比", "winner_rate": "获利盘比例",
    "holder_chg": "股东户数环比",
}

# Key explainable features shown per stock (attribution), with a friendly unit.
# unit semantics: "pct" = decimal -> percent (*100); "pct_raw" = already percent;
# "num" = raw number; "mktcap" = 万元 -> 亿元 (*1e-4).
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


def _cross_sectional_ic(seg: pd.DataFrame, model: HistGradientBoostingRegressor,
                        feats: list[str]) -> float:
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
    """申万一级行业映射（ts_code -> industry name），采集自 index_member_all."""
    p = root / "runtime/sw_industry.json"
    if not p.exists():
        return {}
    return {k: str(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}


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
    names = _load_symbol_names(root / "runtime/full-market")
    st = {s for s, n in names.items() if "ST" in n.upper()}
    df = load_full_market(root / "runtime/full-market")
    df = df[~df["symbol"].isin(st)]
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
        ["ts_code", "trade_date", "circ_mv", "pe_ttm", "pb", "ps_ttm", "dv_ttm",
         "turnover_rate", "volume_ratio"]
    ].rename(columns={"ts_code": "symbol"})
    panel = panel.merge(db, on=["symbol", "trade_date"], how="left")
    panel["size_log"] = -np.log(panel["circ_mv"].where(panel["circ_mv"] > 0))
    panel["ep"] = 1.0 / panel["pe_ttm"].where(panel["pe_ttm"] > 0)

    mf = _load_by_date(alt / "moneyflow")[
        ["ts_code", "trade_date", "net_mf_amount", "buy_lg_amount", "sell_lg_amount"]
    ].rename(columns={"ts_code": "symbol"})
    panel = panel.merge(mf, on=["symbol", "trade_date"], how="left")
    panel["main_flow"] = panel["net_mf_amount"] * 10000.0 / panel["amount"].where(
        panel["amount"] > 0)
    panel["lg_net"] = (panel["buy_lg_amount"] - panel["sell_lg_amount"]) * 10000.0 / panel[
        "amount"].where(panel["amount"] > 0)

    cyq = _load_per_symbol(alt / "cyq_perf")[
        ["ts_code", "trade_date", "winner_rate"]].rename(columns={"ts_code": "symbol"})
    panel = panel.merge(cyq, on=["symbol", "trade_date"], how="left")

    mg = _load_by_date(alt / "margin_detail")[["ts_code", "trade_date", "rzye"]].rename(
        columns={"ts_code": "symbol"}).sort_values(["symbol", "trade_date"])
    mg["margin_chg"] = mg.groupby("symbol")["rzye"].pct_change()
    panel = panel.merge(mg[["symbol", "trade_date", "margin_chg"]],
                        on=["symbol", "trade_date"], how="left")

    hd = _load_per_symbol(alt / "stk_holdernumber")[
        ["ts_code", "end_date", "holder_num"]].rename(
        columns={"ts_code": "symbol", "end_date": "ann_date"})
    hd = hd.dropna(subset=["holder_num"]).sort_values(["symbol", "ann_date"]).drop_duplicates(
        ["symbol", "ann_date"], keep="last")
    parts = []
    for sym, grp in panel.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=False):
        h = hd[hd["symbol"] == sym].sort_values("ann_date")
        grp = grp.copy()
        if h.empty:
            grp["holder_num"] = np.nan
        else:
            pos = h["ann_date"].to_numpy().searchsorted(
                grp["trade_date"].to_numpy(), side="right") - 1
            vals = h["holder_num"].to_numpy()
            grp["holder_num"] = np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], np.nan)
        parts.append(grp)
    panel = pd.concat(parts, ignore_index=True)
    panel["holder_chg"] = panel.groupby("symbol")["holder_num"].pct_change()

    # no look-ahead: moneyflow & margin are post-close / T+1, use yesterday
    for c in ("main_flow", "lg_net", "margin_chg"):
        panel[f"{c}_prev"] = panel.groupby("symbol")[c].shift(1)
    for c in ("size_log", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "turnover_rate",
              "volume_ratio", "winner_rate", "ep"):
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


def rolling_gbdt(prepared: pd.DataFrame, prepared_tail: pd.DataFrame,
                 full_panel: pd.DataFrame, feats: list[str],
                 window: int = WINDOW, refit: int = REFIT_EVERY) -> Fit:
    """Hybrid rolling GBDT: main model (label60) + tail transition model (label20).

    First-principles fix for the M25 tail-gap bug: ``LABEL_HORIZON``-day labels
    do not exist for the most recent ``LABEL_HORIZON`` days, so a single-label
    rolling model stalls there. The main model (slow alpha) covers history; a
    tail model trained on a shorter ``TAIL_LABEL`` horizon keeps the rolling
    system updating all the way to the latest bar.

    Each refit records its observability card (frank-quant style): train/valid
    split, train IC, valid IC, gap penalty, promotion score, and full 27-factor
    permutation importance.
    """
    from bisect import bisect_right

    full_dates = sorted(full_panel["trade_date"].unique())
    full_by_date = {d: g for d, g in full_panel.groupby("trade_date", sort=False)}
    split_date = full_dates[len(full_dates) - 1 - LABEL_HORIZON]

    dates_long = sorted(prepared["trade_date"].unique())
    dates_tail = sorted(prepared_tail["trade_date"].unique())
    main_refit = [dates_long[i] for i in range(window, len(dates_long), refit)
                  if dates_long[i] <= split_date]
    tail_refit = [dates_tail[i] for i in range(window, len(dates_tail), refit)
                  if dates_tail[i] > split_date]
    all_refit = sorted(set(main_refit + tail_refit))
    valid_len = refit

    preds: dict[pd.Timestamp, np.ndarray] = {}
    last_model = None
    model_evolution: list[dict] = []

    def _next(cur):
        for rd in all_refit:
            if rd > cur:
                return rd
        return full_dates[-1]

    def _train_card(seg, cur, idx):
        """Fit a model on seg's trailing `window`, record its observability card."""
        seg_dates = sorted(seg["trade_date"].unique())
        i = seg_dates.index(cur)
        tr = seg[(seg["trade_date"] >= seg_dates[i - window])
                 & (seg["trade_date"] <= seg_dates[i])]
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=5,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42)
        m.fit(tr[feats].to_numpy(), tr["_label"].to_numpy())
        tr_dates = sorted(tr["trade_date"].unique())
        cut = max(1, len(tr_dates) - valid_len)
        train_part = tr[tr["trade_date"] <= tr_dates[cut - 1]]
        valid_part = tr[tr["trade_date"] >= tr_dates[cut]]
        train_ic = _cross_sectional_ic(train_part, m, feats)
        valid_ic = _cross_sectional_ic(valid_part, m, feats)
        gap = 0.5 * max(0.0, train_ic - valid_ic)
        imp_part = (train_part.sample(8000, random_state=42)
                    if len(train_part) > 8000 else train_part)
        imp = permutation_importance(
            m, imp_part[feats].to_numpy(), imp_part["_label"].to_numpy(),
            n_repeats=1, random_state=42)
        importance = [{"feature": f, "label": FEATURE_LABELS.get(f, f),
                       "importance": round(float(v), 6)}
                      for f, v in zip(feats, imp.importances_mean, strict=True)]
        end = _next(cur)
        model_evolution.append({
            "index": idx,
            "refit_date": str(cur.date()),
            "start_date": str(cur.date()),
            "end_date": str(end.date()),
            "train_start": str(seg_dates[i - window].date()),
            "train_end": str(tr_dates[cut - 1].date()),
            "valid_start": str(tr_dates[cut].date()),
            "valid_end": str(cur.date()),
            "n_train_days": len(tr_dates) - valid_len,
            "n_valid_days": valid_len,
            "train_ic": round(train_ic, 4),
            "valid_ic": round(valid_ic, 4),
            "gap_penalty": round(gap, 4),
            "score": round(valid_ic - gap, 4),
            "feature_importance": importance,
        })
        return m, end

    idx = 0
    for cur in main_refit:
        m, end = _train_card(prepared, cur, idx)
        last_model = m
        idx += 1
        lo = bisect_right(full_dates, cur)
        hi = bisect_right(full_dates, end)
        for day in full_dates[lo:hi]:
            dframe = full_by_date[day]
            dframe = dframe[dframe[feats].notna().all(axis=1)]
            if len(dframe):
                preds[day] = (dframe["symbol"].to_numpy(),
                              m.predict(dframe[feats].to_numpy()))
    for cur in tail_refit:
        m, end = _train_card(prepared_tail, cur, idx)
        last_model = m
        idx += 1
        lo = bisect_right(full_dates, cur)
        hi = bisect_right(full_dates, end)
        for day in full_dates[lo:hi]:
            dframe = full_by_date[day]
            dframe = dframe[dframe[feats].notna().all(axis=1)]
            if len(dframe):
                preds[day] = (dframe["symbol"].to_numpy(),
                              m.predict(dframe[feats].to_numpy()))
    return Fit(predictions=preds, last_model=last_model, model_evolution=model_evolution)


def backtest(
    panel: pd.DataFrame,
    fit: Fit,
    scale: pd.Series,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Hold-to-exit backtest with realistic fills (next-open + slippage).

    Daily buy/sell capped (human-executable); cashout drained at max_sell/day
    (no single-day book dump). Exit priority: stop > profit > cashout > signal.

    ``panel`` is the FULL frame (features + prices on every trading day) so the
    backtest runs to the latest bar; signals come from ``fit.predictions``.
    """
    dates = sorted(panel["trade_date"].unique())
    open_p = panel.pivot_table(index="trade_date", columns="symbol", values="open")
    nav = 1.0
    navs, tos, trades = [], [], []
    holdings: dict[str, tuple[float, int]] = {}
    exit_queue: set[str] = set()
    for i in range(1, len(dates) - 1):
        sd, buy_day, sell_day = dates[i - 1], dates[i], dates[i + 1]
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
        o = open_p.loc[buy_day]
        n_sells = n_buys = 0
        sells_today = 0
        if sc == 0.0:
            exit_queue |= set(holdings.keys())

        order: list[tuple[int, str]] = []
        for sym in holdings:
            bp, bi = holdings[sym]
            price = o.get(sym, np.nan)
            if pd.isna(price):
                prio = 9
            elif price <= bp * (1 - STOP_LOSS):
                prio = 0
            elif price >= bp * (1 + TAKE_PROFIT):
                prio = 1
            elif sym in exit_queue:
                prio = 2
            elif i - bi >= MIN_HOLD and sym not in in_buffer:
                prio = 3
            else:
                prio = 9
            order.append((prio, sym))
        order.sort()
        for _, sym in order:
            if sells_today >= MAX_SELL:
                break
            bp, bi = holdings[sym]
            price = o.get(sym, np.nan)
            if pd.isna(price):
                continue
            reason = None
            if price <= bp * (1 - STOP_LOSS):
                reason = "stop"
            elif price >= bp * (1 + TAKE_PROFIT):
                reason = "profit"
            elif sym in exit_queue:
                reason = "cashout"
            elif i - bi >= MIN_HOLD and sym not in in_buffer:
                reason = "signal"
            if reason:
                del holdings[sym]
                exit_queue.discard(sym)
                n_sells += 1
                sells_today += 1
                trades.append({"date": str(buy_day.date()), "symbol": sym, "side": "sell",
                               "price": round(float(price), 2), "reason": reason})
        if sc > 0.0 and sd in fit.predictions:
            syms, pred = fit.predictions[sd]
            buys_today = 0
            for sym in syms[np.argsort(-pred)]:
                if buys_today >= MAX_BUY:
                    break
                if len(holdings) >= TOP_K:  # full book — do not buy until a slot frees
                    break
                if sym not in in_pool or sym in holdings:
                    continue
                price = o.get(sym, np.nan)
                if pd.isna(price):
                    continue
                holdings[sym] = (price, i)
                n_buys += 1
                buys_today += 1
                trades.append({"date": str(buy_day.date()), "symbol": sym, "side": "buy",
                               "price": round(float(price), 2), "reason": "signal"})
        o_cur, o_next = open_p.loc[buy_day], open_p.loc[sell_day]
        rets = []
        for sym in holdings:
            p0, p1 = o_cur.get(sym, np.nan), o_next.get(sym, np.nan)
            rets.append(0.0 if (pd.isna(p0) or pd.isna(p1)) else p1 / p0 - 1.0)
        gross = float(np.nanmean(rets)) if rets else 0.0
        turnover = (n_sells + n_buys) / max(len(holdings) + n_sells, 1)
        nav *= 1.0 + gross - turnover * COST_PER_SIDE
        navs.append(nav)
        tos.append(turnover)
    return (pd.Series(navs, index=pd.DatetimeIndex(dates[1:-1]), name="nav"),
            pd.Series(tos, index=pd.DatetimeIndex(dates[1:-1])), trades)


def paper_backtest(
    panel: pd.DataFrame,
    fit: Fit,
    scale: pd.Series,
    initial_cash: float = 1_000_000.0,
) -> tuple[pd.Series, list[dict]]:
    """Paper-trading engine: a cash account that mimics real execution.

    Unlike the research backtest (equal-weight, full-invested), this models an
    actual wallet:
      - fixed budget per name (initial_cash / TOP_K)
      - buy deducts cash, sell adds it back (cost applied both sides)
      - board-lot 100 shares; limit-up cannot be bought, limit-down cannot be sold
      - suspended names (no open) cannot trade
      - NAV = cash + mark-to-market value (close)
    """
    dates = sorted(panel["trade_date"].unique())
    open_p = panel.pivot_table(index="trade_date", columns="symbol", values="open")
    close_p = panel.pivot_table(index="trade_date", columns="symbol", values="close")
    pc = panel.sort_values(["symbol", "trade_date"]).groupby("symbol")["close"].shift(1)
    prev_close_p = panel.assign(_pc=pc).pivot_table(
        index="trade_date", columns="symbol", values="_pc")

    # Tiered slippage by float market cap (万元). Small names really do cost more
    # to trade; probed 2026-08-16: -1.1pp vs flat 0.15%, worth it for realism.
    circ_mv = panel.drop_duplicates("symbol").set_index("symbol")["circ_mv"]

    def slippage(sym: str) -> float:
        mv = circ_mv.get(sym, np.nan)
        if pd.isna(mv) or mv < 50 * 1e4:
            return 0.003
        if mv < 200 * 1e4:
            return 0.002
        return 0.001

    cash = initial_cash
    holdings: dict[str, dict] = {}
    navs, trades = [], []
    for i in range(1, len(dates) - 1):
        sd, buy_day = dates[i - 1], dates[i]
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
        o = open_p.loc[buy_day]
        prev_row = prev_close_p.loc[buy_day]
        # ---- sells ----
        for sym in list(holdings.keys()):
            price = o.get(sym, np.nan)
            if pd.isna(price):  # suspended — cannot sell
                continue
            prev = prev_row.get(sym, np.nan)
            if not pd.isna(prev) and price <= prev * 0.905:  # limit-down — cannot sell
                continue
            h = holdings[sym]
            cost = h["cost"]
            reason = None
            if price <= cost * (1 - STOP_LOSS):
                reason = "stop"
            elif price >= cost * (1 + TAKE_PROFIT):
                reason = "profit"
            elif sc == 0.0:
                reason = "cashout"
            elif i - h["buy_i"] >= MIN_HOLD and sym not in in_buffer:
                reason = "signal"
            if reason:
                cash += h["shares"] * price * (1 - slippage(sym))
                del holdings[sym]
                trades.append({"date": str(buy_day.date()), "symbol": sym, "side": "sell",
                               "price": round(float(price), 2), "reason": reason})
        # ---- buys ----
        if sc > 0.0 and sd in fit.predictions:
            # dynamic budget = current total assets / TOP_K (keeps full-invested
            # equal weight as the account compounds, instead of a fixed 12.5万)
            c0 = close_p.loc[buy_day]
            mv = 0.0
            for _sym, h in holdings.items():
                _p = c0.get(_sym, np.nan)
                mv += h["shares"] * (h["cost"] if pd.isna(_p) else _p)
            total_assets = cash + mv
            budget = total_assets / TOP_K
            syms, pred = fit.predictions[sd]
            for sym in syms[np.argsort(-pred)]:
                if len(holdings) >= TOP_K:  # full book — no new buys until a slot frees
                    break
                if cash <= 0 or sym not in in_pool or sym in holdings:
                    continue
                price = o.get(sym, np.nan)
                if pd.isna(price):
                    continue
                prev = prev_row.get(sym, np.nan)
                if not pd.isna(prev) and price >= prev * 1.095:  # limit-up — cannot buy
                    continue
                shares = int(min(budget, cash) / price / 100) * 100
                if shares < 100:
                    continue
                cash -= shares * price * (1 + slippage(sym))
                holdings[sym] = {"shares": shares, "cost": float(price), "buy_i": i}
                trades.append({"date": str(buy_day.date()), "symbol": sym, "side": "buy",
                               "price": round(float(price), 2), "reason": "signal",
                               "shares": shares})
        # ---- mark to market ----
        mv = 0.0
        c = close_p.loc[buy_day]
        for sym, h in holdings.items():
            p = c.get(sym, np.nan)
            if not pd.isna(p):
                mv += h["shares"] * p
        navs.append(cash + mv)
    return (pd.Series(navs, index=pd.DatetimeIndex(dates[1:-1]), name="nav"),
            trades)


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

    def _make_prepared(horizon: int) -> pd.DataFrame:
        p = panel.copy()
        p["_fwd"] = p.groupby("symbol")["close"].shift(-horizon) / p["close"] - 1.0
        p["_label"] = p.groupby("trade_date")["_fwd"].transform(lambda s: s - s.mean())
        p = p.dropna(subset=FEATURES + ["_label"])
        p = p[np.isfinite(p[FEATURES].to_numpy()).all(axis=1)]
        return p

    prepared = _make_prepared(LABEL_HORIZON)
    prepared_tail = _make_prepared(TAIL_LABEL)

    market_ret = panel.groupby("trade_date")["ret_1d"].mean()
    scale = hysteresis(market_ret)
    fit = rolling_gbdt(prepared, prepared_tail, panel, FEATURES)

    # research backtest (equal-weight full-invested) — for model evaluation only
    research_nav, research_turnover, _research_trades = backtest(panel, fit, scale)
    # paper-trading backtest (cash account) — mimics real execution
    paper_nav, paper_trades = paper_backtest(panel, fit, scale)
    nav, trades = paper_nav, paper_trades

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
                        val = float(val) * 1e-4  # 万元 -> 亿元
                    pctl = float((col <= r[f]).mean()) if col.notna().sum() > 0 else np.nan
                    attribution.append({
                        "feature": f,
                        "label": label,
                        "unit": unit,
                        "value": round(val, 2),
                        "percentile": round(pctl, 3),
                    })
            close_price = float(r["close"]) if len(row) else float(mv.get(s, np.nan))
            # 1M CNY initial capital, equal-weight across TOP_K names, board-lot 100
            budget = 1_000_000.0 / TOP_K
            shares = max(100, int(budget / close_price / 100) * 100) if close_price > 0 else 0
            signals.append({
                "symbol": str(s),
                "name": names.get(str(s), ""),
                "industry": industry.get(str(s), ""),
                "score": float(round(float(sc), 4)),
                "market_cap": float(mv.get(s, np.nan)) * 1e-4,  # 万元 -> 亿元
                "weight": round(1.0 / TOP_K, 4),
                "close": round(close_price, 2),
                "shares": shares,
                "amount": round(shares * close_price, 2),
                "attribution": attribution,
            })

    # feature importance (permutation, on the last trained model — the tail
    # transition model, so score it against the tail's own labels)
    feature_importance: list[dict] = []
    if fit.last_model is not None:
        imp_panel = prepared_tail[prepared_tail["trade_date"].isin(
            sorted(prepared_tail["trade_date"].unique())[-WINDOW:])]
        if len(imp_panel) > 20000:
            imp_panel = imp_panel.sample(20000, random_state=42)
        imp = permutation_importance(
            fit.last_model, imp_panel[FEATURES].to_numpy(), imp_panel["_label"].to_numpy(),
            n_repeats=3, random_state=42)
        feature_importance = [
            {"feature": f, "label": FEATURE_LABELS.get(f, f),
             "importance": round(float(v), 6)}
            for f, v in sorted(zip(FEATURES, imp.importances_mean, strict=True),
                               key=lambda x: -x[1])
        ]

    daily = nav.pct_change(fill_method=None).dropna()
    yearly = []
    for yr in sorted({d.year for d in daily.index}):
        seg = daily[daily.index.year == yr]
        if len(seg) >= 5:
            yearly.append({"year": yr, "return": float((1 + seg).prod() - 1.0)})

    # exit-reason distribution (for the "why do we sell" explanation)
    reason_labels = {"stop": "止损", "profit": "止盈", "signal": "信号失效", "cashout": "空仓"}
    reason_counts: dict[str, int] = {}
    for t in trades:
        if t["side"] == "sell":
            reason_counts[t["reason"]] = reason_counts.get(t["reason"], 0) + 1
    exit_reasons = [
        {"reason": k, "label": reason_labels.get(k, k), "count": v}
        for k, v in sorted(reason_counts.items(), key=lambda x: -x[1])
    ]

    # rolling-refit timestamps + factor IC timeline (from model_evolution, so
    # they cover BOTH the main and tail models and extend to the latest bar —
    # the tail-gap fix also un-stalls this timeline, which used to stop at the
    # last label60 day).
    refit_dates = [me["refit_date"] for me in (fit.model_evolution or [])]
    long_last = sorted(prepared["trade_date"].unique())[-1]
    factor_ic_timeline = []
    for me in (fit.model_evolution or []):
        cur = pd.Timestamp(me["refit_date"])
        seg = prepared if cur <= long_last else prepared_tail
        seg_dates = sorted(seg["trade_date"].unique())
        i = seg_dates.index(cur)
        tr = seg[(seg["trade_date"] >= seg_dates[i - WINDOW])
                 & (seg["trade_date"] <= seg_dates[i])]
        ics = []
        for f in FEATURES:
            f_rank = tr.groupby("trade_date", sort=False)[f].rank()
            label_rank = tr.groupby("trade_date", sort=False)["_label"].rank()
            ic = f_rank.corr(label_rank)
            ics.append({
                "feature": f,
                "label": FEATURE_LABELS.get(f, f),
                "ic": round(float(ic), 4) if pd.notna(ic) else 0.0,
            })
        factor_ic_timeline.append({
            "date": me["refit_date"],
            "factors": sorted(ics, key=lambda x: -abs(x["ic"])),
        })

    # cashout history: merge consecutive cashout days (one flat event drains the
    # book over a few days) into single events, separated by a 5-day gap.
    cashout_days = sorted({t["date"] for t in trades if t["reason"] == "cashout"})
    cashout_history: list[dict] = []
    for d in cashout_days:
        if cashout_history:
            gap = (pd.Timestamp(d) - pd.Timestamp(cashout_history[-1]["date"])).days
            if gap <= 5:
                continue
        cashout_history.append({"date": d, "event": "大盘跌破均线，清仓空仓"})

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
            buy_log[t["symbol"]] = {"buy_price": t["price"], "buy_date": t["date"],
                                    "shares": t.get("shares", 0)}
        else:
            b = buy_log.pop(t["symbol"], None)
            if b is not None:
                pnl = (t["price"] - b["buy_price"]) / b["buy_price"]
                days = (pd.Timestamp(t["date"]) - pd.Timestamp(b["buy_date"])).days
                closed_trades.append({
                    "symbol": t["symbol"],
                    "name": names.get(t["symbol"], ""),
                    "buy_date": b["buy_date"], "buy_price": round(b["buy_price"], 2),
                    "sell_date": t["date"], "sell_price": round(t["price"], 2),
                    "pnl": round(pnl, 4), "days": days, "reason": t["reason"],
                })

    # historical daily signals: top-k for every trading day (history view).
    # A day where the market-timing scale is 0 (flat) shows NO names — the
    # strategy was out of the market, so the signal must say "空仓", not list
    # 10 stocks it would never have held.
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
                sig_list.append({
                    "symbol": s,
                    "name": names.get(s, ""),
                    "industry": industry.get(s, ""),
                    "score": round(float(pred[order[j]]), 4),
                    "close": round(float(close_map.get(s, float("nan"))), 2)
                             if pd.notna(close_map.get(s, float("nan"))) else None,
                    "market_cap": round(float(mv_map.get(s, float("nan"))) * 1e-4, 1)
                                  if pd.notna(mv_map.get(s, float("nan"))) else None,
                })
        daily_signals.append({
            "date": str(d.date()),
            "flat": flat,
            "signals": sig_list,
        })

    return {
        "generated_at": str(latest_date.date()),
        "signal_date": str(latest_date.date()),
        "strategy": "阿醒的 AI 策略 0813 · 模拟仓",
        "model_note": "训练模型：Deepseek-V4-Pro",
        "config": {
            "top_k": TOP_K,
            "label_horizon": LABEL_HORIZON,
            "min_hold": MIN_HOLD,
            "max_buy": MAX_BUY,
            "max_sell": MAX_SELL,
            "stop_loss": STOP_LOSS,
            "take_profit": TAKE_PROFIT,
            "cost_per_side": COST_PER_SIDE,
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
        "nav": [{"date": str(d.date()), "nav": round(float(v) / 1_000_000.0, 4)}
                for d, v in nav.items()],
        "benchmark": [{"date": str(d.date()), "nav": round(float(v), 4)}
                      for d, v in bench_nav.items()],
        "metrics": _metrics(nav / 1_000_000.0, research_turnover),
        "segments": {name: _segment(nav / 1_000_000.0, t0, t1) for name, t0, t1 in SEGMENTS},
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
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    m = out["metrics"]
    print(f"signal_date={out['signal_date']}  top_k={len(out['signals'])}")
    print(f"年化 {m['annual_return']:.2%} 夏普 {m['sharpe']:.2f} "
          f"回撤 {m['max_drawdown']:.2%} 换手 {m['avg_turnover']:.2%}")
    for seg, v in out["segments"].items():
        print(f"  {seg:6s} 年化 {v.get('annual_return', 0):.2%} 夏普 {v.get('sharpe', 0):.2f}")
    print(f"written -> {out_dir / 'champion.json'}")
