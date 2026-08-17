"""Mid-term reversal / oversold-rebound strategy iteration on the run_hold track.

Usage:
    python apps/research/scripts/reversal_research.py --n 800   # small-sample iteration
    python apps/research/scripts/reversal_research.py --n 0     # full universe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "apps" / "research" / "src"))

from types import SimpleNamespace  # noqa: E402

from ashare_research_app.cross_sectional_ml import (  # noqa: E402
    market_trend_filter,
    prepare_panel,
    rolling_fit,
)
from ashare_research_app.factors import add_factors  # noqa: E402
from ashare_research_app.strategy import HoldConfig, run_hold  # noqa: E402

FULL = ROOT / "runtime" / "full-market"
DAILY_BASIC = ROOT / "runtime" / "alt-data" / "daily_basic"
CYQ = ROOT / "runtime" / "alt-data" / "cyq_perf"

TREND = None  # cached market_trend_filter


def load_full(n: int = 800) -> pd.DataFrame:
    files = sorted(FULL.glob("[0-9]*.json"))
    if n > 0:
        files = files[:n]
    frames = []
    names = {}
    for p in files:
        doc = json.loads(p.read_text(encoding="utf-8"))
        data = doc.get("data")
        if not data:
            continue
        sym = doc["symbol"]
        names[sym] = str(doc.get("name", ""))
        fr = pd.DataFrame(data)
        fr.insert(0, "symbol", sym)
        frames.append(fr)
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["name"] = df["symbol"].map(names)
    return df.sort_values(["symbol", "trade_date"], ignore_index=True)


def load_daily_basic(symbols: set[str]) -> pd.DataFrame:
    files = sorted(DAILY_BASIC.glob("*.json"))
    frames = []
    for p in files:
        doc = json.loads(p.read_text(encoding="utf-8"))
        fr = pd.DataFrame(doc["items"], columns=doc["fields"])
        fr = fr[fr["ts_code"].isin(symbols)]
        if fr.empty:
            continue
        frames.append(fr[["ts_code", "trade_date", "circ_mv", "turnover_rate",
                           "volume_ratio", "pe_ttm", "pb", "dv_ttm"]])
    db = pd.concat(frames, ignore_index=True)
    db["trade_date"] = pd.to_datetime(db["trade_date"], format="%Y%m%d")
    return db.rename(columns={"ts_code": "symbol"})


def load_cyq(symbols: set[str]) -> pd.DataFrame:
    frames = []
    for s in symbols:
        p = CYQ / f"{s}.json"
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        fr = pd.DataFrame(doc["items"], columns=doc["fields"])
        frames.append(fr[["ts_code", "trade_date", "winner_rate"]])
    if not frames:
        return pd.DataFrame(columns=["symbol", "trade_date", "winner_rate"])
    cyq = pd.concat(frames, ignore_index=True)
    cyq["trade_date"] = pd.to_datetime(cyq["trade_date"], format="%Y%m%d")
    return cyq.rename(columns={"ts_code": "symbol"})


def build_panel(n: int = 800) -> pd.DataFrame:
    df = load_full(n)
    df = add_factors(df)
    symbols = set(df["symbol"].unique())
    db = load_daily_basic(symbols)
    cyq = load_cyq(symbols)
    df = df.merge(db, on=["symbol", "trade_date"], how="left")
    df = df.merge(cyq, on=["symbol", "trade_date"], how="left")
    return df


def apply_quality(panel: pd.DataFrame, *, min_circ_mv: float = 3e5,
                  min_pb: float = 1.0, no_st: bool = True,
                  min_ret_1d: float = -0.05) -> pd.DataFrame:
    """Hard quality filter: market-cap floor, drop ST, drop broken book value
    (value trap), and drop same-day falling knives.

    circ_mv is in ten-thousand CNY (3bn CNY = 3e5). min_ret_1d avoids catching
    a name that is crashing on the signal day.
    """
    out = panel.copy()
    out["is_st"] = out["name"].astype(str).str.upper().str.contains("ST", na=False)
    out = out[~out["is_st"]] if no_st else out
    out = out[out["circ_mv"] >= min_circ_mv]  # ten-thousand CNY
    out = out[out["pb"] >= min_pb]
    out = out[out["ret_1d"] >= min_ret_1d]
    return out


def manual_fit(prepared: pd.DataFrame, factors: list[str],
               weights: list[float]):
    """Deterministic composite factor score (non-ML); stable ranks, low turnover."""
    score = sum(w * prepared[f"{f}_z"].to_numpy() for f, w in zip(factors, weights, strict=True))
    score = pd.Series(score, index=prepared.index)
    preds: dict[pd.Timestamp, np.ndarray] = {}
    for d, sub in prepared.groupby("trade_date", sort=True):
        preds[d] = score.loc[sub.index].to_numpy()
    return SimpleNamespace(predictions=preds, weights=None)


def run(panel: pd.DataFrame, factors: list[str], *, label_horizon: int = 15,
        top_k: int = 50, stop_loss: float = 0.08, take_profit: float = 0.25,
        buffer_days: int = 3, max_hold_days: int = 15, window: int = 60,
        refit_every: int = 10, model: str = "ridge", use_trend: bool = False,
        manual_weights: list[float] | None = None):
    t0 = time.perf_counter()
    prepared, feats = prepare_panel(panel, factors, label_horizon=label_horizon)
    if manual_weights is not None:
        fit = manual_fit(prepared, factors, manual_weights)
    else:
        fit = rolling_fit(prepared, feats, window=window, refit_every=refit_every,
                          model=model)
    ps = market_trend_filter(prepared) if use_trend else None
    cfg = HoldConfig(top_k=top_k, stop_loss=stop_loss, take_profit=take_profit,
                     buffer_days=buffer_days, max_hold_days=max_hold_days,
                     cost_per_side=0.0015, position_scale=ps)
    res = run_hold(prepared, fit, cfg)
    print(f"  [run] label={label_horizon} hold={max_hold_days} topk={top_k} "
          f"buf={buffer_days} sl={stop_loss} trend={use_trend} "
          f"manual={manual_weights} elapsed={time.perf_counter()-t0:.1f}s")
    return res, fit, prepared


def seg_metrics(res, prepared=None):
    nav = res.nav
    segs = [("train", "2023-01-01", "2024-06-30"),
            ("valid", "2024-07-01", "2025-06-30"),
            ("test", "2025-07-01", "2026-08-31")]
    out = {}
    for name, a, b in segs:
        sub = nav[(nav.index >= a) & (nav.index <= b)]
        if len(sub) < 2:
            out[name] = None
            continue
        n = len(sub)
        ann = sub.iloc[-1] ** (252 / n) - 1
        r = sub.pct_change(fill_method=None).dropna()
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        mdd = (sub / sub.cummax() - 1).min()
        out[name] = (ann, sharpe, mdd)
    return out


def describe_all(res, prepared=None, label=""):
    d = res.describe()
    print(f"=== {label} ===")
    print(f"  full: {d}")
    sm = seg_metrics(res, prepared)
    for name, v in sm.items():
        if v is None:
            print(f"  {name:6s}: no data")
        else:
            print(f"  {name:6s}: annual {v[0]:7.2%} sharpe {v[1]:5.2f} maxdd {v[2]:7.2%}")
    return sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="sample size, 0 = full universe")
    ap.add_argument("--min-circ-mv", type=float, default=3e5)
    ap.add_argument("--min-pb", type=float, default=1.0)
    ap.add_argument("--label", type=int, default=15)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--buf", type=int, default=10)
    ap.add_argument("--stop", type=float, default=0.25)
    ap.add_argument("--take", type=float, default=0.5)
    ap.add_argument("--trend", action="store_true")
    ap.add_argument("--winner", action="store_true", help="add the winner_rate factor")
    ap.add_argument("--ret1d-floor", type=float, default=-0.05)
    args = ap.parse_args()

    panel = build_panel(args.n)
    print(f"panel rows={len(panel)} symbols={panel['symbol'].nunique()} "
          f"dates={panel['trade_date'].min().date()}~{panel['trade_date'].max().date()}")

    panel = apply_quality(panel, min_circ_mv=args.min_circ_mv, min_pb=args.min_pb,
                          min_ret_1d=args.ret1d_floor)
    print(f"after quality filter: rows={len(panel)} symbols={panel['symbol'].nunique()}")

    factors = ["ret_20d", "ret_60d"]
    if args.winner:
        factors.append("winner_rate")

    res, fit, prepared = run(panel, factors, label_horizon=args.label,
                             top_k=args.topk, max_hold_days=args.hold,
                             buffer_days=args.buf, stop_loss=args.stop,
                             take_profit=args.take, use_trend=args.trend,
                             manual_weights=[-1] * len(factors))
    describe_all(res, prepared, label=f"n={args.n} factors={factors}")


if __name__ == "__main__":
    main()
