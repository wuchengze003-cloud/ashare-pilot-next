"""Stock profile generator — an independent, low-coupling research module.

Generates a per-stock "research card" (industry / valuation / financials /
funding) for the current holdings + latest signals, WITHOUT touching the
champion strategy.

Outputs: runtime/stock-profiles/{code}.json
Data sources (all already wired or fetched via teajoin):
    - industry: runtime/sw_industry.json (Shenwan Level-1)
    - valuation: runtime/alt-data/daily_basic (circ_mv, pe_ttm, pb, dv_ttm)
    - financials: fina_indicator via teajoin (roe, margins, yoy growth) -> cached
    - moneyflow: runtime/alt-data/moneyflow (net_mf_amount)
    - shareholder count: runtime/alt-data/stk_holdernumber
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def _load_by_date(dirpath: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(dirpath.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame(j["items"], columns=j["fields"]))
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    return out


def _load_per_symbol(dirpath: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(dirpath.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame(j["items"], columns=j["fields"]))
    out = pd.concat(frames, ignore_index=True)
    for col in ("trade_date", "end_date", "ann_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _target_symbols(root: Path) -> list[str]:
    """Union of current holdings and latest signals (the ~20 stocks we care about)."""
    doc = json.loads((root / "runtime/dashboard/champion.json").read_text(encoding="utf-8"))
    syms = {h["symbol"] for h in doc.get("current_holdings", [])}
    syms |= {s["symbol"] for s in doc.get("signals", [])}
    return sorted(syms)


def _fetch_fina(root: Path, symbols: list[str]) -> dict[str, dict]:
    """Fetch latest fina_indicator row per symbol via teajoin; cache to disk."""
    cache_dir = root / "runtime/stock-profiles/fina"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "tools"))
    from fetch_market_data import DataApi, load_env  # noqa: E402

    load_env()
    api = DataApi()
    out: dict[str, dict] = {}
    for code in symbols:
        cache = cache_dir / f"{code}.json"
        if cache.exists():
            rec = json.loads(cache.read_text(encoding="utf-8"))
        else:
            r = api.call("fina_indicator", {"ts_code": code})
            items = r.get("data", {}).get("items", [])
            fields = r.get("data", {}).get("fields", [])
            rec = {}
            if items:
                rows = [dict(zip(fields, row, strict=False)) for row in items]
                rows = sorted(rows, key=lambda x: str(x.get("end_date", "")))
                rec = rows[-1]  # latest reporting period
            cache.write_text(json.dumps(rec, ensure_ascii=False, default=str), encoding="utf-8")
        out[code] = rec
    return out


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def generate(root: Path) -> list[dict]:
    """Generate one research card per target stock.

    Financials now surface the FULL fina_indicator (108 fields) grouped by
    quality / growth / solvency / cashflow. Funding/positioning blocks
    (margin, north-bound, dragon-tiger list) are read from the already-landed
    tushare datasets under runtime/alt-data.
    """
    symbols = _target_symbols(root)
    names = {}
    for p in sorted((root / "runtime/full-market").glob("[0-9]*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        names[str(doc.get("symbol"))] = str(doc.get("name", ""))
    industry = json.loads((root / "runtime/sw_industry.json").read_text(encoding="utf-8"))

    db = _load_by_date(root / "runtime/alt-data/daily_basic")
    db = db[db["ts_code"].isin(symbols)]
    db_latest = db.sort_values("trade_date").drop_duplicates("ts_code", keep="last")

    mf = _load_by_date(root / "runtime/alt-data/moneyflow")
    mf = mf[mf["ts_code"].isin(symbols)].sort_values("trade_date")

    hd = _load_per_symbol(root / "runtime/alt-data/stk_holdernumber")
    fina = _fetch_fina(root, symbols)

    margin = _load_by_date(root / "runtime/alt-data/margin_detail")
    margin = margin[margin["ts_code"].isin(symbols)].sort_values("trade_date")

    north = _load_by_date(root / "runtime/alt-data/hsgt_top10")
    north = north[north["ts_code"].isin(symbols)].sort_values("trade_date")

    lhb = _load_by_date(root / "runtime/alt-data/top_list")
    lhb = lhb[lhb["ts_code"].isin(symbols)].sort_values("trade_date")

    def _num(v):
        f = _to_float(v)
        return round(f, 3) if f is not None else None

    def _yi(v):  # CNY -> hundred-million CNY
        f = _to_float(v)
        return round(f / 1e8, 2) if f is not None else None

    cards = []
    for code in symbols:
        fin = fina.get(code, {})
        db_row = db_latest[db_latest["ts_code"] == code]
        mf_5 = mf[mf["ts_code"] == code].tail(5)
        hd_sym = hd[hd["ts_code"] == code].sort_values("end_date").tail(2)

        # shareholder count change
        holder_chg = None
        if len(hd_sym) >= 2:
            a, b = hd_sym["holder_num"].to_numpy()
            if b and b > 0:
                holder_chg = round(float((b - a) / a), 4)

        # margin trading (financing and securities lending)
        mg_sym = margin[margin["ts_code"] == code].tail(1)
        margin_block = None
        if len(mg_sym):
            r = mg_sym.iloc[0]
            margin_block = {
                "date": str(r["trade_date"].date()),
                "rzye": _yi(r.get("rzye")),        # financing balance (100m CNY)
                "rzmre": _yi(r.get("rzmre")),      # financing buy amount (100m CNY)
                "rqye": _yi(r.get("rqye")),        # securities lending balance (100m CNY)
            }

        # north-bound top-10 traded names
        north_sym = north[north["ts_code"] == code].tail(20)
        north_block = None
        if len(north_sym):
            # net_amount is often None recently; fall back to buy - sell
            na = pd.to_numeric(north_sym["net_amount"], errors="coerce")
            if "buy" in north_sym.columns and "sell" in north_sym.columns:
                na = na.fillna(
                    pd.to_numeric(north_sym["buy"], errors="coerce")
                    - pd.to_numeric(north_sym["sell"], errors="coerce"))
            net5 = na.tail(5).sum()
            net20 = na.sum()
            north_block = {
                "date": str(north_sym.iloc[-1]["trade_date"].date()),
                "net_5d": _yi(net5),
                "net_20d": _yi(net20),
                "n_days": int(len(north_sym)),
            }

        # dragon-tiger list (recent)
        lhb_sym = lhb[lhb["ts_code"] == code].tail(3)
        lhb_block = []
        for _, r in lhb_sym.iterrows():
            lhb_block.append({
                "date": str(r["trade_date"].date()),
                "net_amount": _yi(r.get("net_amount")),
                "reason": str(r.get("reason", ""))[:30],
            })

        cards.append({
            "symbol": code,
            "name": names.get(code, ""),
            "industry": industry.get(code, ""),
            "valuation": {
                # hundred-million CNY
                "market_cap": _num(db_row["circ_mv"].iloc[0] * 1e-4) if len(db_row) else None,
                "pe_ttm": _num(db_row["pe_ttm"].iloc[0]) if len(db_row) else None,
                "pb": _num(db_row["pb"].iloc[0]) if len(db_row) else None,
                "dv_ttm": _num(db_row["dv_ttm"].iloc[0]) if len(db_row) else None,
                "turnover_rate": _num(db_row["turnover_rate"].iloc[0]) if len(db_row) else None,
            },
            "financials": {
                # Profitability quality.
                "roe": _num(fin.get("roe_yearly")),          # annualized ROE (primary)
                "roe_dt": _num(fin.get("roe_dt")),           # ROE excluding non-recurring items
                "roic": _num(fin.get("roic")),               # return on invested capital
                "netprofit_margin": _num(fin.get("netprofit_margin")),
                "gross_margin": _num(fin.get("grossprofit_margin")),
                # Growth.
                "netprofit_yoy": _num(fin.get("netprofit_yoy")),
                "dt_netprofit_yoy": _num(fin.get("dt_netprofit_yoy")),
                "revenue_yoy": _num(fin.get("or_yoy")),
                "q_op_qoq": _num(fin.get("q_op_qoq")),        # single-quarter net profit QoQ
                # Solvency and capital structure.
                "debt_to_assets": _num(fin.get("debt_to_assets")),
                "quick_ratio": _num(fin.get("quick_ratio")),
                "netdebt": _yi(fin.get("netdebt")),  # 100m CNY; negative = net cash
                "interestdebt": _yi(fin.get("interestdebt")),  # interest-bearing debt (100m CNY)
                # Cash flow and per-share metrics.
                "ocfps": _num(fin.get("ocfps")),              # operating cash flow per share
                "fcff": _yi(fin.get("fcff")),                 # free cash flow (100m CNY)
                "bps": _num(fin.get("bps")),                  # book value per share
                "eps": _num(fin.get("eps")),
                "report_date": str(fin.get("end_date", ""))[:10],
            },
            "moneyflow": {
                "net_5d": _num(mf_5["net_mf_amount"].sum() * 10000) if len(mf_5) else None,
            },
            "holder_chg": holder_chg,
            "margin": margin_block,
            "north": north_block,
            "lhb": lhb_block,
        })
    return cards


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    cards = generate(root)
    out_dir = root / "runtime/stock-profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cards:
        (out_dir / f"{c['symbol'].split('.')[0]}.json").write_text(
            json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated {len(cards)} stock profiles -> {out_dir}")
    for c in cards[:3]:
        print(f"  {c['symbol']} {c['name']} {c['industry']} PE={c['valuation']['pe_ttm']} "
              f"ROE={c['financials']['roe']}")
