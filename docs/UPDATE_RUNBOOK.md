# Update Runbook — how to refresh the dashboard

This document is the single entry point for any agent (or human) that needs to
refresh the A-share research dashboard. Read it top-to-bottom once and you can
run the full update chain without prior project context.

## 1. What this system is

A daily-quant research dashboard. The pipeline is:

```
raw data (tushare proxy)  ->  retrain rolling model  ->  per-stock research  ->  static HTML
```

- Signals are generated **after the close** of day T, executed at the **open of
  day T+1**. No look-ahead, no same-day-close fills.
- The web layer is **read-only static HTML**. It never contains strategy,
  backtest, or order logic.

## 2. Where the data lives (`runtime/`)

| Path | What |
|---|---|
| `runtime/full-market/` | Per-stock daily bars (OHLCV) |
| `runtime/alt-data/` | Alt datasets: `daily_basic` (valuation), `moneyflow`, `top_list`/`top_inst` (dragon-tiger), `hsgt_top10` (north-bound), `margin_detail` (margin), `cyq_perf` (chip), `stk_holdernumber` (shareholders) |
| `runtime/index/` | Benchmark index (SSE) |
| `runtime/sw_industry.json` | Shenwan L1 industry map |
| `runtime/stock-profiles/` | Per-stock research cards (`{code}.json`) + connector snapshots (`{code}_extended.json`) |
| `runtime/dashboard/` | `champion.json` (the single source of truth the renderer reads) + generated HTML |
| `runtime/panel_cache.pkl` | Wide panel cache (factors) |

All of `runtime/` is git-ignored. Never edit it by hand except through the
scripts below.

## 3. The three update tiers

Run from the repo root, via the project venv:

```bash
# A) Intraday quick refresh — ~1 min, NO retrain
#    Refreshes current holdings' latest close + rerenders. Safe to run any time.
uv run python tools/update.py intraday

# B) End-of-day full update — ~8 min, retrains the model
#    Collects alt data -> retrains rolling model -> regenerates research cards -> renders.
uv run python tools/update.py full

# C) Render only — seconds
#    Rerenders HTML from the existing champion.json + stock-profiles. No fetch, no retrain.
uv run python tools/update.py render
```

### What each tier does / does not do

| Tier | Fetch data | Retrain model | Regenerate signals | Rerender |
|---|---|---|---|---|
| `intraday` | latest close for holdings only | ❌ | ❌ | ✅ |
| `full` | ✅ (alt data up to latest trade date) | ✅ | ✅ | ✅ |
| `render` | ❌ | ❌ | ❌ | ✅ |

## 4. The underlying scripts (if you need finer control)

| Script | Purpose |
|---|---|
| `tools/fetch_market_data.py` | Single-stock fetch (`daily`, `mins`, `daily_basic`) |
| `tools/collect_alt_data.py` | Bulk alt-data collector (breakpoint-resume). `ALT_END_DATE` env var overrides the end date |
| `apps/research/src/ashare_research_app/champion.py` | `python -m ashare_research_app.champion` — retrain + backtest + emit `champion.json` |
| `apps/research/src/ashare_research_app/stock_profile.py` | `python -m ashare_research_app.stock_profile .` — per-stock research cards |
| `tools/render_dashboard.py` | Renders `index.html` + `acceptance.html` + model pages + detail pages |

## 5. Connector snapshots (concepts / consensus / chip / news / rating)

The per-stock research card also shows live-connector data (通达信 concepts,
腾讯自选股 consensus/chip/rating, 东方财富妙想 news). These are MCP tools, not
scripts, so they are collected in-session by an agent and landed to
`runtime/stock-profiles/{code}_extended.json`. Key facts:

- `tdx_api_data` concept boards: keep only rows with `配置分类 == 2` (real
  concepts); drop `配置分类 == 4` (dynamic tags like 大盘股/业绩预升/通达信热股).
- `data_chip` accepts comma-separated `codes` for bulk fetch.
- `data_consensus` returns empty for small caps without analyst coverage.
- `mx_finance_search_news` requires the 东方财富妙想 connector to be connected.

## 6. Gotchas / boundaries

- **Never** hand-edit `champion.json` to change metrics; regenerate via `champion.py`.
- The rolling model uses a hybrid label (label60 history + label20 tail). Tail
  models let the refit reach the latest bar; do not "fix" this back to a single
  horizon without re-running the tail-label sweep.
- `glob("*.json")` on `stock-profiles/` must skip `*_extended.json` (they share
  a code key with the base card and will overwrite it).
- Intraday price is the latest **daily close**, not a live tick; for true
  intraday quotes use the tdx/westock connectors in-session.
- Tushare calls are rate-limited by `DataApi` (>=0.2s) and retried; keep it that
  way. The project uses `urllib`, not `requests`.
