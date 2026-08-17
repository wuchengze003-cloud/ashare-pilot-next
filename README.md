# ashare-pilot-next

A daily A-share quant research and signal system. It runs a rolling
HistGradientBoosting champion, validates it against the repository's immutable
data and contract foundation, and renders a static read-only dashboard.

The system publishes **target positions only**: it does not connect to a broker,
does not report real holdings, orders, fills, or execution success, and does not
constitute investment advice.

## What is running today

- **Rolling GBDT champion**: 27 cross-sectional factors, hybrid label
  (`label60` + `label20` tail), top-10 equal-weight selection, index-based
  market-timing filter, and explicit exit rules (6% stop-loss / 25%
  take-profit / signal-expiry / cash-out).
- **Static dashboard**: `runtime/dashboard/` is generated as plain HTML —
  NAV curve, segment metrics, factor importance and IC evolution, per-model
  pages, closed-trade win rate, historical signal calendar, and per-stock
  research cards. The web layer is read-only and contains no strategy or
  financial calculation.
- **Independent update system**: `tools/update.py intraday|full|render`
  refreshes prices, retrains the champion, rebuilds research cards, and
  re-renders the dashboard.

## Data and update flow

```text
tools/fetch_market_data.py   -> runtime/ raw daily bars and metadata
tools/collect_alt_data.py    -> runtime/alt-data/* (moneyflow, margin, chips, ...)
apps/research/champion.py    -> runtime/dashboard/champion.json (train + backtest)
apps/research/stock_profile.py -> runtime/stock-profiles/*.json (research cards)
tools/fetch_stock_news.py    -> keyless Eastmoney news into *_extended.json
tools/render_dashboard.py    -> runtime/dashboard/*.html (static render)
tools/update.py              -> orchestrates intraday / full / render refreshes
```

The pilot market-data source is a Tushare-compatible HTTP endpoint. News is
fetched directly from the keyless Eastmoney search API; concepts, consensus,
chips, and ratings are cached connector snapshots collected through the
configured MCP connectors. Historical research reads immutable, hashed datasets
by `dataset_id`; it never depends on mutable HTTP responses or SQLite tables.

## Modules

| Path | Responsibility |
|---|---|
| `packages/quant_core/` | Sole Python authority for financial semantics: point-in-time interfaces, costs, execution, portfolio, and strategy protocol |
| `packages/obs/` | Structured logging and auditable errors |
| `apps/research/` | Experiments, factor/model research, evaluation, and champion promotion |
| `apps/signal_runner/` | Loads frozen champion packages and generates production target positions |
| `apps/web/` | Read-only rendering; no financial calculation |
| `services/data_gateway/` | Vendor acquisition, immutable datasets, quality proof, and online queries |
| `contracts/` | JSON Schemas, golden examples, and compatibility rules |
| `ops/` | Orchestration, validation, publication, and rollback |

## Local validation

All commands must work in a clean directory without the legacy repository,
vendor data, or machine-specific absolute paths.

```bash
uv sync --all-packages --dev
uv run ruff check .
uv run pytest
uv run python tools/validate_contracts.py
uv run python tools/check_boundaries.py
uv run python tools/check_language.py
```

`tools/check_language.py` machine-enforces the language policy: technical docs,
code comments, and docstrings are English; Chinese is allowed only in
user-facing dashboard/CLI strings and Chinese stock names in test fixtures.

## Authoritative documentation

- [Architecture entry point](docs/architecture/README.md)
- [Contract catalog](docs/architecture/CONTRACT_CATALOG.md)
- [Dependency rules](docs/architecture/DEPENDENCY_RULES.md)
- [State machine](docs/architecture/STATE_MACHINE.md)
- [Migration policy](docs/architecture/MIGRATION_POLICY.md)
- [Foundation closure](docs/architecture/FOUNDATION_CLOSURE.md)
- [Evidence reimplementation](docs/architecture/EVIDENCE_REIMPLEMENTATION.md)
- [Foundation acceptance](docs/architecture/ACCEPTANCE.md)
- [Dashboard update runbook](docs/UPDATE_RUNBOOK.md)
- [Handoff status](docs/HANDOFF.md)
