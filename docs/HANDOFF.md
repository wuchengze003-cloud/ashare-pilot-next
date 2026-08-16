# Handoff Document — ashare-pilot-next

> Prepared for the incoming engineer (DeepSeek) to continue development.
> This doc consolidates 16 rounds of iteration into one place: current state,
> the user's explicit requests, known tech debt, and the deployment checklist.
> The full work log lives in `.workbuddy/memory/2026-08-16.md` (297 lines).

---

## 1. What this project is

A daily-quant model for A-shares that produces **actionable signals for a
retail investor** (stock selection + market-timing position + exit risk-control).
The user is a non-quant retail investor — all explanations must use plain
language + analogies, but **technical docs/code/comments/commits are in English**.

Current output is a **static dashboard** (`runtime/dashboard/`): NAV curve,
factor evolution, per-model pages, closed-trade win-rate, historical signals
calendar, and per-stock research cards.

## 2. Tech stack & architecture

- Python 3.11, `uv` package manager, `.venv/`.
- `pytest` (82 tests), `ruff` (lint), JSON Schema Draft 2020-12 contracts.
- Data source: **Tushare Pro protocol via `https://teajoin.com` HTTP proxy**
  (`tools/fetch_market_data.py`). Uses `urllib`, **not** `requests`.
- Connectors (MCP, in-session): 通达信 tdx (concepts), 腾讯自选股 westock
  (consensus/chip/rating), 东方财富妙想 mx-ds (news/research).

### Data flow

```
fetch_market_data.py / collect_alt_data.py  -> runtime/ raw data
ashare_research_app.champion                 -> runtime/dashboard/champion.json (retrain + backtest)
ashare_research_app.stock_profile            -> runtime/stock-profiles/*.json (research card)
tools/render_dashboard.py                    -> runtime/dashboard/*.html (static render)
tools/update.py                              -> orchestrator (intraday / full / render)
```

## 3. What's been done (16 rounds, condensed)

- **Champion model**: rolling GBDT (HistGradientBoosting), 27 factors,
  label60 + label20 tail (hybrid), top-10 equal-weight, hysteresis market-timing
  (empty position when index breaks 20-day MA), 6% stop-loss / 25% take-profit.
- **Final metrics** (after tiered slippage): annualized **68.4%**, Sharpe **2.41**,
  max drawdown **-14.78%**, three segments train +13.5% / valid +149.8% / test +102.1%.
  Deflated Sharpe 0.972 (significant), MC bootstrap 100% prob(profit).
- **Rule probes** (data-driven, not gut): cooldown after take-profit = NEGATIVE
  (dropped); gap-limit delay = NEGATIVE (dropped); tiered slippage by market-cap
  = ADOPTED.
- **Dashboard**: factor evolution (27 factors, checkboxes + hover), NAV with
  model-stage markers (click-through), per-model pages (28), closed-trade
  win-rate (54.7%, profit/loss ratio 2.7), historical signals calendar with
  flat-market handling, per-stock research cards (valuation / 4-group financials
  / margin / north-bound / dragon-tiger / chip / rating / concepts / consensus).
- **Independent update system**: `tools/update.py` (intraday / full / render) +
  `docs/UPDATE_RUNBOOK.md`.
- **Decision skill**: `~/.workbuddy/skills/ashare-decision/SKILL.md`.

## 4. User's explicit requests for this handoff (核心诉求)

1. **Clean up the codebase** — multiple rounds of iteration left "spaghetti +
   redundancy". Remove dead code, duplicate logic, and `/tmp/*.py` throwaway
   scripts that got copy-pasted into the repo.
2. **README is stale and wrong** — it still says "架构骨架, not for real trading"
   and is written in Chinese. Update it to reflect the real system AND to English.
3. **Business conciseness** — keep the same *content*, but cut excessive
   explanatory text. **Simulate the experience of a real user** reading the
   dashboard: trim verbose helper sentences, and audit for any accidental
   leakage of internal terms/prompt fragments into the front-end.
4. **Audit unfinished tasks** — e.g. technical docs must be English (README +
   `docs/architecture/*.md` are currently mixed/Chinese); check whether prompts/
   docstrings stay concise.
5. **Deploy** — push the front-end to the user's Aliyun server, and update the
   GitHub repo (so the user can hand it to web GPT/Claude for review, saving tokens).

## 5. Known tech debt & audit checklist

### 5.1 Git is far behind
- Current branch `feat/frozen-champion-lifecycle` (PR15). `main` is still at PR14.
- **Almost all 16 rounds of work are uncommitted** — see `git status`: many new
  files (`champion.py`, `stock_profile.py`, `causality.py`, `factors.py`,
  `cross_sectional*.py`, `deflated_sharpe.py`, `mc_bootstrap.py`, `split.py`,
  `strategy.py`, `market_data.py`, `docs/UPDATE_RUNBOOK.md`,
  `docs/architecture/FINAL_ROADMAP.md`, `packages/obs/`, plus modified
  `pyproject.toml`, `uv.lock`, `check_boundaries.py`).

### 5.2 Language mix (violates the English-docs rule)
- `README.md` — **Chinese**, stale.
- `docs/architecture/README.md`, `ACCEPTANCE.md`, `CONTRACT_CATALOG.md`,
  `DEPENDENCY_RULES.md`, `STATE_MACHINE.md`, `MIGRATION_POLICY.md`,
  `FOUNDATION_CLOSURE.md`, `EVIDENCE_REIMPLEMENTATION.md`, `milestones/M1_ACCEPTANCE.md`
  — **Chinese** (early docs).
- `docs/architecture/FINAL_ROADMAP.md`, `docs/UPDATE_RUNBOOK.md`,
  `ADR-*.md` — English.
- `champion.py` has 33 Chinese occurrences (comments, not UI). `render_dashboard.py`
  has 219 Chinese (mostly UI text, which is fine — the user reads Chinese UI).

### 5.3 Possible leakage / verbosity in the front-end
- `render_dashboard.py` contains internal quant jargon in user-facing text
  (e.g. "frank-quant", "permutation importance", "label60", "DSR/MC").
  Decide what a real user should see vs. what should stay in docs.
- Several `<div class="sub">` helper sentences are verbose. Trim while keeping
  the numbers/tables intact.

### 5.4 Candidate dead code / redundancy
- `/tmp/experiment.py`, `/tmp/land_chip.py`, `/tmp/merge_extended.py`,
  `/tmp/land_extended.py`, `/tmp/sentiment_demo.py` — throwaway scripts that
  should be deleted or promoted into `tools/` if still needed.
- `runtime/dashboard/sentiment_demo.html` — demo page, superseded by the
  integrated per-stock cards.
- `runtime/` has ~22k files (raw data + generated HTML). Verify `.gitignore`
  covers all of it (AGENTS.md rule 9: runtime is never tracked).

### 5.5 Unfinished items (explicit)
- **News/research expansion to 18 holdings** — blocked: 东方财富妙想 MCP
  (mx-ds-mcp) is currently **disconnected**; needs the user to reconnect.
- **Rating** was just completed (14/17 have data).
- Task list still shows stale pending items (#7 Stage C, #22 alt-factor
  causality gate, #40 rewrite backtest engine, #46 research/sim split, #59
  sentiment demo) — several are already done de-facto; reconcile them.

## 6. Deployment tasks

### 6.1 GitHub
- Commit + push the 16 rounds of work. **Do NOT** force-push, do NOT merge into
  `main` directly (AGENTS.md rule 13: main changes only via PR).
- Recommended: create a feature branch off `feat/frozen-champion-lifecycle`
  (e.g. `feat/dashboard-16-rounds`) and open a PR, so the user can hand the
  branch/PR to web GPT/Claude for review.
- Ensure `runtime/` and `.env` are git-ignored before committing.

### 6.2 Aliyun (front-end)
- The dashboard is **pure static HTML** (`runtime/dashboard/`). Deploy by
  copying `index.html`, `detail/*.html`, `model/*.html`, `acceptance.html` to
  the Aliyun web root (or an OSS bucket). No server-side runtime needed.
- **Need from user**: server host/credentials or OSS bucket name.

## 7. Key conventions (do not break)

- Chinese in conversation, English in docs/code/comments/commits.
- A-share color: **red = up, green = down**.
- Currency: ¥ (CNY).
- Web is read-only; no strategy/backtest/cost/order logic in the web layer.
- Multiple `<script>` blocks share global scope — top-level `const` must be
  unique (or wrapped in IIFE). (Bit us once with `const PS`.)
- `glob("*.json")` on `stock-profiles/` must skip `*_extended.json`.
- tdx concept boards: keep only `配置分类 == 2`, drop `== 4` (dynamic tags).
- `data_chip` accepts comma-separated `codes` for bulk fetch.

## 8. Key files

| File | Role |
|---|---|
| `apps/research/src/ashare_research_app/champion.py` | retrain + backtest → `champion.json` |
| `apps/research/src/ashare_research_app/stock_profile.py` | per-stock research card |
| `tools/render_dashboard.py` | static HTML renderer (biggest file, ~76KB) |
| `tools/update.py` | orchestrator (intraday / full / render) |
| `tools/fetch_market_data.py` | tushare proxy client |
| `tools/collect_alt_data.py` | bulk alt-data collector |
| `docs/UPDATE_RUNBOOK.md` | how to refresh the dashboard |
| `.workbuddy/memory/2026-08-16.md` | full 16-round work log |
