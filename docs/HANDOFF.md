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

### 5.1 Git status
- Current branch: `feat/dashboard-handoff` (based on the frozen-champion
  lifecycle branch). `main` remains at PR14 and must not be touched.
- The 16 rounds of work were committed in `c50dca5`; the cleanup described in
  section 9 lands as a follow-up commit on `feat/dashboard-handoff`.

### 5.2 Language policy (resolved)
- `README.md` and all `docs/architecture/**/*.md` (including ADRs) are now
  English.
- Code comments and docstrings under apps/, packages/, services/, tools/, and
  ops/ are English.
- Chinese is retained only in user-facing UI/output strings, enforced by the
  new `tools/check_language.py`.

### 5.3 Front-end audit (resolved)
- Investor-facing pages (index, detail, model) no longer contain frank-quant,
  label60, permutation importance, DSR, or MC.
- `acceptance.html` intentionally keeps those terms because it is the technical
  validation report.
- Verbose `<div class="sub">` sentences were trimmed to one data-fact sentence;
  all numbers, tables, and charts remain generated from `champion.json`.

### 5.4 Candidate dead code / redundancy
- The previously listed `/tmp/*.py` throwaway scripts are gone from this machine
  and no `/tmp/*.py` path is tracked in Git.
- The obsolete demo page was deleted.
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

### 6.2 Aliyun (front-end) - deployed
- The dashboard is **pure static HTML** (`runtime/dashboard/`). It is deployed
  to the Aliyun ECS host found in this machine's A-share assistant project
  (host stored locally as `DEPLOY_HOST=root@<aliyun-host>`; do not commit the
  real address). No server-side runtime is needed; the server has 2 vCPU,
  1.6 GiB RAM, and ~24 GiB free disk, which is ample for static hosting.
- Deployment: files are uploaded to `/var/www/ashare-pilot-next/` and served by
  Nginx at `/ashare-pilot-next/` via
  `/etc/nginx/snippets/ashare-pilot-next.conf`.
- Self-check passed: index, acceptance, model, and detail pages return HTTP
  200; existing `/` and `/a-share` routes still return HTTP 200 after reload.

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

## 9. Ready-to-use /goal condition

Paste this after `/goal` to have the AI keep working across turns until the
handoff cleanup is complete (self-verifying, per the /goal evaluator contract):

```text
Finish the ashare-pilot-next handoff cleanup:

1. `uv run pytest` exits 0 and `uv run ruff check .` reports no errors.
2. README.md is rewritten in English and describes the real system (rolling
   GBDT champion + static dashboard); no "架构骨架" or "不用于真实交易" remains.
3. README.md and every docs/architecture/*.md contain zero Chinese characters;
   code comments in apps/ tools/ packages/ services/ are English — Chinese is
   allowed only in render_dashboard.py's user-facing UI strings.
4. Dead code removed: no /tmp/*.py scripts are committed and the obsolete demo
   page is deleted.
5. Front-end audit: render_dashboard.py user-facing text no longer contains
   internal terms (frank-quant, label60, permutation importance, DSR, MC), and
   verbose <div class="sub"> helper sentences are trimmed while keeping all
   numbers and tables intact.
6. `git status` on feat/dashboard-handoff is clean (all work committed).

Inviolable: do not change champion.py strategy logic or the hybrid label
config; do not touch the main branch. Stop after 30 turns if not complete.
```
