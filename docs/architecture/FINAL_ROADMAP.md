# Final Development Plan: From Engineering Foundation to a Daily-Investment Quant Model

> Status: planning document (no contract or financial-semantic change)
> Reference: frank-quant's *Four LLMs, One Quant Exam* (EP004) and its methodology
> Goal: advance this repo from a "data/engineering foundation" to a quant system that guides daily A-share investing, using first principles

## 0. Bottom line

**The foundation (data integrity + reproducibility) is already ahead of the reference. What is missing is the "research engine" above it — factors, cross-sectional stock selection, out-of-sample validation, overfitting defense, and the final daily product.**

The engineering can be built in one pass. But "the model actually guides investing" cannot be *proven* by writing code once — that is bounded by information theory (out-of-sample data can only be produced by waiting for time to pass). It converges only through time + physically-isolated out-of-sample + external audit. This plan delivers the former and builds an auditable evidence loop for the latter.

---

## 1. First principles: what you actually want

A quant system that "guides daily investing" answers three questions **every day**:

| Question | Quant form | Required capability |
|---|---|---|
| What to buy | Cross-sectional selection: of N stocks, which outperform over the next 5/20 days | Factors + ranking model |
| How much / when | Timing + sizing: market high/low → full/half/zero position; per-name weight | Timing signal + portfolio optimization |
| When to sell | Exit / risk control: conditions to reduce or liquidate | Risk rules + state machine |

Three A-share constraints determine what the system **must** have — and these are exactly what your foundation already encodes:

1. **T+1 + long-only** → no short hedge; "sit out the bear market" is the only defense → **timing and risk control matter more than pure alpha**.
2. **Limit up/down + suspension** → orders can fail to fill → precise **execution semantics** (`quant_core` already covers T+1, limit, suspension, per-item cost).
3. **Stamp duty / transfer fee / minimum commission / rounding** → cost model must be exact per item (`Cost Model 2.0` covers this, but only post-2022-04-29 + SSE/SZSE A-shares).

> Conclusion: your `ACTIVE/HOLD/REDUCE_ONLY/FLAT` state machine, point-in-time snapshots, and cost model are not over-engineering — they are the **correct encoding of A-share constraints**. The foundation direction is right; do not rebuild it.

---

## 2. Gap diagnosis: your workflow vs. real quant

The EP004 reference reveals the core of a real quant workflow. Compared to this repo:

| Dimension | This repo today | frank-quant / real quant | Gap |
|---|---|---|---|
| Data integrity | Very strong (PIT, coverage audit, hash chain, tamper-proof) | Medium (Freqtrade + Docker) | ✅ you are stronger |
| Reproducibility | Very strong (contracts + immutable snapshots + state machine) | Medium | ✅ you are stronger |
| **Three-segment OOS isolation** | Weak (walk-forward only, no physically-isolated TEST) | **Strong** (TRAIN/VALID/TEST, TEST on a separate disk, model cannot access) | ❌ missing |
| **Look-ahead gate** | Medium (PIT blocks leakage, but no full-vs-truncated causality comparison) | **Strong** (`factor_causality_check`) | ❌ missing |
| **Overfitting quantification** | None | **Strong** (Deflated Sharpe + Monte Carlo bootstrap) | ❌ missing |
| **Alpha factors / model** | Weak (baseline HGB regression, no cross-sectional factor library) | Medium (LLM-written strategies) | ❌ missing |
| **Final product form** | None (only web-state rendering) | Medium (strategy + backtest report) | ❌ missing |

**Key insight: what is valuable in the reference is "research discipline", not the framework.** These four disciplines can be rebuilt from scratch, without Freqtrade:

1. **Three-segment physical isolation** — not "prompt says don't look", but TEST data placed where the runner process cannot reach it.
2. **Look-ahead gate** — same timestamp: compute signal on full data vs. truncated-to-timestamp data; mismatch = used future data.
3. **Deflated Sharpe** — answer "given your search size, how high could pure luck push the Sharpe".
4. **Monte Carlo bootstrap** — block resampling yields `prob(profit)`, not a single backtest number.

---

## 3. Final roadmap (3 stages, one big audit)

> Principle: do NOT adopt Freqtrade or any open-source strategy framework. Reuse and extend the foundation (contracts, PIT, state machine, cost). Borrow only the "research discipline" methodology from the reference. Three large stages, then ONE big audit — no small incremental PRs.
>
> **A-share adaptation (do NOT copy the crypto reference directly):** the reference (EP004) is 20 crypto perpetuals — two-sided (long+short), 24/7, 30m-1d, funding-fee costs. This project is A-shares — **long-only** (short via margin is out of scope), **T+1 daily** frequency, **limit-up/down + suspension**, **stamp duty + commission + transfer fee**. Only the *discipline* (isolation, look-ahead gate, Deflated Sharpe, MC bootstrap) carries over; strategy constraints, cost model, frequency, and shorting are rebuilt for A-shares.

### Stage A — research-engine foundation (in progress)

| Component | Responsibility | Acceptance |
|---|---|---|
| `obs` structured logging | JSON-lines events + `AuditableError` provenance under each run dir; injectable clock (no wall-clock) | Every failure traceable to module/input; log exportable and replayable |
| Three-segment splitter | TRAIN (fit) / VALID (score only) / TEST (physically isolated: separate dir + permission isolation) | TEST data unreachable from the runner's working directory |
| `factor_causality_check` | Look-ahead gate: full vs. truncated signal comparison | Any factor leaking future rows → block promotion |
| `deflated_sharpe` | Deflated Sharpe (trial-count corrected) | Output the "luck ceiling", written into promotion evidence |
| `mc_bootstrap` | Block bootstrap resampling | Output `prob(profit)` + confidence interval |
| Adjustment-factor application | forward/backward adjusted price (M1 collected `adj_factor` but did not apply it) | Adjusted series reproducible |
| Full-market history expansion | M1's 20 names → all A-shares long history | Coverage audit clean |

### Stage B — alpha research loop (cross-sectional selection)

| Item | Notes |
|---|---|
| Factor library | momentum / reversal / value / quality / volatility / liquidity — each passes the Stage-A look-ahead gate |
| Cross-sectional model | replace baseline HGB; predict "next-N-day return ranking"; train/valid correlation as promotion threshold |
| Portfolio optimization | per-name cap, industry cap, turnover constraint, objective (reuse `quant_core` portfolio semantics) |
| Walk-forward backtest + OOS validation | produce backtest curves and metrics under the three-segment split |

### Stage C — productization + audit-ready (guides daily investing)

| Item | Notes |
|---|---|
| Daily selection list | Top-K + score + attribution + risk state |
| Timing / sizing | wire into `ACTIVE/HOLD/FLAT`, output daily target weights |
| Static pre-rendered dashboard | list page + backtest curve + validation report (read-only web, no financial calc) |
| Audit evidence chain | every conclusion traceable to `dataset_id` + commit hash + validation report; logs exportable and replayable |

> PR #15 (frozen Champion lifecycle) is folded into Stage A as a cleanup item rather than a standalone merge gate.

---

## 4. Production vs. audit split

| Role | Responsibility | Deliverable |
|---|---|---|
| **Me (production)** | write code, run data, produce validation reports | Auditable evidence chain: causality-check pass + Deflated Sharpe + MC bootstrap + OOS report |
| **Other model (audit)** | verify my conclusions hold | Check look-ahead, survivorship bias, slippage optimism, OOS authenticity |

**Audit-friendliness is a hard requirement**: every number must be independently replayable by the auditor, never just my self-reported metric.

---

## 5. Honest labeling: what can be "done in one pass" vs. what cannot

| Item | One-pass? | Why |
|---|---|---|
| Production system (foundation + research engine + product) | ✅ yes | Pure engineering, clear acceptance |
| Data integrity / reproducibility | ✅ yes | Already frozen, keep extending |
| **Model "actually guides investing"** | ❌ no | OOS validation needs data the model has never seen — only time produces it; overfitting is a function of search size, only repeatable via Deflated Sharpe / MC |

**Precise definition of "complete"**: the production system can be built in one pass; model validity converges only through time + physically-isolated OOS + external audit. This plan delivers the former and builds the auditable evidence loop for the latter.

---

## 6. Language policy (confirmed 2026-08-15)

- **Technical artifacts — English only**: README, AGENTS.md, `docs/`, code docstrings/comments, commit messages, Schema field names, contract naming.
- **Conversation with the owner — Chinese**.
- Rationale: quant terminology (factor, alpha, beta, Sharpe, walk-forward, point-in-time) is English-native; mixed Chinese/English in technical docs introduces ambiguity for both models and auditors, and pure-English is the quant-community standard.
