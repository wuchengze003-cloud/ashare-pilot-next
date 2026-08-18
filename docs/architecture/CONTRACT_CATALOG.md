# Contract Catalog

Every contract must have a unique schema, version, producer, consumer, date
semantics, failure behavior, and golden examples.

| Contract | Producer | Consumer | Failure behavior |
|---|---|---|---|
| Dataset Manifest | Data Gateway | Research, Signal Runner | Dataset is not read |
| Universe | Data Gateway/controlled membership task | Research, Signal Runner | Candidate or inference is blocked |
| Coverage Audit | Data Gateway | Ops, Research governance | Incomplete historical dataset is blocked |
| Cost Model | Architecture owner | quant_core | Backtest and inference are blocked when no unique date/market rate segment exists |
| Market Rules | Architecture owner | quant_core | Relevant securities are blocked |
| Execution Policy | Architecture owner | quant_core | Backtest and target generation are blocked |
| Portfolio Risk | Risk owner | quant_core, Signal Runner | Target publication is blocked |
| Experiment Config | Research | Research | Experiment is invalid |
| Promotion Gate | Research governance | Research | No promotion |
| Champion | Research promotion flow | Signal Runner, Web | Degrade per the state machine |
| Strategy Adapter | Research promotion flow | Signal Runner | Do not load, or keep the previous target per the state machine |
| Production Signal | Signal Runner | Web, future execution adapter | Do not display as the new target |
| Signal Head | Signal Runner | Signal Runner, Web, Ops | Do not switch the current production signal |
| Runtime Manifest | Signal Runner/Ops | Web, audit | Do not publish |
| Stage Health | Each stage | Ops, Web operations | Block downstream stages |
| Simulated Market Day | Data Gateway/Ops | Simulated Account | Account state does not advance |
| Simulated Account State | Simulated Account | Simulated Account, future Web | Keep the previous committed state |
| Simulated Account Head | Simulated Account | Simulated Account, future Web | Do not switch the current state |

## Dataset Manifest 2.0

- Records data family, normalized record-schema summary, normalization version,
  source version, and parent Manifest.
- Every file records content hash, byte count, line count, and trading-date range.
- Signal Runner hashes, parses, and validates primary keys and dates on the same
  bytes it reads once, then constructs an immutable `DatasetSnapshot`; strategies
  never receive the data directory, paths, or file handles.
- `DatasetSnapshot` contains only records with `trade_date <= as_of`; its hash is
  generated from data family, date, schema, normalization version, and normalized
  visible records.

## Coverage Audit 1.0

- Rebuild point-in-time membership for every open trading day and truncate
  validity by listing and delisting dates.
- Every expected member-day must have bars or explicit suspension evidence;
  zero-observation members must never be silently skipped.
- Optional per-day member-count constraints are enforced daily; anomalous days
  are emitted as structured evidence.
- Audit numbers are generated live from the current inputs; old report numbers
  are never accepted as proof.

## Coverage Audit 2.0

- Keeps the 1.0 schema and golden examples unchanged; the contract registry uses
  `contract_id + schema_version` as identity, so both major versions can be
  audited at once.
- Each open member-day is classified exactly once, in the fixed order of bars,
  filtered `suspend_type == "S"` suspension evidence, same-day delisting, and
  unexplained missing. Resume records must not enter the suspension key.
- When a delisting day has neither bars nor suspension evidence, emit one
  `expected_delisted_member_days` record; days after the delisting day are no
  longer expected member-days.
- Classification counts must satisfy
  `expected_member_days = bar_member_days + suspended_member_days + expected_delisted_member_day_count + len(missing_member_days)`.
- `provenance_warnings` keep data-source warnings as structured reason codes.
  Non-official vendor endpoints must include
  `NON_OFFICIAL_VENDOR_ENDPOINT`; that warning alone does not fail the audit.

## Universe 2.0

- The daily Universe is a point-in-time membership snapshot that also declares a
  stable generation-rule ID and version.
- The `UniverseSnapshot` hash is generated from rule identity, date, and
  normalized members.
- Member content may change day to day; a change in generation-rule ID or version
  is contract drift.

## Production Signal 4.0

- Uses `contract_set` to bind separately the day's Dataset Manifest,
  point-in-time Dataset Snapshot, point-in-time Universe Snapshot, Champion,
  cost, market rules, execution rules, portfolio risk, code, config, and
  lockfile hashes.
- `HOLD` and `REDUCE_ONLY` must reference and load the previous complete signal.
- Every signal has a monotonically increasing sequence, previous-signal hash, and
  previous-head hash; the first signal sequence is fixed at 1.
- A `HOLD` target must equal the previous valid target.
- `REDUCE_ONLY` must not add securities and must not raise any security's target
  weight.
- After Signal Runner builds the output it must pass the formal JSON Schema again;
  validation failure means no publication.

## Signal Head 1.0

- The head binds the current run, current signal, signal content hash, sequence,
  date, and previous-head hash.
- A new run may only continue from the current committed head; old heads, skipped
  sequences, future signals, and concurrent forks must not switch it.
- The run directory is first marked `COMMITTED` as a complete commit, then the
  current head is atomically replaced. Consumers read self-verified run
  directories only through the current head.
- A failure after the run directory commits but before the head switch leaves an
  unactivated complete directory; retrying with the same immutable artifacts
  recovers. A disk-confirmation failure after the head switch may also retry
idempotently.

## Simulated Account 1.0

- A Simulated Market Day binds one complete execution session to one immutable
  Dataset Snapshot and records the exact previous trading date used for limits
  and sizing.
- The account consumes only a verified committed Production Signal and advances
  one signal sequence at a time. Replay, skipped sequences, and date rollback
  fail closed.
- All fills are explicitly simulated by `quant_core`; artifacts never claim
  broker orders, broker fills, or real holdings.
- Each immutable state binds the source signal, market day, cost model, market
  rules, execution policy, and previous account state by canonical SHA-256.
- Runs are committed before the account head advances atomically. Model retraining
  cannot rewrite an earlier state or trade.

## Cost Model 2.0

- Broker commission is an account-level assumption; regulatory fees must cite a
  public basis.
- Every calculation receives the trading date, market, and buy/sell direction
  explicitly.
- Stamp duty and transfer fees are calculated separately by date segment and
  direction, rounded per contract rule item by item, then summed.
- Date segments for one market must be continuous and non-overlapping, and the
  final segment must be open-ended; when no unique rate segment exists, fail
  closed.
- Slippage and market impact belong to the Execution Policy and must not be
  duplicated in the Cost Model.
- The first supported date is `2022-04-29` and the first markets are SSE/SZSE
  A-shares; earlier dates and BSE must not be backtested before audited rates are
  added.

## Champion 3.0

- `promotion_evidence` stores the promotion-time data Manifest, Dataset Snapshot,
  and Universe Snapshot hashes, purely as immutable audit evidence.
- `promotion_compatibility` binds the data family, record schema, normalization
  version, Universe rule ID, and rule version that future runs must preserve.
- `fixed_contract_set` binds cost, market rules, execution rules, portfolio risk,
  strategy code, config, and lockfile hashes; these fixed contracts must match
  the promotion environment exactly.
- A Champion must record its adapter ID and adapter artifact hash.
- Normal additions of bars or member changes do not invalidate a Champion just
  because daily content hashes change; a Champion must not activate when schema,
  normalization logic, Universe generation rules, or fixed contracts change.
- Signal Runner must never accept a caller-injected strategy object; it locates
  the adapter package only from the Champion's `adapter_id` under the controlled
  root.

## Strategy Adapter 1.0

- Contains adapter, strategy, and entry identity, plus code, config, and
  normalized package summaries.
- Signal Runner validates the Manifest, Champion binding, code bytes, and config
  bytes first, then executes the already-verified code bytes; it never reads
  code a second time by path.
- The adapter-returned object's strategy ID, version, and protocol are validated
  again.
- An adapter package is the immutable carrier of a formal strategy. The current
  repository commits only a purely synthetic reference package under the test
  directory; no promotable formal strategy exists.

## Version Rules

- Meaning, unit, or required-field changes: major version bump.
- Only new optional fields with clear old-consumer behavior: minor version bump.
- Published contract artifacts must never be rewritten in place.
- A consumer without support for a contract version fails closed.
- JSON Schema is the structural authority; financial calculation exists only in
  `quant_core`.
