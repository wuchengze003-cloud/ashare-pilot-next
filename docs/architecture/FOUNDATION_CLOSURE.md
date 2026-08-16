# Production Foundation Closure Checklist

This checklist scopes the remaining work before real data can be connected.
Every item must land through an independent pull request and must not mix in
legacy repository directories, historical reports, runtime data, vendor data, or
production strategies.

## Gate 1: Unified Production State Pipeline

- Signal Runner has exactly one production build entry point.
- State is derived from system validation results; callers cannot declare health
  directly.
- Contract drift, data staleness, and hash anomalies prefer `HOLD`.
- Fail closed when no verifiable previous signal exists; never fabricate
  holdings.
- A never-activated Champion produces `FLAT`; a risk exit produces `REDUCE_ONLY`
  or `FLAT`.

## Gate 2: Immutable Point-in-Time Snapshots

- Data and Universe snapshots are constructed read-only from the same
  once-validated bytes.
- Strategies receive only snapshots, never paths, network clients, or separate
  market-data copies.
- A Champion binds fixed rule versions; the day's data and Universe snapshots are
  bound separately as run evidence.
- Appending future data never changes historical visible snapshots, features, or
  targets.

## Gate 3: Trusted Strategy and Environment Identity

- The production entry loads only registered, approved adapters; injecting an
  arbitrary Python object is forbidden.
- The runtime itself verifies the adapter, config, Git commit, and lockfile
  digests.
- When declared identity differs from locally computed identity, fail closed or
  keep the previous valid target.

## Gate 4: Signal Chain and Publication Recovery

- Every run may reference only the currently committed head.
- The previous signal must satisfy time, sequence, and chained-hash constraints.
- Published artifacts are immutable; consumers read only fully committed run
  directories.
- Rename failures and process restarts have deterministic recovery rules and
  attack tests.

## Gate 5: Data Gateway Evidence Capabilities

Reimplement and re-verify only the three capabilities described in
`MIGRATION_POLICY.md`:

- Deterministic feature replay with a future-data negative control.
- Delisted-security and historical-member coverage audit.
- Fail-closed behavior for stale data and stale signals.

Legacy implementations, legacy tests, legacy numbers, and legacy reports must
not be migrated directly.

## Freeze Conditions

After all five gates pass the repository boundary checks, contract validation,
unit tests, and attack tests, the production foundation freezes. The first real
data adapter may only be connected after that freeze; before it, no real backtest
or trading capability may be claimed.

## Freeze Status

As of 2026-07-30, all five gates passed independent pull requests and cloud
checks on the main branch. The foundation is currently frozen as "synthetic
loop complete, real capabilities not yet connected":

- Later pull requests may connect the first real data adapter and formal
  strategy research.
- Bypassing contracts, the head, point-in-time snapshots, coverage audit, or
  freshness gates is forbidden.
- Migrating legacy strategies, reports, runtime artifacts, data files, or
  compatibility entry points is forbidden.
- Any contract-meaning or module-boundary change must update the schemas, attack
  tests, and this checklist together.
