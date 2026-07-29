# Repository Rules

This repository is the clean production successor to the legacy A-share
assistant. Every rule below is enforced now. Future intentions belong in
architecture documents, not in this file.

## Non-negotiable Rules

1. `packages/quant_core/` is the sole Python authority for financial semantics.
2. Research and Signal Runner import the same `quant_core`; they do not copy
   cost, execution, portfolio, point-in-time, or strategy protocol logic.
3. Signal Runner cannot train, tune, promote, or read experiment internals.
4. Web is read-only and cannot contain strategy, backtest, optimizer, cost,
   portfolio, degraded-state, or order logic.
5. Historical research and production inference read immutable datasets by
   `dataset_id`; they never depend on mutable HTTP responses or SQLite tables.
6. Every decision binds explicit `as_of`, contract versions, input hashes, and
   code/config hashes. Wall-clock time is not market time.
7. The system outputs target positions only. It does not claim broker holdings,
   orders, fills, or execution success.
8. Production state follows the accepted deterministic state machine.
9. Runtime data, market data, historical reports, vendor exports, caches,
   credentials, and generated artifacts are never tracked in Git.
10. No source, import, path, runtime, database, deployment state, or symlink may
    reference the legacy repository.
11. Agents may edit only explicitly allowed paths. Prompts are not security
    boundaries.
12. Never reset, clean, stash, overwrite, or merge unrelated user or agent work.

## Change Classes

- **Leaf:** one module, no contract or financial-semantic change.
- **Contract:** changes shape, units, dates, versioning, freshness, or failure
  behavior. Requires producer and consumer tests.
- **Financial semantics:** changes costs, market rules, execution, portfolio,
  universe, strategy protocol, backtest, promotion, or state transitions.
  Requires focused statistical and architecture review.
- **Production:** changes champion activation, signal publication, deployment,
  or rollback. Requires explicit human approval.

## Required Handoff

State the base commit, changed paths, commands actually run, results, remaining
risks, and whether any contract, financial semantic, runtime, data, deployment,
or production state changed.
