# Backend Roadmap

> Status: architecture plan. This document does not authorize production activation.

## Objective

The backend has two separate products:

1. A research system that evaluates factors and models on immutable point-in-time datasets.
2. A simulated account that consumes committed production signals and maintains an append-only cash and position ledger.

The static Web application is a read-only consumer and is deliberately postponed until these backend boundaries are stable.

## Completed foundation

- Immutable dataset manifests and explicit `as_of` dates.
- Shared A-share execution, cost, lot, suspension, and limit semantics in `quant_core`.
- Deterministic Production Signal publication and state transitions.
- Rolling model training that only admits labels whose outcome date is known at the refit date.
- Point-in-time holder data keyed by disclosure date rather than report-period end date.
- An independent simulated-account package with append-only state and committed-signal verification.

## Remaining backend work

### Research evidence

- Import all research inputs through immutable dataset snapshots.
- Record factor inputs, label horizon, training window, refit date, code hash, and configuration hash.
- Run causality checks by comparing a full calculation with a calculation truncated at the same `as_of` date.
- Keep TRAIN, VALID, and TEST roles explicit. TEST results must not select parameters or activate a Champion.
- Treat Deflated Sharpe and block bootstrap as diagnostics. Their values must be computed from stored inputs and must never be hard-coded in a report.

### Promotion and activation

- Validate promotion evidence against the registered contract.
- Verify cross-document dataset, snapshot, model-bundle, code, and configuration hashes.
- Keep promotion and human activation as separate commands.
- Verify the complete Champion package in an isolated Signal Runner environment.

### Daily backend cycle

- Publish refreshed market data through a staging directory and an immutable manifest.
- Abort before research or signal generation when any required input is partial or stale.
- Run the Signal Runner with an explicit dataset manifest, universe, and `as_of` date.
- Advance the simulated account only from a verified committed signal and a complete execution-day snapshot.
- Produce stage-health evidence for every failure and successful publication.

## Web boundary

The future Web application may render verified Production Signal and simulated-account contracts. It must not train a model, calculate a portfolio, infer fills, or present research reconstruction as a live account.

## Acceptance criteria

- Changing prices or fundamentals after a historical refit date cannot change that refit's prediction.
- A disclosure is unavailable before its actual announcement date.
- A suspended holding retains the last explicit valuation price instead of becoming zero.
- Replaying or skipping a signal sequence is rejected by the simulated account.
- Every simulated state binds the source signal, prior state, dataset snapshot, market-day snapshot, cost rules, market rules, and execution policy by hash.
- Clean isolated installs of Research, Signal Runner, and Simulated Account pass their own tests without relying on another application's dependencies.
- No report contains a manually asserted pass result or a hard-coded statistical significance value.

## Deferred work

- Rebuild the static Web application after the backend contracts and daily cycle are accepted.
- Add broker integration only as a separate future system. The current repository makes no claim about real holdings, orders, fills, or execution success.
