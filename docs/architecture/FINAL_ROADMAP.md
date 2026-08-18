# Backend Architecture Status

> Status: implemented backend boundaries. This document does not authorize
> production activation.

## Objective

The backend has two separate products:

1. A research system that evaluates factors and models on immutable point-in-time datasets.
2. A simulated account that consumes committed production signals and maintains an append-only cash and position ledger.

The static Web application is a read-only consumer of the committed signal and
simulated-account contracts.

## Completed foundation

- Immutable dataset manifests and explicit `as_of` dates.
- Shared A-share execution, cost, lot, suspension, and limit semantics in `quant_core`.
- Deterministic Production Signal publication and state transitions.
- Rolling model training that only admits labels whose outcome date is known at the refit date.
- Point-in-time holder data keyed by disclosure date rather than report-period end date.
- An independent simulated-account package with append-only state and committed-signal verification.

## Implemented backend controls

### Research evidence

- Research inputs are loaded through immutable Dataset and Feature Dataset
  Manifests.
- Reports record factor inputs, label horizon, training window, refit date, and
  immutable input identities.
- Causality tests compare full calculations with calculations truncated at the
  same `as_of` date.
- TRAIN, VALID, and TEST roles are explicit. Observed TEST results do not select
  parameters or activate a Champion.
- SPA block bootstrap is computed from stored returns and is diagnostic only.

### Promotion and activation

- Promotion evidence is validated against the registered contract.
- Dataset, snapshot, model-bundle, code, and configuration hashes are verified
  across the package.
- Promotion and human activation are separate commands.
- The complete Champion package is verified in an isolated Signal Runner
  environment.

### Daily backend cycle

- Refreshed market data is staged and published under an immutable Manifest.
- Partial or stale required inputs fail before research or signal generation.
- Signal Runner receives an explicit Dataset Manifest, Universe, and `as_of`.
- Simulated Account advances only from a verified committed signal and a direct
  successor execution-day Dataset Manifest.

## Web boundary

The implemented Web application renders only a verified Production Signal and
the simulated-account state bound to that signal. It does not train a model,
calculate a portfolio, infer fills, or present Research reconstruction as a
live account. Static release bytes are hash-bound at build time and verified
again before the bundled server starts.

## Acceptance criteria

- Changing prices or fundamentals after a historical refit date cannot change that refit's prediction.
- A disclosure is unavailable before its actual announcement date.
- A suspended holding retains the last explicit valuation price instead of becoming zero.
- Replaying or skipping a signal sequence is rejected by the simulated account.
- Every simulated state binds the source signal, prior state, dataset snapshot, market-day snapshot, cost rules, market rules, and execution policy by hash.
- Clean isolated installs of Research, Signal Runner, and Simulated Account pass their own tests without relying on another application's dependencies.
- No report contains a manually asserted pass result or a hard-coded statistical significance value.

## Deferred work

- Add broker integration only as a separate future system. The current
  repository makes no claim about real holdings, orders, fills, or execution
  success.
- Build an immutable multi-year minute-bar dataset before evaluating intraday
  models. The existing one-day local cache is not research evidence.
