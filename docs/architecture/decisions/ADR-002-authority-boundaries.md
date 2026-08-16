# ADR-002: Financial Semantics and Production Inference Are Decoupled

- Status: accepted
- Date: 2026-07-29

## Decision

1. `packages/quant_core/` is the sole authority for point-in-time interfaces,
   costs, market rules, execution simulation, portfolio, backtest, and strategy
   protocol.
2. `apps/research/` owns experiments, races, evaluation, and promotion and may
   depend on `quant_core`.
3. `apps/signal_runner/` only loads immutable Champions and produces target
   positions; it may depend on `quant_core` and must not depend on Research
   implementation.
4. `apps/web/` consumes versioned artifacts only.
5. `services/data_gateway/` only provides data artifacts and online data
   interfaces.

## Rationale

Research needs exploration; production inference needs stability and
determinism. The two share core semantics but never share experiment lifecycle.
A single Python core also avoids separate TypeScript and Python implementations
of trading rules.

## Forbidden

- Research copies production inference logic.
- Signal Runner imports experiment, parameter-search, or promotion modules.
- Web recomputes strategy, state, or positions.
- Data Gateway decides strategies or positions from data content.
