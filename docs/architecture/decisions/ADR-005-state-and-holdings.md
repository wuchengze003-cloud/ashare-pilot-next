# ADR-005: Degraded States and the Real-Holdings Boundary

- Status: accepted
- Date: 2026-07-29

## Decision

The first version publishes target positions only. It does not connect to a
broker, does not own real holdings, and never emits a "filled" conclusion. Web
must keep targets, user confirmation, and real fills distinct.

State priority and transitions are in
[`../STATE_MACHINE.md`](../STATE_MACHINE.md).

## Execution Boundary

- Signal Runner outputs target weights and reasons, not broker order status.
- The system does not handle partial fills, cancellations, or next-day
  continuation sells.
- Future automated trading must use a separate `execution_adapter` contract.
- Until an execution adapter is connected, no artifact may contain `filled`,
  `executed`, or equivalent claims.
