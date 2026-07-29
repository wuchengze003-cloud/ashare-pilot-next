# Quant Core Rules

This package owns financial semantics and must remain deterministic.

- Require explicit `as_of`; never derive market time from wall-clock time.
- Do not import applications, services, Web, Ops, vendor SDKs, or experiment code.
- Costs, market rules, execution, portfolio, and state transitions require
  focused tests and architecture review.
- Do not add a production strategy formula here. Strategies implement the
  stable protocol and are promoted as immutable versions.
- Every behavior change requires replay and boundary tests.
