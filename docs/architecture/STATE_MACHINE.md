# Production State Machine

## States

| State | Meaning | Target-position behavior |
|---|---|---|
| `ACTIVE` | Champion, contracts, decision data, and execution environment are valid | Publish normal targets |
| `HOLD` | Decision or execution capability cannot be confirmed reliably | Reference the previous signal and copy the previous valid target item by item, unchanged |
| `REDUCE_ONLY` | Increasing risk is forbidden, but valid data permits lowering targets | Reference the previous signal; do not add securities and do not raise any security's target |
| `FLAT` | The explicit target is zero and there is a basis for publishing it | All target weights are zero |

## Deterministic Priority

The first match wins, top to bottom:

1. Execution price, trading status, or the previous valid target cannot be
   confirmed: `HOLD`.
2. An independent liquidation rule fires and the required data is valid: `FLAT`.
3. The current Champion is revoked, an independent risk-reduction rule fires, or
   only risk reduction is allowed: `REDUCE_ONLY`.
4. Decision data, contracts, hashes, or Champion health checks fail: `HOLD`.
5. A qualified Champion has never existed and the system has no production
   target: `FLAT`.
6. All checks pass: `ACTIVE`.

Data anomalies by themselves do not authorize a blind liquidation. Returning to
`ACTIVE` requires re-validating every input, not merely clearing an error flag.

Before publishing `HOLD` or `REDUCE_ONLY`, the system must load the currently
committed `Signal Head` and the complete signal it points to. The head, signal
hash, time, and sequence must cross-check. A lone hash, an arbitrary historical
signal, or an uncommitted run directory cannot prove state semantics and must
fail closed.

## Concurrent-Failure Examples

| Champion | Decision data | Execution data | Risk rule | Result |
|---|---|---|---|---|
| Revoked | Fresh | Unconfirmable | Reduce | `HOLD` |
| Revoked | Fresh | Valid | Reduce | `REDUCE_ONLY` |
| Healthy | Stale | Valid | None | `HOLD` |
| Healthy | Stale | Valid | Independent liquidation | `FLAT` |
| Missing and never activated | Fresh | Valid | None | `FLAT` |
| Healthy | Fresh | Valid | None | `ACTIVE` |
