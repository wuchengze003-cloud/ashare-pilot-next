# ADR-004: Universe Is a Strategy-Bound Point-in-Time Contract

- Status: accepted
- Date: 2026-07-29

## Decision

The system does not hard-code any index as permanent architecture. Every
strategy must bind a point-in-time Universe contract that records member
validity, source, version, quality status, and `as_of`.

The initial general reference scope is `csi800-pit/v1`. Future additions such as
`csi300-pit/v1`, `csi1000-pit/v1`, or thematic Universes must provide complete
historical membership and independent verification.

Today's static watch list cannot backfill history and cannot independently grant
production trading eligibility.

## Final Eligibility

Universe members must still pass data completeness, listing age, ST, suspension,
limit-up/down, liquidity, and strategy-specific constraints. Base membership and
final eligibility must be recorded separately with reasons; nothing may be
silently skipped.
