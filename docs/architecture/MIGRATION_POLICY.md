# Legacy Capability Migration Policy

This repository never copies whole directories. Every migration must answer:

1. Does the current production goal actually need this capability?
2. What is the source commit and license?
3. Does it contain legacy strategies, legacy Universe, legacy runtime, or an
   implicit data source?
4. Can it be rewritten to the current contracts and dependency direction?
5. What anti-leakage, replay, boundary, and failure tests exist?
6. Can it still run independently after the Legacy directory is deleted?

## Explicitly Forbidden in the First Migration

- Legacy TypeScript strategies, backtests, and optimizers.
- Legacy Dashboard and historical signal pages.
- Legacy rules, multi-strategy selection, and fallback.
- Historical reports, backtest details, the full minute-frequency requirement
  set, and vendor exports.
- Legacy runtime, SQLite caches, environment files, and deployment state.

The useful legacy point-in-time filtering, delisted-stock coverage, and replay
tests may only serve as requirement and test evidence. They must be reimplemented
and independently accepted in the new Python core.

## First Evidence Asset List

The following items are migration requirements, not directly copyable code:

| Asset | Legacy baseline evidence | New-repository acceptance |
|---|---|---|
| Feature-panel replay determinism | `6536b56` | After a reviewed copy, identical inputs produce byte-identical outputs; the attack test must fail when point-in-time filtering is removed |
| Delisted-stock and historical-member coverage | `535177d` | Rerun the full audit with no silent skipping; new audit numbers are produced by tests or a manifest |
| Stale-signal fail-closed behavior | `5421ec4` | Both positive and negative freshness gates pass; an old date cannot masquerade as the current signal |

Migration order must be: read the source commit and test intent, reimplement
under the new contracts, run positive and negative tests, and record new
evidence. No `cherry-pick`, no copying legacy reports, no copying runtime, and
no referencing legacy repository paths.

Until all three assets are re-accepted in the new repository, the legacy
repository stays frozen but unarchived. After acceptance, apply the final tag
and make it read-only.

## Current Status

All three assets have been rebuilt independently under the current contracts and
dependency direction; verification locations are in
`EVIDENCE_REIMPLEMENTATION.md`. This repository contains no migrated legacy code,
legacy reports, legacy runtime artifacts, or legacy audit numbers.
