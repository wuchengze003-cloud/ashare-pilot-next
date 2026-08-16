# Foundation Architecture Acceptance

## Current Skeleton

- The repository contains no Legacy code, historical reports, runtime data, or
  machine-specific absolute paths.
- All schemas are valid and all golden examples pass their schemas.
- Python package dependency direction passes machine checks.
- State-machine concurrent-failure cases have tests.
- CI runs boundary, contract, test, and secret checks on push and pull request.

## Minimum Vertical Slice

Later milestones must complete this slice with fixed synthetic data:

```text
Dataset Manifest
-> PIT Universe
-> test-only reference strategy
-> quant_core portfolio semantics
-> Signal Runner target positions
-> Production Signal
-> Runtime Manifest
-> read-only Web rendering
```

Acceptance must include:

- Identical inputs and hashes produce byte-identical target results.
- Appending data after `as_of` never changes historical results.
- Cost golden examples and market-rule boundaries pass.
- Cost calculation covers both sides of the stamp-duty change date, minimum
  commission, per-item rounding, and cumulative costs at high turnover; missing
  or overlapping date segments and unsupported markets must fail closed.
- `ACTIVE/HOLD/REDUCE_ONLY/FLAT` combination tests pass.
- After deleting Web, Research and Signal Runner tests still pass.
- All validation still passes when no Legacy directory exists.

The reference strategy validates the system only. It does not enter formal races
and must not be presented as a profitable strategy.

## Current Verification Progress

The loop through `Runtime Manifest` has been completed with purely synthetic JSON
data, covering:

- Byte determinism for identical inputs.
- Data-file hash and JSON line-count verification.
- Champion 3.0 separation of fixed contracts, compatibility rules, and daily
  point-in-time evidence.
- Blocking mismatches in Champion adapter ID, declared adapter hash, and declared
  lockfile hash.
- Production entry rejecting injected strategy objects, declared Git SHAs, and
  declared lockfile digests.
- Signal Runner loading the content-addressed adapter from the Champion-specified
  path and computing Git and lockfile identity locally; code tampering is blocked
  before execution.
- One production entry producing `ACTIVE/HOLD/REDUCE_ONLY/FLAT` from validation
  results.
- Contract drift, lockfile drift, and stale data keep the previous target when a
  trusted previous signal exists.
- Without a trusted previous signal, forged degraded positions are rejected; a
  never-activated Champion means an explicit flat book.
- Targets outside the point-in-time Universe are blocked.
- Single-name, position-count, and gross-exposure constraints.
- Dataset and Universe immutable point-in-time snapshots are constructed from the
  same validated bytes/contracts.
- Strategies receive only snapshots, not paths, file handles, or separate market
  data copies.
- Future rows and file line-order changes do not alter historical Snapshot hashes
  or reference targets.
- Normal daily Manifest changes are not misclassified as Champion contract drift.
- Production Signal, Signal Head, and Runtime Manifest are built as immutable
  bytes.
- The current head is atomically switched only after the run directory is fully
  persisted with a `COMMITTED` marker.
- Old-head forks, future-signal replay, uncommitted directories, and
  cross-artifact hash errors are all blocked.
- Failures before/after run-directory rename and after head switch recover
  idempotently from the same artifacts.
- Research builds a byte-deterministic point-in-time feature panel from immutable
  snapshots; future rows cannot rewrite the historical panel.
- Data Gateway audits the historical Universe member-day by member-day;
  suspension, delisting, zero observation, and member-count anomalies have
  explicit semantics.
- Data freshness is evaluated against an explicit trading calendar; the current
  production signal must match the trading day the consumer requires.

Read-only Web rendering is still a later milestone; until then this repository
remains labeled "not for real trading".

The current adapter has content addressing and controlled loading, but the
snapshot loop still uses purely synthetic JSON data and is not equivalent to real
vendor-data replay. No formal adapter has entered the repository yet; an approved
adapter is trusted code, and this gate does not promise an OS-level file or
network sandbox against a malicious approved adapter.
