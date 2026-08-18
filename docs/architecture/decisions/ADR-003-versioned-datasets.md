# ADR-003: Research and Production Use Immutable Datasets

- Status: accepted
- Date: 2026-07-29

## Decision

Data Gateway writes vendor responses into immutable raw snapshots and produces
normalized datasets and quality manifests. Every consumable dataset has a unique
`dataset_id`, `as_of`, file hashes, source, schema version, and quality status.

Research and Signal Runner read by `dataset_id`, never from mutable HTTP
responses or SQLite internal tables. Online HTTP serves only Web display and
operations diagnostics; it cannot become an implicit input to historical
backtests.

## Publication Protocol

1. Write into a temporary directory.
2. Validate schema, coverage, and file hashes.
3. Generate the manifest.
4. Atomically move the data directory.
5. Atomically publish the manifest last.

A failed task must not overwrite the previous complete dataset and must not
disguise partial data as empty data.
