# Signal Runner Rules

- Load only immutable, promoted strategy versions.
- Import financial semantics from `ashare_quant_core`.
- Never train, tune, search, promote, or import Research internals.
- Require explicit dataset, universe, champion, contract, and code hashes.
- Publish target positions only; never claim broker orders or fills.
- Write artifacts atomically and publish the manifest last.
