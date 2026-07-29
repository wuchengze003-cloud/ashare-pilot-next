# Web Rules

- Consume versioned production contracts only.
- Do not implement strategy, backtest, optimizer, cost, portfolio, state, or order logic.
- Empty targets are not a state; render the explicit state contract.
- Never claim targets are orders or fills.
- Do not read provider caches, SQLite tables, Research internals, or runtime directories.
- Generated contract types must come from the canonical schemas.
