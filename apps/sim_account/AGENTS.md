# Simulated Account Rules

- Consume only a verified committed Production Signal.
- Import all trading, cost, T+1, limit, lot, and valuation semantics from `quant_core`.
- State advances monotonically and is append-only; model retraining never rewrites prior states.
- Every state binds the source signal and previous account state by canonical SHA-256.
- This application models a simulated account only. It never claims broker orders, fills, or holdings.
- All market dates and generated timestamps are explicit inputs; do not read wall-clock time.
