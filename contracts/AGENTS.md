# Contract Rules

- JSON Schema is the structural authority.
- Every schema has an immutable version and at least one synthetic example.
- Meaning, units, required fields, or failure behavior changes require a major version.
- Golden fixtures must be small, synthetic, deterministic, and safe to commit.
- Contract changes require producer and consumer tests.
- Never add market data, production signals, secrets, or runtime output as fixtures.
