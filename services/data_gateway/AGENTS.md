# Data Gateway Rules

- Own provider access, raw snapshots, normalization, quality, and dataset manifests.
- Do not import Quant Core, Research, Signal Runner, or Web.
- Never select strategies, rank securities, size positions, or infer production state.
- Upstream fallback cannot silently change dataset meaning.
- Publish immutable datasets atomically; expose source date, units, and quality.
- Audit every point-in-time member-day; suspensions and delist dates require explicit evidence.
- Assess freshness against an explicit trading calendar, never the process wall clock.
- Credentials and vendor data stay outside Git.
