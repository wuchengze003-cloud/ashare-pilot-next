# Research Rules

- Import financial semantics from `ashare_quant_core`.
- Experiment code may search and evaluate, but cannot publish production signals.
- Final windows, promotion gates, candidate trials, code/config/data hashes, and
  selection history are immutable evidence.
- Do not import Signal Runner or Web.
- Notebooks and generated experiment output are not tracked in Git.
- Feature replay must consume immutable point-in-time snapshots and reject future rows.
