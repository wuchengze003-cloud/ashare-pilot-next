# Evidence Reimplementation Record

This record keeps only the requirement source and the verification location in
the new repository. It does not copy legacy implementations, legacy reports,
legacy runtime artifacts, or legacy audit numbers.

| Capability | Requirement source commit | Current implementation | Current counter-verification |
|---|---|---|---|
| Point-in-time feature replay | `6536b56` | `apps/research` builds a deterministic feature panel from an immutable `DatasetSnapshot` | Appending future rows never changes the historical panel; a snapshot leaking future rows fails immediately |
| Historical member coverage | `535177d` | `services/data_gateway` checks members per trading day against listing/delisting ranges, bars, and suspension evidence | Missing member days, zero-observation members, and member-count anomalies all fail |
| Stale-artifact closure | `5421ec4` | Data Gateway evaluates freshness against an explicit trading calendar; Signal Runner reads the current head for the requested trading day | Stale data produces a failed health state; an old signal cannot be read as the current signal for a new trading day |

These capabilities were rebuilt independently with purely synthetic inputs. The
results prove gate behavior and module boundaries; they do not prove real vendor
coverage, strategy effectiveness, or tradable returns.
