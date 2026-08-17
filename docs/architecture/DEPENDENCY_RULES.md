# Dependency Direction

## Allowed

```text
apps/research       -> packages/quant_core
apps/signal_runner  -> packages/quant_core
apps/web            -> generated contract types or read-only HTTP
services/data_gateway -> contracts
ops                 -> each application's public commands
```

## Forbidden

```text
quant_core -> apps, services, ops, Web
signal_runner -> research
research -> signal_runner, Web
data_gateway -> strategy, portfolio, promotion
Web -> quant_core Python implementation, Research internals, data cache tables
any module -> the Legacy repository
```

Cross-module communication must go through versioned contracts or public package
interfaces. CI verifies Python dependencies with AST and path checks rather than
trusting this document alone.
