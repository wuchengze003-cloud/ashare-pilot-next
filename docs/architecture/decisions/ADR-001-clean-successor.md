# ADR-001: Build a New, Physically Isolated Production Project

- Status: accepted
- Date: 2026-07-29

## Decision

`ashare-pilot-next` is built from an empty Git root. The legacy `ashare-pilot`
stays unchanged until the new system completes shadow operation and acceptance.
At the future switch, the legacy project is renamed Legacy and the new project
inherits the production name.

## Constraints

1. Do not copy legacy strategies, legacy TypeScript backtests, legacy Dashboard,
   historical reports, or runtime.
2. Legacy capabilities may only be migrated through independent audited tasks.
3. Migrated code must comply with current contracts, dependency direction, and
   test requirements.
4. The new project never knows Legacy's machine paths, remote addresses, or
   deployment state.
5. Deleting and archiving the legacy system is outside this repository's scope.

## Consequences

The current repository has no compatibility period and no Legacy fallback.
Missing capabilities stay missing until a new implementation passes acceptance.
