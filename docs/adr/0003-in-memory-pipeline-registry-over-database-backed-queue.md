# 3. In-Memory Pipeline Registry over Database-Backed Queue

Date: 2026-04-01

## Status

Accepted (revisit at scale)

## Context 

Need to manage long-running training pipelines with start/stop/status semantics.

## Decision

Use bounded in-memory dictionary with threading locks. Cap at 64 runners (configurable). Evict completed runs first when at capacity.

## Consequences

- (+) Zero infrastructure dependencies for self-hosted users
- (+) Simple implementation, easy to reason about
- (+) Fast status lookups (O(1) dict access)
- (-) State lost on process restart
- (-) Not shared across API replicas
- (-) Must migrate to persistent backend for hosted platform (Redis/PostgreSQL)
