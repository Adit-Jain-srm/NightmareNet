# 1. Sleep-Inspired Phase Architecture over Monolithic Training

Date: 2026-03-15

## Status 

Accepted

## Context

Need a training methodology that addresses robustness, forgetting, and compression simultaneously.

## Decision

Implement 4 discrete phases (Wake/Dream/Nightmare/Compress) as separate classes rather than a single loss function combining all objectives.

## Consequences

- (+) Each phase is independently testable and configurable
- (+) Phases can be reordered, repeated, or skipped via config
- (+) Clear separation of concerns; easier to reason about
- (+) Aligns with neuroscience literature (memory consolidation during sleep)
- (-) More complex orchestration logic in Trainer
- (-) Phase transitions introduce potential information loss
