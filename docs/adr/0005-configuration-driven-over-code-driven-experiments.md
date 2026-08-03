# 5. Configuration-Driven over Code-Driven Experiments

Date: 2026-04-15

## Status 

Accepted

## Context

Users need reproducible experiments without modifying source code.

## Decision

All experiment parameters defined in YAML config files with schema validation. CLI accepts `--config` path. Python API accepts dict. No experiment requires code changes.

## Consequences

- (+) Full reproducibility via config file versioning
- (+) Non-programmers can run experiments
- (+) Config diffing shows exactly what changed between experiments
- (+) Validates at load time with clear error messages
- (-) Limited expressiveness compared to Python config (e.g., Hydra)
- (-) Complex conditional logic requires config nesting
- (-) Schema must be maintained in sync with implementation
