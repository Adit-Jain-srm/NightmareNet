## Summary

Implement Distributed multi-GPU cycle execution with fault-tolerant checkpointing.

## Motivation

Closes #30

## Changes

- Created `nightmarenet/distributed/` package for GPU discovery, DDP wrapping, and phase-specific strategies.
- Implemented `AtomicCheckpointer` and `ResumeManager` to safely save state with `.complete` sentinels and resume training.
- Updated `cli.py` to add `--distributed` and `--resume` flags.
- Modified `pipeline.py` and `trainer.py` to seamlessly execute distributed strategies per-phase without breaking the pipeline.

## Acceptance Criteria

- [x] DevicePool discovers all available GPUs
- [x] Wake and Nightmare phases run with DDP when multiple GPUs available
- [x] Dream phase distributes augmentation batches across GPUs
- [x] Checkpoint saved atomically after each phase
- [x] `--resume` flag recovers from any phase checkpoint
- [x] Config hash verified on resume
- [x] `--distributed auto` works zero-config
- [x] Falls back to single-GPU if DDP fails
- [x] Audit trail records devices used per phase
- [x] Works with existing Docker setup (nvidia-docker)
- [x] Tests: checkpoint roundtrip, device pool logic, DDP init/teardown

## Type

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] Feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would break existing behavior)
- [ ] Refactor (no functional change)
- [ ] Documentation
- [x] Tests

## Pre-submission Checklist

- [x] I have **starred** this repository
- [x] I have **followed** [@Adit-Jain-srm](https://github.com/Adit-Jain-srm)
- [x] I have read [CONTRIBUTING.md](https://github.com/Adit-Jain-srm/NightmareNet/blob/main/CONTRIBUTING.md)

## Quality Checklist

- [x] `ruff check nightmarenet/ tests/` passes with 0 errors
- [x] `mypy nightmarenet/ --ignore-missing-imports` passes
- [x] `pytest tests/` — all tests pass (445+)
- [x] Added tests for new functionality (if applicable)
- [x] Updated documentation (if applicable)
- [x] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- [x] All acceptance criteria from the linked issue are satisfied (or exceptions noted above)

## Screenshots (if UI change)
