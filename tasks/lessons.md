# NightmareNet — Lessons Learned

_(Updated after each correction or mistake)_

## Phase 1
- `from __future__ import annotations` breaks FastAPI's `Body(...)` parameter resolution with Pydantic v2. Remove it from API modules and use `Optional[]` instead.
- When adding `Request` as first param for slowapi, rename body params to avoid shadowing (`request` → `body`) and add explicit `Body(...)` annotation.

## Phase 2
- HuggingFace `IterableDataset` does not support `len()`, `.select()`, or `.train_test_split()`. Use `.take()`, `.filter()`, and `.with_format("torch")` instead.
- For streaming tokenization, use `dataset.column_names` which may be `None` for some IterableDatasets — provide fallback list.

## Phase 3
- Model type dispatch (`causal_lm`/`masked_lm`/`seq_classification`) should be isolated to the Trainer init, not threaded through every phase, since phases only care about loss computation which the model handles internally.
- Learned adversarial generator should gracefully fallback when model unavailable — test with nonexistent model name to verify fallback path.

## Verification Audit
- CLI flags that modify config are not automatically wired end-to-end. Always trace from argparse → config mutation → object construction → usage. The `--tracker` flag in `evaluate.py` modified `config["tracking"]` but never created a tracker or passed it to `Evaluator`.
- Script files (`scripts/*.py`) should use the same dispatch/factory patterns as the library code. `evaluate.py` hardcoded `AutoModelForCausalLM` instead of using the `_MODEL_TYPE_MAP` already defined in `trainer.py`.

## Phase 4 — Remaining Improvements
- Early stopping needs separate counters for epoch adjustment vs. halt. The `AdaptiveScheduler` uses `_no_improvement_count` for epoch scaling and `_es_no_improvement` for stopping — merging them causes conflicting behavior.
- `mock.patch` path must match where the import is resolved, not where it's defined. `load_dataset` imported inside a function body in `glue.py` needs `@patch("datasets.load_dataset")` not `@patch("nightmarenet.evaluation.glue.load_dataset")`.
- Distributed wrappers must be no-ops when the library is absent. Always gate on `_ACCELERATE_AVAILABLE` and fall back to single-device semantics silently.
- Type union syntax `str | torch.device` requires Python 3.10+ at runtime but works under `from __future__ import annotations`. Verify CI matrix includes 3.9.
