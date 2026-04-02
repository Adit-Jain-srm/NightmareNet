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
