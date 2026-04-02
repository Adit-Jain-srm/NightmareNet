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

## Phase 5 — PR Review Fixes & Hardening
- `SlowAPIMiddleware` must be explicitly registered via `app.add_middleware(SlowAPIMiddleware)` — creating a `Limiter` and attaching it to `app.state` is NOT enough. Without the middleware, decorators are no-ops.
- CORS `allow_origins` must be stripped of whitespace. `"http://a.com , http://b.com".split(",")` yields `[" http://a.com ", " http://b.com "]` which won't match any origin header.
- In DDP training, ALL I/O operations (tokenizer save, history save, tracker.finish) that aren't model weights must be gated behind `dist_ctx.is_main_process`. Only model saving should use `dist_ctx.save_model()` (which handles sharding internally).
- Experiment trackers (wandb, tensorboard) must only be initialized on the main process. Non-main processes should get a no-op tracker to avoid duplicate runs/conflicts.
- Ruff line-length=100 must be checked before every commit. Long dict comprehensions and function signatures are the most common violators — extract helpers or wrap to multi-line.
- Hardcoded counts (like `tests_passing=159`) become immediately stale. Either compute dynamically or make optional.

## Phase 6 — Full Lint Cleanup
- Ruff's `select` key at `[tool.ruff]` top-level is deprecated. Use `[tool.ruff.lint]` section with `select` and `ignore` keys.
- UP007 (`Union[X, Y]` → `X | Y`) and UP045 (`Optional[X]` → `X | None`) must be ignored when targeting Python 3.9. Add them to `[tool.ruff.lint] ignore`.
- `ruff check --fix` for I001/F401 is reliable. Always auto-fix import sorting and unused imports first, then handle manual fixes.
- UP035 auto-fix (`typing.Iterator` → `collections.abc.Iterator`) can break I001 import ordering. Always run I001 fix again after UP035.
- B904: In FastAPI exception handlers, use `from e` for 4xx (preserves cause for debugging) and `from None` for 5xx (hides internals after logging).
- B008: FastAPI `Body(...)` in function defaults triggers B008. Move to module-level singletons like `_DISTORTION_BODY = Body(...)`.
- B023: Lambda inside a loop that captures a loop variable is a classic Python closure bug. Fix with default argument: `lambda x, _s=strength: ...`.
- E721: Use `is` instead of `==` for type comparisons (`expected_type is float`, not `expected_type == float`).
- N812: `import torch.nn.functional as F` is industry convention but violates N812. Use `# noqa: N812` rather than renaming.
