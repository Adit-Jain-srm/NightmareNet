# Plan: NightmareNet Systematic Improvements

Implement 8 prioritized improvements across 3 phases. P0 (security/performance/CI) → P1 (observability/scale/deploy) → P2 (research capabilities). All phases have parallel steps within them; phases are sequential.

---

## Phase 1: P0 — Performance, Security & CI

### Step 1.1: Mixed-Precision Training (AMP)
- Add `use_amp` and `gradient_checkpointing` config keys to `configs/default.yaml` + `DEFAULT_CONFIG` in `nightmarenet/utils/config.py`
- Add `_SCHEMA` entry: `"training.use_amp": (bool, None, None, False)`
- In `nightmarenet/training/trainer.py`:
  - Import `torch.amp.GradScaler` and `torch.amp.autocast`
  - In `Trainer.__init__()`: create `self.scaler = GradScaler("cuda")` if `use_amp=True` and device is cuda
  - Pass `use_amp` flag to each phase runner
- In `nightmarenet/training/phases.py`:
  - **WakePhase.run()**: Wrap forward pass in `with autocast("cuda"):`, scale loss via `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()`
  - **DreamPhase.run()**: Same wrapping around forward + KL loss computation
  - **NightmarePhase.run()**: Same wrapping around forward pass
  - **CompressionPhase.run()**: Same wrapping around fine-tune loop
  - Each phase accepts optional `scaler` parameter; falls back to no-op if None
- Add `gradient_checkpointing: false` config key; enable via `model.gradient_checkpointing_enable()` in `Trainer.__init__()` when True
- **Files**: `nightmarenet/training/trainer.py`, `nightmarenet/training/phases.py`, `nightmarenet/utils/config.py`, `configs/default.yaml`

### Step 1.2: API Authentication & Rate Limiting *(parallel with 1.1)*
- Add `python-jose[cryptography]` and `slowapi` to `[api]` optional dependencies in `pyproject.toml`
- Create `nightmarenet/api/auth.py`:
  - `APIKeyMiddleware` — validates `X-API-Key` header against config/env var `NIGHTMARENET_API_KEY`
  - If env var not set, auth is disabled (dev mode) with startup warning
  - Return 401 with `ErrorResponse` on invalid key
  - Exempt `/api/v1/health` and `/docs` from auth
- In `nightmarenet/api/app.py`:
  - Import and add `APIKeyMiddleware`
  - Add `slowapi` rate limiter: 60 req/min per IP on distortion endpoints, 10 req/min on robustness
  - Tighten CORS: read allowed origins from `NIGHTMARENET_CORS_ORIGINS` env var (default: `["*"]` for backward compat)
- Update `nightmarenet/api/schemas.py`: Add `AuthErrorResponse` model
- Add tests in `tests/test_api.py`: auth header validation, rate limit 429 response, health endpoint bypass
- **Files**: `nightmarenet/api/auth.py` (new), `nightmarenet/api/app.py`, `nightmarenet/api/schemas.py`, `pyproject.toml`, `tests/test_api.py`

### Step 1.3: CI/CD Pipeline *(parallel with 1.1 and 1.2)*
- Create `.github/workflows/ci.yml`:
  - **Trigger**: push to main, pull requests
  - **Matrix**: Python 3.9, 3.11, 3.12 on ubuntu-latest
  - **Steps**: checkout → setup-python → pip install -e ".[dev,api]" → ruff check → pytest --cov=nightmarenet --cov-report=xml → upload coverage
  - **Caching**: pip cache
- Create `.github/workflows/release.yml`:
  - **Trigger**: tag push (v*)
  - **Steps**: build sdist + wheel → publish to PyPI (optional, placeholder)
- **Files**: `.github/workflows/ci.yml` (new), `.github/workflows/release.yml` (new)

---

## Phase 2: P1 — Observability, Scalability & Deployment *(depends on Phase 1)*

### Step 2.1: Wandb/TensorBoard Integration
- Add `wandb>=0.16` and `tensorboard>=2.14` to new `[tracking]` optional dependency group in `pyproject.toml`
- Create `nightmarenet/utils/tracking.py`:
  - `ExperimentTracker` class wrapping wandb/tensorboard/none backends
  - `__init__(backend, project_name, run_name, config)` — lazy init (only imports wandb if backend="wandb")
  - `log_metrics(metrics_dict, step)` — logs to chosen backend
  - `log_phase(cycle, phase, metrics)` — structured phase logging
  - `log_config(config)` — log hyperparameters
  - `finish()` — cleanup
  - Factory: `create_tracker_from_config(config)` — reads `tracking.backend` from config
- Add config keys to `configs/default.yaml` and `DEFAULT_CONFIG`:
  ```yaml
  tracking:
    backend: none  # none|wandb|tensorboard
    project: nightmarenet
    log_dir: logs/runs
  ```
- In `nightmarenet/training/trainer.py`:
  - Create tracker in `__init__()`
  - Call `tracker.log_phase()` after each phase result in `train()`
  - Call `tracker.log_config()` at start
  - Call `tracker.finish()` at end
- In `nightmarenet/evaluation/evaluator.py`:
  - Accept optional tracker; log each metric result after computation in `evaluate()`
- Add `--tracker` CLI flag to `scripts/train.py` and `scripts/evaluate.py` (overrides config)
- **Files**: `nightmarenet/utils/tracking.py` (new), `nightmarenet/training/trainer.py`, `nightmarenet/evaluation/evaluator.py`, `nightmarenet/utils/config.py`, `configs/default.yaml`, `scripts/train.py`, `scripts/evaluate.py`, `pyproject.toml`

### Step 2.2: Streaming Dataset Support *(parallel with 2.1)*
- In `nightmarenet/data/loader.py`:
  - Add `streaming: bool` parameter to `DatasetWrapper.__init__()`
  - When `streaming=True`: use `load_dataset(..., streaming=True)` → returns `IterableDataset`
  - Implement `_estimate_length()` for progress bars (optional peek at first N samples)
  - Keep non-streaming path unchanged for backward compat
  - Add `streaming: false` config key under `dataset` section
- In `nightmarenet/data/generator.py`:
  - Support `IterableDataset` input in `generate()`: use generator function instead of `.map()`
  - `DreamDatasetGenerator.generate()`: detect IterableDataset → yield distorted examples lazily
  - `NightmareDatasetGenerator.generate()`: same lazy distortion
  - Keep `.generate_and_save()` materializing (saving requires materialization)
- In `nightmarenet/training/trainer.py`:
  - `_tokenize_dataset()`: handle `IterableDataset` — use `torch.utils.data.IterableDataset` wrapper instead of `.map()` + DataLoader
- Add config keys: `dataset.streaming: false` in default.yaml and DEFAULT_CONFIG
- **Files**: `nightmarenet/data/loader.py`, `nightmarenet/data/generator.py`, `nightmarenet/training/trainer.py`, `nightmarenet/utils/config.py`, `configs/default.yaml`

### Step 2.3: Dockerfile & Docker Compose *(parallel with 2.1 and 2.2)*
- Create `Dockerfile`:
  - Multi-stage: builder (install deps) → runtime (slim image)
  - Base: `python:3.11-slim`
  - Install core + api dependencies
  - Copy source, set entrypoint to `uvicorn nightmarenet.api.app:app`
  - Expose port 8000
  - Non-root user for security
- Create `docker-compose.yml`:
  - `api` service: builds from Dockerfile, port mapping 8000:8000, env vars for API key and CORS
  - `train` service profile: runs training script with volume mounts for checkpoints/data/configs
  - Volume mounts: `./configs:/app/configs`, `./data:/app/data`, `./checkpoints:/app/checkpoints`
- Create `.dockerignore`: .git, .venv, __pycache__, checkpoints/, logs/, .pytest_cache/
- **Files**: `Dockerfile` (new), `docker-compose.yml` (new), `.dockerignore` (new)

---

## Phase 3: P2 — Research Capabilities *(depends on Phase 2)*

### Step 3.1: Multi-Model Architecture Support
- In `nightmarenet/utils/config.py`:
  - Add `model.type` config key: `causal_lm` (default) | `masked_lm` | `seq_classification`
  - Add `model.num_labels` for classification models
  - Add schema entries for new keys
- In `nightmarenet/training/trainer.py`:
  - Replace hardcoded `AutoModelForCausalLM` with model type dispatch:
    - `causal_lm` → `AutoModelForCausalLM`
    - `masked_lm` → `AutoModelForMaskedLM`
    - `seq_classification` → `AutoModelForSequenceClassification`
  - Abstract loss computation: causal uses `labels=input_ids`, masked uses `labels` from dataset, classification uses `labels` column
- In `nightmarenet/training/phases.py`:
  - `WakePhase.run()`: dispatch loss based on model type (currently assumes causal LM shift)
  - `DreamPhase.run()`: KL divergence computation works for all — logits shape may differ for classification
  - Pass `model_type` through config to phases
- In `nightmarenet/evaluation/metrics.py`:
  - `compute_perplexity()`: only valid for LM models; skip for classification
  - Add `classification_metrics()`: accuracy, F1, confusion matrix (using sklearn)
  - `recall_score()`: branch based on model type
- Update `configs/default.yaml` with new model type keys
- **Files**: `nightmarenet/training/trainer.py`, `nightmarenet/training/phases.py`, `nightmarenet/evaluation/metrics.py`, `nightmarenet/evaluation/evaluator.py`, `nightmarenet/utils/config.py`, `configs/default.yaml`

### Step 3.2: Learned Adversarial Distortions *(parallel with 3.1)*
- Create `nightmarenet/distortions/learned.py`:
  - `LearnedAdversarialGenerator` class:
    - Uses a small auxiliary model (e.g., `distilbert-base-uncased`) to generate adversarial token replacements
    - `__init__(model_name, device, strength)`: loads tokenizer + MLM model
    - `generate(text, strength)` → str: mask high-importance tokens (via attention/gradient), replace with MLM predictions that maximize confusion
    - `_get_token_importance(text)` → importance scores per token (attention-based)
    - `_adversarial_replace(text, token_indices)` → replace important tokens with confusing alternatives
  - Falls back to template-based if model unavailable (import error, OOM)
- In `nightmarenet/distortions/adversarial.py`:
  - Add `learned_adversarial` to distortion config options
  - In `apply_adversarial_distortions()`: if `learned` key in config, invoke `LearnedAdversarialGenerator`
  - Keep existing template functions as fallback
- Add config keys under `distortion.adversarial`:
  ```yaml
  adversarial:
    learned: 0.0  # weight for learned adversarial (0 = disabled)
    learned_model: distilbert-base-uncased
  ```
- Add tests in `tests/test_distortions.py` for learned adversarial (mock model for CI speed)
- **Files**: `nightmarenet/distortions/learned.py` (new), `nightmarenet/distortions/adversarial.py`, `nightmarenet/utils/config.py`, `configs/default.yaml`, `tests/test_distortions.py`

---

## Relevant Files

### Modified
| File | Changes |
|------|---------|
| `nightmarenet/training/trainer.py` | AMP, gradient checkpointing, tracker integration, multi-model dispatch, streaming tokenization |
| `nightmarenet/training/phases.py` | AMP scaler in all 4 phase runners |
| `nightmarenet/utils/config.py` | ~10 new config keys + schema entries |
| `configs/default.yaml` | All new config sections |
| `nightmarenet/api/app.py` | Auth middleware, rate limiting, CORS tightening |
| `nightmarenet/api/schemas.py` | AuthErrorResponse model |
| `nightmarenet/data/loader.py` | Streaming dataset support |
| `nightmarenet/data/generator.py` | Lazy IterableDataset distortion |
| `nightmarenet/evaluation/evaluator.py` | Tracker integration |
| `nightmarenet/evaluation/metrics.py` | Classification metrics, model-type branching |
| `nightmarenet/distortions/adversarial.py` | Learned adversarial integration |
| `pyproject.toml` | New dep groups: [tracking], updated [api] |
| `scripts/train.py` | --tracker CLI flag |
| `scripts/evaluate.py` | --tracker CLI flag |
| `tests/test_api.py` | Auth + rate limit tests |
| `tests/test_distortions.py` | Learned adversarial tests |

### New Files
| File | Purpose |
|------|---------|
| `nightmarenet/api/auth.py` | API key middleware |
| `nightmarenet/utils/tracking.py` | Experiment tracker abstraction |
| `nightmarenet/distortions/learned.py` | Learned adversarial generator |
| `.github/workflows/ci.yml` | CI pipeline |
| `.github/workflows/release.yml` | Release pipeline |
| `Dockerfile` | Container image |
| `docker-compose.yml` | Service orchestration |
| `.dockerignore` | Docker build exclusions |

---

## Verification

### Phase 1
1. Run `pytest tests/ -v` — all 159+ existing tests pass (no regressions)
2. Run `python scripts/train.py --config configs/default.yaml --dry-run` — verify AMP config loads correctly
3. Run training with `use_amp: true` on GPU — confirm ~2x speedup via timing, no NaN losses
4. Start API with `NIGHTMARENET_API_KEY=test123` → verify 401 without header, 200 with correct header, health exempt
5. Hit distortion endpoint rapidly → verify 429 after rate limit exceeded
6. Push to GitHub → verify CI workflow runs, all tests green across Python 3.9/3.11/3.12
7. Run `ruff check nightmarenet/ scripts/ tests/` — no lint errors in new code

### Phase 2
8. Run training with `tracking.backend: wandb` → verify metrics appear in wandb dashboard
9. Run training with `tracking.backend: tensorboard` → verify `tensorboard --logdir logs/runs` shows data
10. Run training with `dataset.streaming: true` → verify training starts without OOM, compare output quality
11. Run `docker build -t nightmarenet .` → verify image builds successfully
12. Run `docker-compose up api` → verify health endpoint responds at localhost:8000
13. Run `docker-compose run train` → verify training executes inside container

### Phase 3
14. Set `model.type: masked_lm`, `model.name: bert-base-uncased` → verify Wake phase trains with MLM loss
15. Set `model.type: seq_classification` with labeled dataset → verify classification training + F1/accuracy metrics
16. Set `distortion.adversarial.learned: 0.3` → verify learned adversarial distortions produce semantically confusing text
17. Run full pipeline with learned adversarial → verify training completes without errors

---

## Decisions
- **Auth approach**: API key via header (simple, stateless) over OAuth (complex, overkill for research tool). Disabled when env var unset for dev convenience.
- **Rate limiting**: Per-IP via slowapi (lightweight) over Redis-backed (infra overhead). Sufficient for single-instance.
- **Tracker abstraction**: Wrapper class over direct wandb calls — allows swapping backends without touching training code.
- **Streaming**: Opt-in via config flag — default remains in-memory for simplicity and backward compat.
- **Learned adversarial**: MLM-based token replacement over gradient-based (TextFooler) — avoids requiring differentiable text pipeline. Falls back to templates on error.
- **Multi-model**: Config-driven dispatch over class hierarchy — minimal code change, fits existing pattern.
- **Evaluation Standardization Layer**: ✅ Already implemented. `Evaluator` class + `metrics.py` module provides standardized `robustness_score()`, `hallucination_rate()`, `recall_score()`, `generalization_score()` with unified `evaluate()` dispatch, config-driven metric selection, `compare()` for baseline vs. trained, and markdown report generation. A dedicated `consistency_score()` is not needed separately — `robustness_score()` already measures performance consistency across distortion strengths (AUC over 9 levels). If a distinct "output stability" metric is needed later (e.g., variance across stochastic runs), it can be added as a new function in `metrics.py` and registered in `Evaluator.evaluate()` with zero architectural changes.
- **Excluded from scope**: Distributed training (DDP), Pydantic config migration, multilingual distortions, downstream benchmarks (GLUE) — deferred to future work.
