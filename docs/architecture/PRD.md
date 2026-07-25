# NightmareNet — Product Requirements Document (PRD)

**Version**: 1.0
**Last Updated**: 2026-05-23
**Status**: Draft
**Owner**: NightmareNet Core Team

---

## 1. Executive Summary

NightmareNet is an autonomous AI self-improvement platform that implements a biologically-inspired sleep training paradigm for neural networks. By cycling models through four distinct phases — Wake (supervised learning), Dream (generative replay), Nightmare (adversarial stress-testing), and Compress (knowledge distillation/pruning) — the platform produces models that are measurably more robust, resilient to distribution shift, and resistant to catastrophic forgetting than those trained with conventional methods alone.

The platform addresses a critical gap in the ML tooling ecosystem: no existing tool combines adversarial example generation, catastrophic forgetting prevention, model compression, and experiment orchestration into a single coherent workflow. With the EU AI Act Article 15 mandating robustness testing for high-risk AI systems by August 2, 2026, there is a time-sensitive market opportunity for a compliance-ready robustness platform.

NightmareNet follows a hybrid open-source/commercial model. The open-source core (Apache 2.0) includes distortion engines, the training loop, evaluation metrics, and a CLI — designed to build community adoption and academic citations. The hosted platform adds multi-GPU orchestration, experiment tracking, team collaboration, compliance reporting, and enterprise security features, targeting paying customers from seed-stage AI startups through enterprise compliance teams.

---

## 2. Problem Statement

### The Pain

Machine learning models deployed in production face three persistent reliability challenges:

1. **Catastrophic Forgetting** — Models fine-tuned on new data lose performance on previously learned tasks. Teams waste cycles retraining from scratch or maintaining ensemble architectures.

2. **Adversarial Fragility** — Small, semantically-preserving perturbations to input data cause dramatic performance degradation. Production models fail silently on edge cases that differ slightly from training distribution.

3. **Robustness/Compliance Gap** — Regulatory frameworks (EU AI Act Article 15, NIST AI RMF) now require demonstrable robustness testing, but no turnkey tool generates the adversarial test suites, runs multi-strength evaluations, and produces audit-ready reports.

4. **Fragmented Tooling** — Teams currently stitch together TextAttack for adversarial generation, custom scripts for continual learning, separate pruning libraries, and ad-hoc evaluation — no single platform provides the full loop.

### Who Feels It

- **ML Engineers** building production models who discover robustness gaps only after deployment failures
- **AI Safety Researchers** who need systematic adversarial stress-testing at scale but lack integrated orchestration
- **Startup CTOs** shipping models under resource constraints (single GPU, small teams) who cannot afford enterprise MLOps platforms
- **Compliance Officers** at enterprises deploying AI in regulated sectors (finance, healthcare, legal) who need audit trails
- **Research Teams** at universities studying continual learning, adversarial robustness, or model compression who need reproducible experiment infrastructure

---

## 3. Target Users

### Persona 1: "Alex" — ML Engineer at Growth-Stage Startup

| Attribute | Details |
|-----------|---------|
| Role | Senior ML Engineer |
| Team Size | 3–8 person ML team |
| Environment | Cloud GPUs (1–4 A100s), CI/CD pipelines |
| Pain | Models degrade in production; no systematic robustness process |
| JTBD | "When I'm deploying a new model version, I want to run adversarial stress tests so I can be confident it won't fail on edge cases my users will encounter." |

### Persona 2: "Dr. Priya" — AI Safety Researcher

| Attribute | Details |
|-----------|---------|
| Role | Postdoc / Research Scientist |
| Team Size | Solo or 2–3 collaborators |
| Environment | University cluster (shared GPUs), local workstation (RTX 3090) |
| Pain | Adversarial evaluation is manual, non-reproducible, paper deadlines loom |
| JTBD | "When I'm writing a paper on adversarial robustness, I want reproducible multi-strength evaluation with standard metrics so I can compare my method against baselines fairly." |

### Persona 3: "Marcus" — CTO of Seed-Stage AI Startup

| Attribute | Details |
|-----------|---------|
| Role | Technical Co-founder / CTO |
| Team Size | 1–3 engineers total |
| Environment | Single RTX 3050 Ti / 4090, tight budget |
| Pain | Cannot afford MLOps platforms; needs robustness but has no dedicated infra team |
| JTBD | "When I'm iterating on our core model with limited GPU, I want an all-in-one training loop that handles robustness and compression so I can ship a production-ready model without enterprise tooling." |

### Persona 4: "Sarah" — Enterprise Compliance Lead

| Attribute | Details |
|-----------|---------|
| Role | AI Governance / Compliance Manager |
| Team Size | Cross-functional (legal, engineering, product) |
| Environment | Enterprise cloud (Azure/AWS), strict security policies |
| Pain | EU AI Act deadline approaching; no tool generates required robustness documentation |
| JTBD | "When I'm preparing our AI system for EU AI Act audit, I want automated robustness reports with quantitative metrics and adversarial test results so I can demonstrate Article 15 compliance." |

### Persona 5: "Jordan" — Research Team Lead

| Attribute | Details |
|-----------|---------|
| Role | Principal Researcher / Lab Director |
| Team Size | 5–15 researchers + students |
| Environment | Multi-GPU cluster, shared experiment infrastructure |
| Pain | Students waste time reinventing experiment tracking; results aren't comparable across papers |
| JTBD | "When I'm managing a research lab studying continual learning, I want a shared experiment platform with standardized evaluation so my team can build on each other's results without reimplementing baselines." |

---

## 4. User Stories

### Must Have (M)

| ID | Persona | Story | Acceptance Criteria |
|----|---------|-------|---------------------|
| US-01 | Alex | As an ML engineer, I want to run a full Wake→Dream→Nightmare→Compress cycle via CLI so that I can improve my model's robustness with a single command. | `nightmarenet train --config cycle.yaml` completes all 4 phases, outputs metrics JSON |
| US-02 | Alex | As an ML engineer, I want to evaluate model robustness across 9 distortion strengths so that I can quantify degradation curves. | `/api/v1/evaluate/robustness` returns scores at each strength level |
| US-03 | Alex | As an ML engineer, I want dream distortions (paraphrase, shuffle, noise) to generate augmented training data so that my model learns invariant representations. | Dream phase produces distorted dataset, perplexity remains within 2x of original |
| US-04 | Alex | As an ML engineer, I want nightmare distortions (adversarial, semantic attack) to stress-test my model so that I can find failure modes before deployment. | Nightmare phase generates adversarial examples that reduce accuracy by ≥20% |
| US-05 | Marcus | As a startup CTO, I want the training loop to support mixed-precision (FP16) and gradient checkpointing so that I can train on my 4GB RTX 3050 Ti. | Training runs without OOM on 4GB GPU with `use_amp: true` |
| US-06 | Dr. Priya | As a researcher, I want reproducible experiments via YAML configs and fixed seeds so that I can replicate results exactly. | Same config + seed produces identical metrics across runs |
| US-07 | All | As a user, I want a REST API for distortion generation so that I can integrate NightmareNet into existing pipelines. | POST `/api/v1/generate/dream` and `/nightmare` return distorted text within 2s |
| US-08 | Sarah | As a compliance lead, I want quantitative robustness scores (0–1 scale) so that I can include them in audit reports. | `robustness_score()` returns float in [0,1] with documented methodology |
| US-09 | Alex | As an ML engineer, I want model compression (pruning + bottleneck) in the compress phase so that I can reduce model size while retaining knowledge. | Compress phase reduces parameters by configured ratio, accuracy drop <5% |
| US-10 | All | As a user, I want a health endpoint so that I can monitor API availability. | GET `/api/v1/health` returns 200 with version and status |

### Should Have (S)

| ID | Persona | Story | Acceptance Criteria |
|----|---------|-------|---------------------|
| US-11 | Alex | As an ML engineer, I want experiment tracking (W&B/TensorBoard) so that I can visualize training progress across phases. | Metrics appear in configured backend after each phase |
| US-12 | Jordan | As a research lead, I want to compare baseline vs. trained model metrics so that I can quantify improvement from the sleep cycle. | `compare()` method returns side-by-side metric table |
| US-13 | Alex | As an ML engineer, I want API authentication (API keys) so that I can secure my deployment. | Requests without valid API key receive 401 |
| US-14 | Marcus | As a startup CTO, I want rate limiting on API endpoints so that I can prevent abuse without additional infrastructure. | Excessive requests receive 429 with retry-after header |
| US-15 | Dr. Priya | As a researcher, I want streaming dataset support so that I can train on datasets larger than RAM. | Training with `streaming: true` processes data without OOM |
| US-16 | Alex | As an ML engineer, I want Docker images so that I can deploy reproducibly across environments. | `docker-compose up api` starts serving within 30s |
| US-17 | Sarah | As a compliance lead, I want markdown evaluation reports so that I can attach them to compliance documentation. | `Evaluator.generate_report()` outputs formatted markdown |
| US-18 | Jordan | As a research lead, I want multi-model support (causal LM, masked LM, classification) so that my team can use NightmareNet on diverse tasks. | Training succeeds with each model type configured |
| US-19 | Alex | As an ML engineer, I want early stopping based on validation loss so that I don't waste compute on overfitting. | Training stops automatically when val loss plateaus for N epochs |
| US-20 | Dr. Priya | As a researcher, I want learned adversarial distortions (MLM-based) so that I can generate more realistic adversarial examples. | Learned distortions produce semantically valid adversarial text |

### Could Have (C)

| ID | Persona | Story | Acceptance Criteria |
|----|---------|-------|---------------------|
| US-21 | Jordan | As a research lead, I want distributed training (DDP) support so that I can scale to multi-GPU. | Training distributes across N GPUs with near-linear scaling |
| US-22 | Sarah | As a compliance lead, I want PDF export of robustness reports so that I can submit to auditors directly. | Export generates PDF with charts and metrics tables |
| US-23 | Alex | As an ML engineer, I want a web dashboard showing real-time training progress so that I can monitor cycles without CLI. | Frontend displays live phase progress, loss curves, metrics |
| US-24 | Marcus | As a startup CTO, I want YAML config validation with clear error messages so that I can catch misconfigurations before training. | Invalid configs produce human-readable error with field path |
| US-25 | Dr. Priya | As a researcher, I want GLUE benchmark integration so that I can report standard NLU scores alongside robustness metrics. | GLUE tasks run through evaluation pipeline and report standard metrics |
| US-26 | All | As a user, I want a pipeline orchestrator (start/stop/status) via API so that I can manage long-running training programmatically. | POST/GET/DELETE `/api/v1/pipeline/*` manage training lifecycle |

### Won't Have (W) — This Version

| ID | Story | Rationale |
|----|-------|-----------|
| US-W1 | Multi-language distortion support | Requires extensive linguistic resources; English-first MVP |
| US-W2 | Visual/image model support | Text-only scope for v1; vision adds major complexity |
| US-W3 | Fine-grained RBAC with team roles | Enterprise feature for hosted platform v2 |
| US-W4 | Custom model architecture plugins | SDK extension point deferred to platform maturity |
| US-W5 | Real-time collaboration on experiments | Requires WebSocket infrastructure; out of scope for core |

---

## 5. Success Metrics

### Adoption Metrics

| KPI | Target (6 months) | Measurement |
|-----|-------------------|-------------|
| GitHub Stars | 1,000+ | GitHub API |
| PyPI Monthly Downloads | 5,000+ | PyPI stats |
| Active API Users (hosted) | 200+ | API key usage analytics |
| Academic Citations | 3+ | Google Scholar |
| Community Contributors | 15+ | GitHub contributor count |

### Product Quality Metrics

| KPI | Target | Measurement |
|-----|--------|-------------|
| Test Pass Rate | 100% on CI | pytest exit code |
| API Latency (p95) | <500ms distortion, <2s evaluation | APM monitoring |
| Training Throughput | ≥80% GPU utilization on supported hardware | nvidia-smi metrics |
| Robustness Improvement | ≥15% robustness score increase after 1 cycle | Automated benchmark |
| Compression Ratio | 20–40% parameter reduction with <5% accuracy loss | Evaluation pipeline |

### Business Metrics (Hosted Platform)

| KPI | Target (12 months) | Measurement |
|-----|---------------------|-------------|
| Paid Customers | 25+ | Stripe subscriptions |
| Monthly Recurring Revenue | $10K+ | Billing system |
| Enterprise Contracts | 3+ | CRM |
| Churn Rate | <5% monthly | Cohort analysis |
| Time to First Value | <15 min from signup to first robustness report | Onboarding funnel |

---

## 6. Functional Requirements

### 6.1 Training Engine

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-T01 | Execute 4-phase training cycle (Wake→Dream→Nightmare→Compress) | M |
| FR-T02 | Support configurable phase order and repetition | S |
| FR-T03 | Mixed-precision training (AMP) with automatic loss scaling | M |
| FR-T04 | Gradient checkpointing for memory-constrained GPUs | M |
| FR-T05 | Early stopping based on configurable patience/metric | S |
| FR-T06 | Learning rate scheduling (linear warmup, cosine decay) | S |
| FR-T07 | Gradient accumulation for effective batch size control | S |
| FR-T08 | Checkpoint save/resume at phase boundaries | M |
| FR-T09 | Multi-model architecture support (causal LM, masked LM, classification) | S |
| FR-T10 | Streaming dataset support for memory-limited environments | S |

### 6.2 Distortion Engine

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-D01 | Dream distortions: paraphrase, word shuffle, noise injection, synonym replacement | M |
| FR-D02 | Nightmare distortions: adversarial token replacement, semantic negation, entity swap | M |
| FR-D03 | Configurable distortion strength (0.0–1.0 continuous) | M |
| FR-D04 | Deterministic output via seed parameter | M |
| FR-D05 | Learned adversarial distortions via MLM model | C |
| FR-D06 | Custom distortion plugin interface | W |
| FR-D07 | Batch distortion generation for dataset augmentation | M |
| FR-D08 | Distortion composition (multiple distortions per sample) | S |

### 6.3 Evaluation & Metrics

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-E01 | Robustness score: AUC across 9 distortion strength levels | M |
| FR-E02 | Perplexity computation for language models | M |
| FR-E03 | Recall/F1 for classification tasks | S |
| FR-E04 | Hallucination rate detection | S |
| FR-E05 | Generalization score on OOD data | S |
| FR-E06 | Baseline vs. trained comparison with statistical significance | S |
| FR-E07 | Markdown report generation with tables and visualizations | S |
| FR-E08 | Metrics export (JSON, CSV) for external analysis | M |

### 6.4 Compression

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-C01 | Magnitude-based weight pruning with configurable ratio | M |
| FR-C02 | Information bottleneck layer insertion | S |
| FR-C03 | Post-compression fine-tuning (knowledge retention) | M |
| FR-C04 | Sparsity statistics reporting | M |
| FR-C05 | Knowledge distillation from full model to compressed student | C |

### 6.5 API Layer

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-A01 | REST API with OpenAPI documentation | M |
| FR-A02 | Dream/Nightmare distortion generation endpoints | M |
| FR-A03 | Multi-strength robustness evaluation endpoint | M |
| FR-A04 | Pipeline lifecycle management (create/status/cancel) | S |
| FR-A05 | File upload for custom datasets | S |
| FR-A06 | Training configuration preview/validation endpoint | S |
| FR-A07 | Health check with version and dependency status | M |
| FR-A08 | API key authentication with per-key rate limits | S |
| FR-A09 | Rate limiting (per-IP and per-key) | S |
| FR-A10 | CORS configuration for cross-origin frontend | M |

### 6.6 Frontend (Dashboard)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-F01 | Real-time pipeline status display | S |
| FR-F02 | Interactive distortion demo (input → distorted output) | M |
| FR-F03 | Training configuration builder UI | S |
| FR-F04 | Robustness evaluation visualization | S |
| FR-F05 | File upload interface for datasets | S |
| FR-F06 | Dark theme cyberpunk aesthetic | M |
| FR-F07 | Responsive design (desktop + tablet) | S |

---

## 7. Non-Functional Requirements

### 7.1 Performance

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| API Response Time (distortion) | p50 < 200ms, p95 < 500ms, p99 < 1s | APM traces |
| API Response Time (evaluation) | p50 < 1s, p95 < 3s, p99 < 5s | APM traces |
| Training Throughput | ≥80% GPU utilization (single GPU) | nvidia-smi |
| Concurrent API Requests | 50 simultaneous connections | Load testing |
| Cold Start (API) | <5s to first response | Health check timing |
| Frontend TTI | <2s on 3G connection | Lighthouse |

### 7.2 Scalability

| Requirement | Target |
|-------------|--------|
| Dataset Size | Up to 10M samples (streaming mode) |
| Model Size | Up to 1B parameters (with gradient checkpointing) |
| API Horizontal Scaling | Stateless design; scale via replicas behind LB |
| Pipeline Concurrency | Up to 64 simultaneous pipeline runners (configurable) |
| Experiment Storage | Unbounded (external tracking backends) |

### 7.3 Security

| Requirement | Target |
|-------------|--------|
| API Authentication | API key validation (header-based) |
| Transport Encryption | TLS 1.2+ in production |
| Secret Management | Environment variables; no hardcoded credentials |
| Dependency Scanning | Automated via Dependabot/Snyk in CI |
| Input Validation | Pydantic models for all request bodies |
| Rate Limiting | Per-IP and per-key configurable limits |
| CORS | Configurable allowed origins (default restrictive in prod) |

### 7.4 Availability & Reliability

| Requirement | Target |
|-------------|--------|
| API Uptime (hosted) | 99.5% monthly |
| Training Fault Tolerance | Checkpoint resume after crash |
| Graceful Degradation | API serves cached responses if GPU unavailable |
| Error Handling | Structured error responses with correlation IDs |
| Logging | Structured JSON logs with configurable verbosity |

### 7.5 Compatibility

| Requirement | Target |
|-------------|--------|
| Python Version | 3.9, 3.10, 3.11, 3.12 |
| PyTorch Version | ≥2.0 |
| OS Support | Linux (primary), Windows (development), macOS (development) |
| GPU Support | NVIDIA CUDA (required for training), CPU-only for inference/API |
| Container Runtime | Docker 20.10+, Podman 4+ |

---

## 8. Scope Boundaries

### What We ARE Building (v1)

- Text-domain robustness training platform (NLP/LLM focused)
- 4-phase sleep-inspired training cycle with configurable parameters
- REST API for programmatic access to distortion and evaluation
- CLI for local training and evaluation workflows
- Web dashboard for interactive demo and monitoring
- Open-source core library (pip-installable)
- Hosted platform MVP (single-tenant, API-key auth)

### What We Are NOT Building (v1)

| Excluded | Rationale |
|----------|-----------|
| Computer vision / image model support | Text-only scope reduces complexity; vision deferred to v2 |
| Multi-language distortion support | English-first; requires per-language linguistic resources |
| Real-time collaborative editing | Requires WebSocket infra; not needed for research workflows |
| Custom model architecture plugins | SDK extension point requires stable API; premature |
| Mobile applications | Web-first; mobile adds platform complexity |
| Data labeling / annotation tools | Orthogonal to robustness; better served by existing tools |
| Model serving / inference optimization | Focus is training robustness, not serving infrastructure |
| Multi-cloud orchestration | Single-cloud deployment initially (Azure or AWS) |
| Fine-grained RBAC / team management | Enterprise feature for hosted platform v2 |
| Automated retraining pipelines (MLOps) | Beyond scope; integrates with existing MLOps tools instead |

---

## 9. Assumptions and Dependencies

### Assumptions

| ID | Assumption | Risk if Wrong |
|----|-----------|---------------|
| A1 | EU AI Act Article 15 enforcement creates demand by Aug 2026 | Market timing miss; pivot to research-first GTM |
| A2 | Single-GPU training is sufficient for MVP target users | Power users churn; accelerate DDP support |
| A3 | Text/NLP is the highest-value entry point | Miss vision/multimodal demand; requires new distortion engines |
| A4 | Open-source adoption converts to hosted platform revenue | Revenue gap; explore consulting/support model |
| A5 | API-first approach provides faster time-to-value than SDK-only | Users prefer CLI; add first-class CLI experience |
| A6 | PyTorch ecosystem dominance continues | Framework lock-in risk; abstract model interface |
| A7 | Academic citations drive awareness in target market | Need parallel developer marketing channels |

### Dependencies

| ID | Dependency | Type | Risk |
|----|-----------|------|------|
| D1 | PyTorch ≥2.0 | Runtime | Low — stable, well-maintained |
| D2 | Hugging Face Transformers | Runtime | Low — industry standard |
| D3 | FastAPI + Pydantic v2 | Runtime | Low — stable, active development |
| D4 | NVIDIA CUDA toolkit | Infrastructure | Medium — required for GPU training |
| D5 | Next.js 14 + Tailwind CSS v4 | Frontend | Low — stable ecosystem |
| D6 | W&B / TensorBoard | Optional integration | Low — optional, graceful fallback |
| D7 | Docker / container runtime | Deployment | Low — ubiquitous |
| D8 | GitHub Actions | CI/CD | Low — standard, replaceable |
| D9 | Cloud GPU availability (A100/H100) | Hosted platform | Medium — supply constraints |
| D10 | Stripe billing integration | Hosted platform | Low — mature API |

---

## 10. Requirements Traceability Matrix

| User Story | Functional Req | Module | Test Coverage |
|------------|---------------|--------|--------------|
| US-01 | FR-T01, FR-T08 | training/trainer.py, training/phases.py | test_training.py::test_full_cycle |
| US-02 | FR-E01 | evaluation/metrics.py, evaluation/evaluator.py | test_evaluation.py::test_robustness_score |
| US-03 | FR-D01, FR-D03, FR-D04 | distortions/text.py, distortions/semantic.py | test_distortions.py::test_dream_* |
| US-04 | FR-D02, FR-D03 | distortions/adversarial.py | test_distortions.py::test_nightmare_* |
| US-05 | FR-T03, FR-T04 | training/trainer.py, training/phases.py | test_training.py::test_amp_training |
| US-06 | FR-D04, FR-T01 | utils/config.py | test_config.py::test_seed_reproducibility |
| US-07 | FR-A02 | api/app.py | test_api.py::test_distortion_endpoints |
| US-08 | FR-E01, FR-E07 | evaluation/metrics.py, evaluation/evaluator.py | test_evaluation.py::test_score_range |
| US-09 | FR-C01, FR-C03 | compression/pruning.py | test_compression.py::test_pruning |
| US-10 | FR-A07 | api/app.py | test_api.py::test_health |
| US-11 | FR-T01 | utils/tracking.py | test_tracking.py::test_wandb_logging |
| US-12 | FR-E06 | evaluation/evaluator.py | test_evaluation.py::test_compare |
| US-13 | FR-A08 | api/auth.py | test_api.py::test_auth_required |
| US-14 | FR-A09 | api/app.py | test_api.py::test_rate_limit |
| US-15 | FR-T10 | data/loader.py | test_data.py::test_streaming |
| US-16 | FR-A01 | Dockerfile, docker-compose.yml | CI docker build test |
| US-17 | FR-E07 | evaluation/evaluator.py | test_evaluation.py::test_report_generation |
| US-18 | FR-T09 | training/trainer.py | test_training.py::test_model_types |
| US-19 | FR-T05 | training/trainer.py | test_training.py::test_early_stopping |
| US-20 | FR-D05 | distortions/learned.py | test_distortions.py::test_learned_adversarial |
| US-21 | — (deferred) | training/distributed.py | — |
| US-22 | FR-E07 | evaluation/evaluator.py | — |
| US-23 | FR-F01, FR-F04 | frontend/src/components/ | E2E tests |
| US-24 | FR-A06 | utils/config.py, utils/validation.py | test_config.py::test_validation |
| US-25 | FR-E01 | evaluation/glue.py | test_evaluation.py::test_glue |
| US-26 | FR-A04 | api/app.py, pipeline_runner.py | test_api.py::test_pipeline_lifecycle |

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| Wake Phase | Standard supervised fine-tuning on real-world (clean) training data |
| Dream Phase | Training on generatively augmented data (paraphrases, shuffles) to build invariant representations |
| Nightmare Phase | Training against adversarial examples designed to maximize model confusion |
| Compress Phase | Model size reduction via pruning/distillation while retaining learned robustness |
| Distortion Strength | Float 0.0–1.0 controlling intensity of text perturbation |
| Robustness Score | AUC of model performance across 9 distortion strength levels (0 = fragile, 1 = fully robust) |
| Catastrophic Forgetting | Loss of previously learned knowledge when fine-tuning on new data |
| Sleep Cycle | One complete Wake→Dream→Nightmare→Compress iteration |

## Appendix B: Competitive Landscape

| Competitor | Strengths | NightmareNet Differentiator |
|-----------|-----------|----------------------------|
| TextAttack | Mature adversarial NLP library | No training loop, no compression, no orchestration |
| Adversarial Robustness Toolbox (ART) | Broad attack/defense library | No sleep paradigm, no dream phase, no integrated pipeline |
| Avalanche (ContinualAI) | Continual learning framework | No adversarial generation, no compression phase |
| NVIDIA NeMo Guardrails | LLM safety rails | Runtime guardrails only, no training-time robustness |
| Robust Intelligence (RIME) | Enterprise AI testing | Closed-source, evaluation-only (no training improvement) |
| NightmareNet | Unified 4-phase pipeline | Only tool combining generation + adversarial + compression + orchestration |
