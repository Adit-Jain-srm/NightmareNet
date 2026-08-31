# Architecture Overview

This document maps the NightmareNet codebase for new contributors: how the modules depend on each other, how the training pipeline flows, and where the OSS core ends and the hosted server begins.

## Module relationships

The OSS core (`nightmarenet/`) is self-contained. The hosted server (`nightmarenet_server/`) and the Next.js `frontend/` build on top of it, but the core never imports from them.

```mermaid
graph TD
    CLI["nightmarenet.cli"]
    Pipeline["nightmarenet.pipeline.Pipeline"]
    Phases["nightmarenet.phases<br/>(ingest, optimize, prepare, train, evaluate, export)"]
    Training["nightmarenet.training<br/>(trainer, Wake/Dream/Nightmare/Compress)"]
    Distortions["nightmarenet.distortions<br/>(registry, base, engines)"]
    Compression["nightmarenet.compression"]
    Evaluation["nightmarenet.evaluation<br/>(Evaluator, metrics)"]
    API["nightmarenet.api.app<br/>(FastAPI)"]

    Server["nightmarenet_server<br/>(auth, DB, workers, search)"]
    Frontend["frontend<br/>(Next.js dashboard)"]

    CLI --> Pipeline
    CLI --> Distortions
    CLI --> Evaluation
    Pipeline --> Phases
    Phases --> Training
    Phases --> Evaluation
    Training --> Distortions
    Training --> Compression
    Evaluation --> Distortions
    API --> Pipeline
    API --> Distortions
    API --> Evaluation

    Server --> API
    Frontend --> API
    Frontend --> Server

    classDef core fill:#06B6D4,stroke:#0891B2,color:#020617
    classDef hosted fill:#818CF8,stroke:#6366F1,color:#020617
    class CLI,Pipeline,Phases,Training,Distortions,Compression,Evaluation,API core
    class Server,Frontend hosted
```

> [!IMPORTANT]
> The OSS core (`nightmarenet/`) must **not** import from `nightmarenet_server/`, and must not depend on PostgreSQL, Redis, Celery, or OAuth libraries. See the OSS/hosted boundary table in [`CONTRIBUTING.md`](../../CONTRIBUTING.md#architecture-pointers).

## Pipeline phase flow

`Pipeline.run()` (in [`nightmarenet/pipeline.py`](../../nightmarenet/pipeline.py)) executes a fixed sequence of orchestration phases, each implemented as a `Phase` subclass. `EvaluatePhase` internally invokes `ExportPhase`.

```mermaid
flowchart LR
    Ingest["IngestPhase<br/>load dataset"]
    Optimize["OptimizePhase<br/>adaption / data prep"]
    Prepare["PreparePhase<br/>dataloaders, trainer, distortion_fn"]
    Train["TrainPhase<br/>Wake -> Dream -> Nightmare -> Compress"]
    Evaluate["EvaluatePhase<br/>metrics + comparison"]
    Export["ExportPhase<br/>artifacts + report"]

    Ingest --> Optimize --> Prepare --> Train --> Evaluate --> Export
```

Each orchestration phase implements the abstract base in [`nightmarenet/phases/base.py`](../../nightmarenet/phases/base.py):

```python
class Phase(abc.ABC):
    name: str = "phase"

    @abc.abstractmethod
    def execute(self, context: PipelineContext) -> PhaseResult:
        ...
```

- **`PipelineContext`** is a dataclass holding all mutable run state (`run_id`, `config`, `dataset`, `train_dl`, `trainer`, `history`, `trained_results`, `comparison`, `report_md`, …). Each phase reads and mutates it in place.
- **`PhaseResult`** reports `success`, `phase_name`, an optional `error`, and a `data` dict.

The four *training* sub-phases (Wake → Dream → Nightmare → Compress) run **inside** `TrainPhase`; their implementations live in `nightmarenet/training/` (trainer and phase logic), not in `nightmarenet/phases/`.

## Evaluation flow

`EvaluatePhase` uses the `Evaluator` (in [`nightmarenet/evaluation/evaluator/core.py`](../../nightmarenet/evaluation/evaluator/core.py)), which dispatches string metric names to functions in [`nightmarenet/evaluation/metrics.py`](../../nightmarenet/evaluation/metrics.py):

```mermaid
flowchart TD
    Config["config.evaluation.metrics<br/>(list of names)"]
    Evaluator["Evaluator.evaluate()"]
    Recall["recall_score"]
    Gen["generalization_score"]
    Rob["robustness_score"]
    Hall["hallucination_rate"]
    Cls["classification_metrics"]

    Config --> Evaluator
    Evaluator --> Recall
    Evaluator --> Gen
    Evaluator --> Rob
    Evaluator --> Hall
    Evaluator --> Cls
```

See [Adding Metrics](adding-metrics.md) for how to extend this dispatch.

## Key entry points

| Entry point | Location | Role |
|---|---|---|
| `Pipeline` | `nightmarenet.pipeline` | Orchestrates the 4-phase cycle. |
| `main` | `nightmarenet.cli` | The `nightmarenet` console command. |
| `get_registry` | `nightmarenet.distortions.registry` | Lazy-singleton distortion registry. |
| `Evaluator` | `nightmarenet.evaluation.evaluator` | Multi-strength robustness scoring. |
| `app` | `nightmarenet.api.app` | FastAPI app for the OSS HTTP surface. |

## Server architecture

The hosted server (`nightmarenet_server/`) adds authentication, a multi-tenant database, Celery workers, and semantic experiment search on top of the OSS API. It is documented separately in [`docs/development/local-stack.md`](local-stack.md); the deployment topology and OSS/hosted split are described in [`docs/architecture/`](../architecture/).

## Related Documentation

- [Adding Distortions](adding-distortions.md) — extend the distortion registry.
- [Adding Metrics](adding-metrics.md) — extend the evaluator.
- [Local Stack](local-stack.md) — run the API + frontend locally.
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — the OSS/hosted boundary and entry points.
