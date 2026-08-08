# Contributing to NightmareNet

Thank you for helping improve NightmareNet. This project uses a **research-first, verification-driven** workflow: every change is justified, tested, and traceable. The sections below take you from a fresh clone to a merged PR.

---

## Before You Start

> **Please complete these steps before opening a Pull Request:**

1. **Star this repository** — It helps us gauge community interest and prioritize features.
2. **Follow [@Adit-Jain-srm](https://github.com/Adit-Jain-srm)** — Stay updated on releases, related projects, and research.
3. **Read this entire guide** — PRs that don't follow the coding standards or skip tests will be asked to revise.

---

## Table of Contents

1. [Before you start](#before-you-start)
2. [Opening issues](#opening-issues)
3. [Issue assignment rules](#issue-assignment-rules)
4. [Code philosophy](#code-philosophy)
5. [Local development setup](#1-local-development-setup)
6. [Architecture pointers](#2-architecture-pointers)
7. [Adding a new distortion](#3-adding-a-new-distortion)
8. [Coding standards](#4-coding-standards)
9. [Documentation](#5-documentation)
10. [PR checklist](#6-pr-checklist)
11. [ECSoC'26 Contributors](#7-ecsoc26-contributors)
12. [Where to ask for help](#8-where-to-ask-for-help)

---

## Opening Issues

Before opening a new issue, **search existing issues** to avoid duplicates. If you find a related issue, comment on it instead of creating a new one.

### Required Format

Every issue must follow this structure:

**Title:** `[Type]: Short descriptive title`

Types: `[Feature]`, `[Bug]`, `[Docs]`, `[Refactor]`, `[Test]`, `[Infra]`

**Body (mandatory sections):**

1. **Problem / Motivation** - What's missing or broken? Why does it matter? Reference actual files, error messages, or user workflows.

2. **Proposed Solution** - Your suggested approach. Include:
   - Which files/modules would be affected
   - Key design decisions and tradeoffs
   - Any new dependencies required

3. **Acceptance Criteria** - Concrete, checkable items that define "done". Use checkboxes:
   ```
   - [ ] Function X returns correct output for input Y
   - [ ] Tests added covering the new behavior
   - [ ] Documentation updated
   ```

4. **Scope / Difficulty Estimate** - Is this a 1-hour fix or a multi-day feature? Help us label it correctly.

**Issues that will be closed without action:**
- One-sentence issues with no proposed solution ("Add tests")
- Issues that duplicate existing functionality without checking the codebase
- Issues requesting features already listed in other open issues
- Issues with no acceptance criteria

### Before You Open

- [ ] Searched existing open AND closed issues for duplicates
- [ ] Read the relevant source code to confirm the gap exists
- [ ] Checked `tests/` to see if what you're proposing is already covered
- [ ] Included file paths and line numbers where relevant

---

## Issue Assignment Rules

Assignments are handled transparently. These rules apply to all contributors equally.

### Requesting Assignment

To request assignment on an issue, comment with:
1. A brief explanation of **your planned approach** (not just "assign me")
2. Which files you'll modify
3. Estimated timeline (days, not weeks)

**Bad:** "Please assign this to me."
**Good:** "I'd like to work on this. Plan: add a `healthcheck` directive to the `api` service in docker-compose.yml using curl against /api/v1/health. Will also add a `start_period` of 15s. Should take about 1 hour."

### Assignment Priority

| Scenario | Who gets assigned |
|----------|-------------------|
| Single request | That person (if approach is reasonable) |
| Multiple requests, all new contributors | First-come-first-served (earliest comment timestamp) |
| Multiple requests, one has better approach | Better approach wins regardless of timing |
| Requester already has 2+ open assigned issues without PRs | Skipped in favor of the next requester |

### Concurrent Assignments (Guidelines, not hard limits)

We encourage contributors to focus on delivering quality over quantity. As a general guideline:

- **New contributors:** Start with 1 issue to build familiarity with the codebase and review process
- **Returning contributors:** Take on more as you're comfortable, but avoid having multiple stale assignments
- **The real rule:** If your existing assignments have no open PRs or progress updates, new requests may be deprioritized in favor of contributors who are actively delivering

### Unassignment

You will be unassigned if:
- 7 days pass with no PR and no progress update comment
- You request assignment on a new issue while your current one has no activity
- Your approach comment shows you haven't read the existing codebase (e.g., proposing to add something that already exists)

### Conflict Resolution

If two people request the same issue simultaneously (within 1 hour):
1. The person with the more detailed, code-aware approach comment wins
2. If approaches are equally strong, the person with fewer current assignments wins
3. If still tied, the earlier timestamp wins

### Pro Tips

- Comment your approach BEFORE asking for assignment - it shows you've done research
- Small PRs merge faster than large ones - if an issue is big, ask if it can be split into sub-issues
- If you're stuck, comment on the issue asking for help - don't go silent for a week
- **We merge quickly.** Focus on completing your current assigned issue before requesting new ones. Deliver first, then pick up more.
- **Think you can do it better?** If an issue is assigned but has no PR or progress after a few days, feel free to comment with your approach. Quality implementations are always welcome - we'd rather merge the best solution regardless of who was "first."
- All assignment decisions are at the maintainer's discretion based on these guidelines. The goal is shipping great code, not bureaucracy.

---

## Code Philosophy

We value **modularity, clarity, and maintainability** over cleverness. Every contribution should:

- **Single responsibility** — One function does one thing. One module owns one concern.
- **Small, focused files** — If a file exceeds 400 lines, consider splitting.
- **Explicit over implicit** — Prefer clear parameter names, type hints, and docstrings over magic.
- **No god objects** — Don't make a class that does everything. Compose small, testable units.
- **Fail fast, fail loud** — Validate inputs early. Raise descriptive errors with context.

### AI-Generated Code Disclosure

If your contribution includes **AI-generated code** (Copilot, ChatGPT, Claude, Cursor, etc.), you must:

1. **Disclose it** in the PR description: "This PR includes AI-assisted code generation."
2. **Review every line** — You are responsible for correctness, not the AI. AI-generated code with obvious bugs or hallucinated APIs will be rejected.
3. **Understand what it does** — Be prepared to explain any code in your PR during review.

We welcome AI-assisted contributions. We reject blindly pasted AI output.

### UI/UX Changes

Any PR that changes the frontend must include:

- **Before/after screenshots** (or a short screen recording) in the PR description
- **Mobile viewport** screenshot (375px width) if the change affects layout
- **Dark + light mode** screenshots if the change affects colors/theming
- **Accessibility check** — describe how keyboard navigation and screen readers interact with your change

---

## 1. Local development setup

### Prerequisites

- Python **3.12** is the recommended development version. The package supports 3.9-3.12; CUDA wheels are easiest on 3.12.
- Git, Node.js 20+ (only if you touch `frontend/`), and Docker (only if you touch the hosted-platform infra).
- Optional: NVIDIA GPU. The repo is dev-tested on a 4 GB RTX 3050 Ti.

### Clone and create a venv

```bash
git clone https://github.com/Adit-Jain-srm/NightmareNet.git
cd NightmareNet

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

### Install in editable mode with all dev tools

```bash
pip install -U pip
pip install -e ".[dev,api]"
```

The `dev` extra brings in `pytest`, `ruff`, `mypy`, and the test fixtures. The `api` extra brings in `fastapi`, `uvicorn`, and `slowapi` for the FastAPI service.

### Pre-commit hooks (recommended)

```bash
pip install pre-commit
pre-commit install
```

This runs `ruff` and a small set of fast checks on every commit. To run on the whole repo at once:

```bash
pre-commit run --all-files
```

### Verify the environment

```bash
pytest --cov=nightmarenet --cov-report=term-missing tests/ -v --tb=short   # 502+ tests, all should pass
ruff check .                         # zero errors expected
mypy nightmarenet/                   # type-check the OSS core
```

If you also touched the dashboard:

```bash
cd frontend
npm install
npm run build                        # production build
npm run dev                          # dev server on :3000
```

### Start the API for ad-hoc testing

```bash
uvicorn nightmarenet.api.app:app --reload --port 8000
```

Hit `http://127.0.0.1:8000/api/v1/health` to confirm.

---

## 2. Architecture pointers

NightmareNet has a strict OSS / hosted boundary. Treat it as a hard constraint when adding code.

| Package | Purpose | Allowed dependencies |
|---------|---------|---------------------|
| `nightmarenet/` | OSS core: distortions, training loop, evaluation, CLI, FastAPI inference endpoints | `torch`, `transformers`, `pydantic`, `fastapi`, `pyyaml`, `slowapi` (optional) |
| `nightmarenet_server/` *(future)* | Hosted platform: auth, multi-tenant DB, Celery workers, billing | OSS core + `sqlalchemy`, `redis`, `celery`, `stripe`, `psycopg2` |
| `frontend/` | Next.js 14 dashboard, design system, charts | npm ecosystem only; talks to OSS API or hosted API via `NEXT_PUBLIC_API_URL` / rewrites |

> [!IMPORTANT]
> The OSS core **must not** import anything from `nightmarenet_server`, and **must not** depend on PostgreSQL, Redis, Celery, OAuth providers, or any hosted-only library. If your change touches both, propose the boundary explicitly in the PR description and split the patches.

### Key entry points

- `nightmarenet.pipeline.Pipeline` — orchestrator for the 4-phase cycle
- `nightmarenet.cli.main` — the `nightmarenet` console entry point
- `nightmarenet.distortions.registry.get_registry` — the lazy-singleton plugin registry
- `nightmarenet.evaluation.evaluator.Evaluator` — multi-strength robustness scoring
- `nightmarenet.api.app` — FastAPI app exposing the OSS HTTP surface

### Documentation map

- [`docs/architecture/PRD.md`](docs/architecture/PRD.md) — product requirements, personas, success metrics, requirements traceability
- [`docs/architecture/TRD.md`](docs/architecture/TRD.md) — technical requirements
- [Interactive API docs](http://localhost:8000/docs) (auto-generated by FastAPI) — OpenAPI spec for the OSS HTTP surface
- [`docs/research/paper-draft.md`](docs/research/paper-draft.md) — academic paper draft (cite this in PRs that touch the algorithm)
- [`docs/research/benchmark-v1.md`](docs/research/benchmark-v1.md) — reproducible benchmark methodology

---

## 3. Adding a new distortion

Distortions are first-class plugins. The full walkthrough is in [`notebooks/03_custom_distortions.ipynb`](notebooks/03_custom_distortions.ipynb); the short version follows.

### The signature

Every distortion must match:

```python
from typing import Optional

DistortionFn = Callable[[str, float, Optional[int]], str]
```

That is: take a string, a strength in `[0.0, 1.0]`, and an optional seed; return the distorted string.

### Registration

For an in-tree distortion, drop a module under `nightmarenet/distortions/your_engine.py` exposing a `distort(text, strength, seed)` function and add it to the registry's `_register_builtins` in `nightmarenet/distortions/registry.py`. For a third-party plugin shipped as a separate package, expose a `register_distortion` decorator pattern (see notebook 03):

```python
from nightmarenet.distortions.registry import get_registry

def register_distortion(name, *, phase='custom', description=''):
    def decorator(fn):
        get_registry().register(name, fn, metadata={'phase': phase, 'description': description})
        return fn
    return decorator

@register_distortion('homoglyph', phase='nightmare', description='Latin -> Cyrillic swap')
def homoglyph(text, strength, seed=None):
    ...
```

### Tests

Mirror the package layout under `tests/`. At minimum:

1. **Determinism** — same `(text, strength, seed)` produces the same output across runs.
2. **Strength 0** is approximately a no-op.
3. **Strength 1** produces a measurable change.
4. **Empty input** returns empty without raising.
5. **Registry round-trip** — `get_registry().apply('your_engine', ...)` returns the same string as calling the function directly.

### Documentation

- Add a row to the README's "Distortion Types" or expand the relevant section.
- If your distortion is a known adversarial attack from a paper, cite the paper in the module docstring and in `docs/research/paper-draft.md` Related Work.

---

## 4. Coding standards

### Python

- **Line length:** 100 (enforced by ruff).
- **Ruff rules:** `E, F, W, I, N, UP, B`. We ignore `UP007` and `UP045` to keep `Union[X, Y]` available in 3.9-targeted code.
- **Imports:** isort via ruff. Order: stdlib, third-party, local; alphabetical within each group.
- **Type hints:**
  - Use `Union[X, Y]` and `Optional[X]` — **not** `X | Y` — in any code path that runs on Python 3.9.
  - Use `from __future__ import annotations` everywhere **except** modules under `nightmarenet/api/` that use FastAPI `Body(...)`. The future import breaks Pydantic v2 at runtime there. Prefer module-level singletons for `Body(...)` defaults to satisfy `B008`.
- **Docstrings:** Google style on public APIs only. Internal helpers can be terse.
- **Type checking:** `nightmarenet/` is checked with `mypy --strict`. Pre-existing errors are frozen in `mypy_baseline.txt` and don't block CI, but **new files and new lines must be clean** — see [`docs/development/code-style.md`](docs/development/code-style.md) for the full policy and how to update the baseline.
- **Errors:** raise with context (`raise X("...") from e`); never bare `raise X`.
- **No NaN/Inf in metrics:** wrap suspicious arithmetic with the helpers in `nightmarenet/evaluation/metrics.py`.
- **Logging:** use module loggers (`logger = logging.getLogger(__name__)`); don't `print` in library code.

### Frontend

- TypeScript only. No `any` in committed code — `@typescript-eslint/no-explicit-any` is enforced as a warning (see [`docs/development/code-style.md`](docs/development/code-style.md)); new code should not introduce new `any` usages even though the rule doesn't yet fail CI.
- `strict: true` in `tsconfig.json` is required; don't disable it, repo-wide or per-file.
- Tailwind v4 — theme lives in the `@theme inline` block, not a `tailwind.config.js`.
- Animations via Framer Motion; respect `prefers-reduced-motion`.
- Keep client bundles lean; lazy-load heavy charts where possible.

### Tests

- Tests live in `tests/` and mirror the package structure (`tests/test_distortions.py`, `tests/test_pipeline.py`, etc.).
- Use `monkeypatch` for env-var manipulation; never mutate `os.environ` directly.
- Aim for fast tests (< 30s for the whole suite excluding training-heavy ones). Mark slow tests with `@pytest.mark.slow`.
- Never reduce the test count. If you delete tests, the PR description must explain why.

### Git

- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `perf:`.
- One concern per commit. Squash exploratory commits before pushing.
- Branch names: `feat/<short-slug>`, `fix/<short-slug>`, `docs/<short-slug>`.

---

## 5. Documentation

All PRs that change user-facing behavior **must** update relevant documentation:

- **API changes** → Update `docs/api/` OpenAPI spec and relevant endpoint docs
- **New features** → Add to `README.md` feature table + relevant section
- **Config changes** → Update `configs/default.yaml` comments + `CLAUDE.md` if applicable
- **Distortion changes** → Update the README distortion table + `docs/research/paper-draft.md`
- **Frontend changes** → Update component inventory in README if adding panels
- **Breaking changes** → Add migration note at the top of PR description

Good documentation is as important as good code. If you're unsure what to update, ask in the PR description and we'll guide you.

---

## 6. PR checklist

> **CI runs `ruff check .` on every PR and will block merge if there are lint errors.** Run it locally before pushing to avoid failed checks.

Before requesting review, confirm every box.

- [ ] I have **starred the repo** and **followed [@Adit-Jain-srm](https://github.com/Adit-Jain-srm)**.
- [ ] `pytest --cov=nightmarenet tests/ -v --tb=short` — green locally.
- [ ] `ruff check .` — zero errors.
- [ ] `mypy nightmarenet/ --python-version 3.12 | mypy-baseline filter` — no new errors (see [`docs/development/code-style.md`](docs/development/code-style.md)).
- [ ] If frontend changed: `npx tsc --noEmit` in `frontend/` — no errors.
- [ ] If frontend changed: `cd frontend && npm run build` succeeds.
- [ ] No `from __future__ import annotations` added under `nightmarenet/api/`.
- [ ] No new `nightmarenet/` import of a hosted-only library (`sqlalchemy`, `redis`, `celery`, `psycopg2`, `stripe`).
- [ ] New code is type-annotated; new public APIs have Google-style docstrings.
- [ ] New distortions / metrics / phases are tested for determinism, edge inputs, and registry round-trip.
- [ ] Documentation updated (see [Section 5](#5-documentation)).
- [ ] PR description includes:
  - one-paragraph summary
  - link to the issue / discussion
  - **before / after** behavior (or numbers, when applicable)
  - any breaking change explicitly called out at the top.
- [ ] **Acceptance criteria from the linked issue are copied into the PR description as a checklist.** Every criterion must be checked off before requesting review. If a criterion cannot be met in this PR, explain why in the description.

> [!NOTE]
> **Why acceptance criteria in the PR?** Issues define what "done" looks like. PRs prove it. Copying the acceptance criteria into your PR description creates a verifiable contract: reviewers check the boxes against your code, and incomplete implementations are caught before review begins (not after). If your PR only addresses a subset of the criteria, state that explicitly and link to a follow-up issue for the rest.

CI mirrors the local checks plus a security scan. Merging is blocked on a green CI and one approving review.

---

## 7. ECSoC'26 Contributors

This section applies to contributors participating in the **ECSoC'26** open-source track.

### Bonus XP Labels

These are applied by maintainers at merge time based on quality. **Do not request them.** Focus on correct PR submission and code quality - bonuses follow naturally.

| Label | Bonus | Criteria |
|-------|-------|----------|
| `good-pr` | +15 XP | Clean PR description, all checklist items checked, CI green, no revision rounds |
| `good-issue` | +10 XP | Well-researched issue with file paths, code references, and clear acceptance criteria |
| `good-ui` | +25 XP | Frontend changes with before/after screenshots, responsive design, accessibility |
| `good-backend` | +50 XP | Backend changes demonstrating architectural understanding, proper error handling, tests |

### Rules for ECSoC'26 Participants

1. **Maximum 5 concurrent assignments.** You can be assigned to at most 5 open issues at any time. Finish and deliver before requesting more.

2. **No spam or AI slop.** PRs that are clearly unreviewed AI output (hallucinated APIs, untested code, copy-pasted without understanding) will be closed immediately and may result in disqualification from the program.

3. **Quality over speed.** A PR that needs 3 revision rounds costs more maintainer time than one that merges on first review. Read the codebase, run tests locally, follow conventions.

4. **Approach comment required.** Before requesting assignment, post a comment explaining your planned approach (which files, what changes, estimated timeline). "Assign me" without context will be ignored.

5. **7-day activity window.** If assigned with no PR and no progress update comment within 7 days, you will be unassigned without warning.

6. **One PR per issue.** Don't bundle unrelated fixes. If you find something else while working, open a separate issue for it.

7. **Run CI locally before pushing.** `ruff check .` + `pytest tests/` + `mypy nightmarenet/`. PRs that fail CI on first push suggest you didn't test locally.

8. **Disclose AI usage.** If you used AI tools (Copilot, ChatGPT, Claude, Cursor), state it in the PR description. We welcome AI-assisted contributions. We reject blindly pasted output.

### How to Maximize Your Score

- Pick issues matching your skill level (start with L1 if new to the codebase)
- Read the source code around your change before implementing
- Include tests for new functionality (automatic `good-pr` signal)
- Write PR descriptions that explain WHY, not just WHAT
- Respond to review comments within 24 hours
- Follow up merged PRs with related improvements (builds trust, gets `good-backend`)

---

## 8. Where to ask for help

- **GitHub Discussions** — `https://github.com/Adit-Jain-srm/NightmareNet/discussions`
  - `q-and-a` for "how do I..." questions
  - `ideas` for feature proposals (RFC threads welcome)
  - `research` for paper-related discussion, benchmark proposals, citation requests
- **GitHub Issues** — bug reports and concrete tasks
- **Discord** — `https://discord.gg/nightmarenet` *(launching with Sprint 8)*; channels for `#dev`, `#research`, `#hosted-platform`, `#help`
- **Direct contact** — for security disclosures, email the maintainers per [`SECURITY.md`](SECURITY.md). Do **not** open public issues for vulnerabilities.

We respond fastest to issues that include a minimal reproducible example, the relevant config snippet, and the output of `pip list | findstr nightmarenet` (or `pip freeze | grep nightmarenet` on Unix).

---

Welcome to the project. We are excited to see what you build.
