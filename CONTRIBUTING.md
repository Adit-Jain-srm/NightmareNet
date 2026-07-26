# Contributing to NightmareNet

Thank you for helping improve NightmareNet. This project uses a **research-first, verification-driven** workflow: every change is justified, tested, and traceable. The sections below take you from a fresh clone to a merged PR.

---

## Before You Start

> **Please complete these steps before opening a Pull Request:**

1. **Star this repository** — It helps us gauge community interest and prioritize features.
2. **Follow [@Adit-Jain-srm](https://github.com/Adit-Jain-srm)** — Stay updated on releases, related projects, and research.
3. **Read this entire guide** — PRs that don't follow the coding standards or skip tests will be asked to revise.
4.  **Please read our [Code of Conduct](CODE_OF_CONDUCT.md)** before contributing to help maintain a welcoming and respectful community.
> Maintainers verify star/follow status before merging. PRs from accounts that haven't completed steps 1-2 will be asked to do so before review begins.

---

## Table of Contents

1. [Before you start](#before-you-start)
2. [Opening issues](#opening-issues)
3. [Issue assignment rules](#issue-assignment-rules)
4. [Code philosophy](#code-philosophy)
5. [Local development setup](#local-development-setup)
6. [Architecture pointers](#architecture-pointers)
7. [Adding a new distortion](#adding-a-new-distortion)
8. [Coding standards](#coding-standards)
9. [Documentation](#documentation)
10. [API Versioning Policy](#api-versioning-policy)
11. [PR checklist](#pr-checklist)
12. [ECSoC'26 Contributors](#ecsoc26-contributors)
13. [Where to ask for help](#where-to-ask-for-help)

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
   ```markdown
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

> [!TIP]
> **Think you can do it better?** Don't be discouraged by an existing assignment. If you believe you can deliver a better or faster implementation, comment with your detailed approach - even if someone else already requested the issue. We evaluate approaches on merit, not arrival order. Include: which files you'll change, what enhancements you'd add beyond the stated requirements, and your timeline. If your plan is demonstrably stronger (more complete, better tested, or addresses edge cases the current assignee missed), we'll reassign. The goal is the best possible contribution, not a queue.

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

## Local development setup

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

### Pre-commit hooks (optional)

There is no `.pre-commit-config.yaml` committed yet. If you want pre-commit locally:

```bash
pip install pre-commit
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
EOF
pre-commit install
```

Or simply run `make lint` before committing.

### Verify the environment

The Makefile mirrors exactly what CI runs — use it instead of memorizing
individual commands:

```bash
make check          # lint + typecheck + test (what CI runs on every PR)
make test            # just pytest with coverage
make lint            # just ruff
make typecheck       # just mypy
make format          # auto-fix formatting with ruff format
```

If you also touched the dashboard:

```bash
make frontend-build  # production build (cd frontend && npm ci && npm run build)
make frontend-test   # frontend test suite
```

Or run everything, Python + frontend:

```bash
make all
```

### Start the API for ad-hoc testing

```bash
uvicorn nightmarenet.api.app:app --reload --port 8000 --env-file .env
```

Hit `http://127.0.0.1:8000/api/v1/health` to confirm.

---

## Architecture pointers

NightmareNet has a strict OSS / hosted boundary. Treat it as a hard constraint when adding code.

| Package | Purpose | Allowed dependencies |
|---------|---------|---------------------|
| `nightmarenet/` | OSS core: distortions, training loop, evaluation, CLI, FastAPI inference endpoints | `torch`, `transformers`, `pydantic`, `fastapi`, `pyyaml`, `slowapi` (optional) |
| `nightmarenet_server/` | Hosted platform: auth, multi-tenant DB, Celery workers, billing | OSS core + `sqlalchemy`, `redis`, `celery`, `stripe`, `psycopg2` |
| `frontend/` | Next.js 16 dashboard, design system, charts | npm ecosystem only; talks to OSS API or hosted API via `NEXT_PUBLIC_API_URL` / rewrites |

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

## Adding a new distortion

Distortions are first-class plugins. The full walkthrough is in [`docs/plugin_development.md`](docs/plugin_development.md) and [`notebooks/03_custom_distortions.ipynb`](notebooks/03_custom_distortions.ipynb); the short version follows.

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

## Coding standards

### Python

- **Line length:** 100 (enforced by ruff).
- **Ruff rules:** `E, F, W, I, N, UP, B`. We ignore `UP007` and `UP045` to keep `Union[X, Y]` available in 3.9-targeted code.
- **Imports:** isort via ruff. Order: stdlib, third-party, local; alphabetical within each group.
- **Type hints & Mypy:**
  - NightmareNet uses a **gradual typing** approach.
  - **Strict Modules:** Specific core modules (including `nightmarenet/phases/`, `nightmarenet/compliance/`, `nightmarenet/hub/`, and `nightmarenet/evaluation/certification.py`) are strictly typed and must pass mypy without errors.
  - **Legacy Modules:** Older components like `nightmarenet/cli.py` and `nightmarenet/api/app.py` remain in relaxed mode.
  - Use `Union[X, Y]` and `Optional[X]` — **not** `X | Y` — in any code path that runs on Python 3.9.
  - Use `from __future__ import annotations` everywhere **except** modules under `nightmarenet/api/` that use FastAPI `Body(...)`. The future import breaks Pydantic v2 at runtime there. Prefer module-level singletons for `Body(...)` defaults to satisfy `B008`.
- **Docstrings:** Google style on public APIs only. Internal helpers can be terse.
- **Errors:** raise with context (`raise X("...") from e`); never bare `raise X`.
- **No NaN/Inf in metrics:** wrap suspicious arithmetic with the helpers in `nightmarenet/evaluation/metrics.py`.
- **Logging:** use module loggers (`logger = logging.getLogger(__name__)`); don't `print` in library code.

### Frontend

- TypeScript only. No `any` in committed code.
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
- **All PRs target `main`** unless the issue specifies otherwise.
- **Do not force-push** to PR branches. Push review fixes as new commits so reviewers can see incremental changes. Maintainers will squash on merge.
- Signed commits are not required but are appreciated.

---

## Documentation

All PRs that change user-facing behavior **must** update relevant documentation:

- **API changes** → Run server and check [auto-generated docs](http://localhost:8000/docs); update relevant endpoint descriptions in code
- **New features** → Add to `README.md` feature table + relevant section
- **Config changes** → Update `configs/default.yaml` comments + `CLAUDE.md` if applicable
- **Distortion changes** → Update the README distortion table + `docs/research/paper-draft.md`
- **Frontend changes** → Update component inventory in README if adding panels
- **Breaking changes** → Add migration note at the top of PR description

Good documentation is as important as good code. If you're unsure what to update, ask in the PR description and we'll guide you.

---

## API Versioning Policy

NightmareNet follows [Semantic Versioning 2.0.0](https://semver.org/) for its API. The current API version is attached to every response via the `X-API-Version` header.

### When to Update the API Changelog

Contributors **must** update [docs/api-changelog.md](docs/api-changelog.md) whenever a Pull Request introduces any changes to the API endpoints, schemas, authentication, or query parameters.

### Classification of Changes

Changes are classified into three main types:

1. **Breaking Changes (Major)**
   - Renaming or deleting existing endpoints.
   - Removing fields from request bodies or response objects.
   - Changing the data type of existing fields.
   - Making previously optional fields mandatory.
   - Tightening rate limits significantly.
   - Action required: Increment major version (e.g., `0.2.0` -> `1.0.0`), document migration steps, and mark Change Type as `Breaking` in the changelog.

2. **Non-breaking Changes (Minor / Patch)**
   - Adding new endpoints.
   - Adding optional fields to request bodies or new fields to response objects.
   - Non-breaking bug fixes or performance improvements.
   - Action required: Increment minor/patch version (e.g., `0.2.0` -> `0.2.1`), and mark Change Type as `Added` or `Fixed` in the changelog.

3. **Deprecated / Removed Changes**
   - Marking an endpoint as deprecated in the OpenAPI docs/schemas before its removal.
   - Removing deprecated endpoints (Breaking).

### Required Contributor Changelog Template

When submitting a PR that affects the API, contributors must append a new entry to `docs/api-changelog.md` using the template below:

```markdown
## [<API Version>] - <Release Date (YYYY-MM-DD)>

### Change Type
- <Breaking / Non-breaking / Deprecated / Added / Fixed / Removed>

### Endpoint(s) Affected
- `<HTTP Method> <Endpoint Path>` (e.g., `POST /api/v1/generate/dream`)

### Description
<Detailed description of what changed, why, and the technical impact.>

### Migration Guide
<Step-by-step instructions for clients to upgrade if the change is breaking. Write "None" if non-breaking.>

### Notes
<Any extra context, performance considerations, or caveats.>
```

### Example Entry

```markdown
## [0.2.0] - 2026-07-01

### Change Type
- Added

### Endpoint(s) Affected
- `GET /api/v1/health`
- `POST /api/v1/generate/dream`
- `POST /api/v1/generate/nightmare`
- `POST /api/v1/evaluate/robustness`

### Description
Initial release of the NightmareNet FastAPI service. Supports generation of dream/nightmare text distortions and evaluation of model robustness.

### Migration Guide
None.

### Notes
Initial release.
```

---

## PR checklist

> **Assignment is mandatory.** Do NOT open a PR for an issue you are not assigned to. Request assignment first (see [Issue Assignment Rules](#issue-assignment-rules)). Unassigned PRs will be closed without review.

> **CI runs `make check` on every PR and will block merge if it fails.** Run it locally before pushing to avoid failed checks.

Before requesting review, confirm every box.

- [ ] I am **assigned** to the linked issue.
- [ ] I have **starred the repo** and **followed [@Adit-Jain-srm](https://github.com/Adit-Jain-srm)**.
- [ ] `make check` — green locally (runs lint + typecheck + test).
- [ ] If frontend changed: `make frontend-build` succeeds.
- [ ] No `from __future__ import annotations` added under `nightmarenet/api/`.
- [ ] No new `nightmarenet/` import of a hosted-only library (`sqlalchemy`, `redis`, `celery`, `psycopg2`, `stripe`).
- [ ] New code is type-annotated; new public APIs have Google-style docstrings.
- [ ] New distortions / metrics / phases are tested for determinism, edge inputs, and registry round-trip.
- [ ] Documentation updated (see [Documentation](#documentation)).
- [ ] PR description includes:
  - one-paragraph summary
  - link to the issue / discussion
  - **before / after** behavior (or numbers, when applicable)
  - any breaking change explicitly called out at the top.
- [ ] **Acceptance criteria from the linked issue are copied into the PR description as a checklist.** Every criterion must be checked off before requesting review. If a criterion cannot be met in this PR, explain why in the description.

> [!NOTE]
> **Why acceptance criteria in the PR?** Issues define what "done" looks like. PRs prove it. Copying the acceptance criteria into your PR description creates a verifiable contract: reviewers check the boxes against your code, and incomplete implementations are caught before review begins (not after). If your PR only addresses a subset of the criteria, state that explicitly and link to a follow-up issue for the rest.

CI mirrors the local checks plus a security scan. Merging is blocked on a green CI and one approving review.

### When to request review from maintainer

Only request review (`@Adit-Jain-srm`) when ALL of the following are true:

1. **CI is green** - All Python lint, type-check, and test jobs pass.
2. **CodeRabbit comments resolved** - Every automated suggestion is either fixed or has a reply explaining why you disagree.
3. **PR template checklists complete** - All boxes checked in Pre-submission, Quality, and Documentation sections.
4. **Acceptance criteria met** - Every checkbox from the linked issue is addressed in the PR.

Do NOT request review with failing CI, unresolved bot comments, or unchecked boxes. Premature review requests waste maintainer time and will be dismissed without reading the code.

### CI failed on files you didn't touch?

PR checks run against a **merge of your branch with current `main`** (GitHub's
`refs/pull/N/merge`), not your branch alone. If `main` itself is broken, every
open PR turns red — including yours — on code you never changed.

Before debugging, triage in this order:

1. **Check the failing test's file path.** Is it inside your diff
   (`git diff origin/main --stat`)? If not, it's likely inherited from `main`.
2. **Check for an open [`[CI]: main branch is failing`](../../issues?q=is%3Aissue+is%3Aopen+label%3Aci-status) issue.**
   If one exists, `main` is known-red. Wait for it to close — do NOT try to fix
   `main`'s breakage inside your PR.
3. **Check `main`'s latest [CI runs](../../actions/workflows/ci.yml?query=branch%3Amain).**
   If the newest run on `main` is red with the same failure, same conclusion.
4. **Once `main` is green again**, refresh your PR: merge `main` into your
   branch (or rebase) and push, or click **Update branch** on the PR page.
   Note: the **Re-run jobs** button re-tests the *same* merge snapshot
   (GitHub reuses the original `GITHUB_SHA`), so it will NOT pick up the
   fixed `main` — you must update the branch to get a fresh merge commit.

If none of the above applies and the failure persists on files outside your
diff, comment on your PR with the failing test name and your local
verification (as much of `make check` as applies) — a maintainer will triage.

---

## ECSoC'26 Contributors

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

1. **Maximum 3 concurrent assignments without an open PR.** Once you have an open PR on an issue, it no longer counts toward the limit. Finish and deliver before requesting more.

2. **No spam or AI slop.** PRs that are clearly unreviewed AI output (hallucinated APIs, untested code, copy-pasted without understanding) will be closed immediately and may result in disqualification from the program.

3. **Quality over speed.** A PR that needs 3 revision rounds costs more maintainer time than one that merges on first review. Read the codebase, run tests locally, follow conventions.

4. **Approach comment required.** Before requesting assignment, post a comment explaining your planned approach (which files, what changes, estimated timeline). "Assign me" without context will be ignored.

5. **7-day activity window.** If assigned with no PR and no progress update comment within 7 days, you will be unassigned without warning.

6. **One PR per issue.** Don't bundle unrelated fixes. If you find something else while working, open a separate issue for it.

7. **Run CI locally before pushing.** `make check` (lint + typecheck + test). PRs that fail CI on first push suggest you didn't test locally.

8. **Disclose AI usage.** If you used AI tools (Copilot, ChatGPT, Claude, Cursor), state it in the PR description. We welcome AI-assisted contributions. We reject blindly pasted output.

9. **Compete on quality, not timing.** See an assigned issue where you have a better approach? Comment with your detailed plan - which files, what enhancements, your timeline. We reassign based on the strongest approach, not who commented first. Don't let an existing assignment stop you from proposing a superior implementation.

### How to Maximize Your Score

- Pick issues matching your skill level (start with L1 if new to the codebase)
- Read the source code around your change before implementing
- Include tests for new functionality (automatic `good-pr` signal)
- Write PR descriptions that explain WHY, not just WHAT
- Respond to review comments within 24 hours
- Follow up merged PRs with related improvements (builds trust, gets `good-backend`)

### Pro Tips (what separates great contributors from average ones)

1. **Resolve CodeRabbitAI suggestions.** Our repo uses automated code review. When CodeRabbit leaves suggestions on your PR, address each one (fix it or explain why you disagree). **Do NOT click "Resolve conversation" without actually fixing the code or replying with a reason.** Maintainers verify every resolved comment against the diff - silently resolving without a fix will be caught and delays your merge. If a suggestion conflicts with the intended design, reply in that thread explaining *why* rather than dismissing it.

2. **Re-request review after addressing feedback.** After making changes requested by the code owner, click "Re-request review" on GitHub. Don't just push commits silently and wait - signal that you're ready for the next round.

3. **Complete ALL checklists before requesting review.** The PR template has Acceptance Criteria, Pre-submission Checklist, and Quality Checklist. Every box must be checked (or have an explicit explanation for why not). Reviewers will reject PRs with unchecked boxes without reading the code.

4. **Ask questions when things are ambiguous.** If the issue description is unclear, a design choice has multiple valid approaches, or you're unsure about scope - comment on the issue and ask. This shows ownership and critical thinking. Contributors who ask smart questions earn bonus labels faster than those who guess wrong and waste review cycles.

5. **Reference specific code in your approach.** When requesting assignment or discussing implementation, cite file paths and line numbers. "I'll modify `nightmarenet/training/phases.py:429-528` to add the distillation loss" is 10x better than "I'll implement the feature."

6. **Maintain a local learnings file.** If you plan to work on multiple issues, create a `LEARNINGS.md` in your local setup (gitignored) to track patterns, gotchas, and conventions you discover. This accelerates your second and third PRs dramatically - you won't repeat mistakes or re-read the same code twice.

---

## Where to ask for help

- **GitHub Discussions** — `https://github.com/Adit-Jain-srm/NightmareNet/discussions`
  - `q-and-a` for "how do I..." questions
  - `ideas` for feature proposals (RFC threads welcome)
  - `research` for paper-related discussion, benchmark proposals, citation requests
- **GitHub Issues** — bug reports and concrete tasks
- **Direct contact** — for security disclosures, email the maintainers per [`SECURITY.md`](SECURITY.md). Do **not** open public issues for vulnerabilities.

We respond fastest to issues that include a minimal reproducible example, the relevant config snippet, and the output of `pip list | findstr nightmarenet` (or `pip freeze | grep nightmarenet` on Unix).

---

Welcome to the project. We are excited to see what you build.
