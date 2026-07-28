---
title: Added `nbmake>=1.5.0` to `project.optional-dependencies
slug: added-nbmake-1-5-0-to-project-optional-dependencies
tags: 
scope: project
updated_at: 2026-07-28T09:01:21.870Z
source: live
hook: Added `nbmake>=1.5.0` to `project.optional-dependencies.dev` in `pyproject.toml`
---

- Added `nbmake>=1.5.0` to `project.optional-dependencies.dev` in `pyproject.toml`
- Added CI step `Validate notebooks with nbmake` to `.github/workflows/ci.yml`
- Step runs on Python 3.12 with 120s timeout per notebook
- Initially configured with `continue-on-error: true` to avoid blocking PRs
- Step uses existing `dev` dependency installation via `pip install -e ".[dev,api]"`
• `pyproject.toml` updated with `nbmake>=1.5.0` in `dev` optional dependencies
• `.github/workflows/ci.yml` added notebook validation step for Python 3.12 with `continue-on-error: true` and 120s timeout
• Environment issue identified: Windows Python installation is Microsoft Store stub, not functional Python
• Changes will be validated in GitHub Actions CI with Ubuntu runner and real Python installation
• Commit created: b171f53 "ci: add notebook validation step with nbmake"
• Changes committed:
- Added `nbmake>=1.5.0` to `dev` optional dependencies in `pyproject.toml`
- Added new CI workflow step "Validate notebooks with nbmake" in `.github/workflows/ci.yml`
• Push to upstream repository denied due to lack of write access
• PR publication requires forking the repository or requesting collaborator access
