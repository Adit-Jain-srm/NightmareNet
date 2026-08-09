## Summary

<!-- One-sentence description of what this PR does. Be specific. -->

## Motivation

<!-- Why is this change needed? What problem does it solve? -->

Closes #

## Changes

<!-- Bullet-point list of what changed. Include file paths for non-obvious changes. -->

-

## Acceptance Criteria

<!-- Copy the acceptance criteria from the linked issue as checkboxes.
     Check each box ONLY when your implementation satisfies it.
     If a criterion is not addressed in this PR, leave unchecked and explain why. -->

- [ ]

## Type

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] Feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would break existing behavior)
- [ ] Refactor (no functional change)
- [ ] Documentation
- [ ] Tests
- [ ] Infrastructure (CI/CD, Docker, tooling)

---

## Pre-submission Requirements

> These are mandatory. PRs missing any item will not be reviewed.

- [ ] I have **starred** this repository ⭐
- [ ] I have **followed** [@Adit-Jain-srm](https://github.com/Adit-Jain-srm)
- [ ] I have read [CONTRIBUTING.md](https://github.com/Adit-Jain-srm/NightmareNet/blob/main/CONTRIBUTING.md) in full

## Quality Checklist

- [ ] `ruff check nightmarenet/ scripts/ tests/` — 0 errors
- [ ] `mypy nightmarenet/ --ignore-missing-imports` — no new errors (run on Python 3.12)
- [ ] `pytest tests/` — all tests pass
- [ ] New functionality has corresponding tests
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`)
- [ ] No `from __future__ import annotations` added under `nightmarenet/api/`
- [ ] No new imports of hosted-only libraries (`sqlalchemy`, `redis`, `celery`) in `nightmarenet/`

## Documentation

<!-- Check all that apply. At least one must be checked for non-test PRs. -->

- [ ] No documentation needed (test-only or internal refactor)
- [ ] README.md updated
- [ ] Docstrings added/updated (Google style, public APIs only)
- [ ] `docs/` updated (architecture, research, or API docs)
- [ ] Config comments updated (`configs/default.yaml`)
- [ ] CHANGELOG entry added (if user-facing change)

## AI Disclosure

<!-- Required by CONTRIBUTING.md. Check one. -->

- [ ] This PR is entirely human-written
- [ ] This PR includes AI-assisted code (Copilot, ChatGPT, Claude, etc.) — I have reviewed every line and can explain it

## Security Considerations

<!-- For any PR touching auth, API endpoints, user input handling, or webhooks -->

- [ ] Not applicable (no security-sensitive changes)
- [ ] User input is validated before use
- [ ] No secrets, keys, or credentials in code or comments
- [ ] CORS / auth / rate limiting behavior unchanged (or explicitly documented)
- [ ] If this is a security fix, see [SECURITY.md](https://github.com/Adit-Jain-srm/NightmareNet/blob/main/SECURITY.md) — do NOT describe the vulnerability publicly until patched

## Screenshots (if UI/UX change)

<!-- Required by CONTRIBUTING.md for any frontend change. All three are needed. -->

| View | Screenshot |
|------|-----------|
| Desktop (dark mode) | |
| Desktop (light mode) | |
| Mobile (375px) | |

## Breaking Changes

<!-- If this is a breaking change, describe: (1) what breaks, (2) migration path -->

N/A

---

<details>
<summary>Reviewer notes (optional)</summary>

<!-- Anything you want to call the reviewer's attention to:
     - Areas you're uncertain about
     - Alternative approaches you considered
     - Performance implications
     - Known limitations -->

</details>
