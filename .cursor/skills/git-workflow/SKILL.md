---
name: git-workflow
description: >-
  Enforces disciplined git practices on every commit. Small atomic commits,
  conventional messages, branch hygiene, and PR readiness checks. Changes how
  the agent HANDLES git — not just what commands it knows. Auto-triggers on any
  git operation. Use when committing, branching, merging, or preparing PRs.
  Also use when git history is messy, commits are too large, or PRs get rejected
  for poor git hygiene.
---

## Overview

Enforce disciplined git practices on every operation — small atomic commits, conventional messages, clean branch hygiene, and PR readiness checks. Changes how the agent handles git, not just what commands it knows.

# Git Workflow

Change how you handle git. Not a reference — a DISCIPLINE.

## Persistence

ACTIVE on every git operation. Every commit, every branch, every PR. Not optional. Off only when user explicitly says "skip git discipline".

## The Discipline

### Every Commit Must Be

```
SMALL: One logical change (if you need "and" to describe it, split it)
ATOMIC: Passes tests on its own (any commit can be reverted safely)
MESSAGED: Conventional format (type: concise description)
CLEAN: No debug logs, no commented code, no TODOs you won't do
```

Before EVERY `git commit`:
1. `git diff --staged` — READ what you're actually committing
2. Is this ONE thing? If not, `git reset HEAD` and commit in parts
3. Write message: `type(scope): what changed` (not "update files")
4. Check: would reverting THIS commit break anything else? (shouldn't)

### Branching

```
RULE: Never commit directly to main/master
RULE: Branch names describe the work: feat/add-auth, fix/login-timeout
RULE: Branches live < 3 days (if longer, you're batching too much)
RULE: Rebase before PR (clean linear history)
```

### Before Creating a PR

Run this checklist YOURSELF before showing the PR to the user:

```
[ ] All commits are atomic and conventional
[ ] No fixup commits left (squash them)
[ ] Branch is rebased on latest main
[ ] Tests pass on the branch
[ ] No unrelated changes snuck in
[ ] PR description explains WHY not just WHAT
[ ] Diff is < 400 lines (if larger, split into multiple PRs)
```

### Conventional Commit Types

| Type | When |
|------|------|
| `feat` | New capability for users |
| `fix` | Bug that affected users |
| `refactor` | Code change, no behavior change |
| `test` | Adding/fixing tests only |
| `docs` | Documentation only |
| `chore` | Build, deps, config |
| `perf` | Performance improvement |

### Bad → Good

```
BAD:  "update stuff"
GOOD: "feat(auth): add JWT refresh token rotation"

BAD:  "fix bug"  
GOOD: "fix(checkout): prevent double-charge on timeout retry"

BAD:  one commit with 15 files changed across 3 features
GOOD: 3 commits, each touching one feature, each revertable alone
```

## Auto-Triggers

When you're about to:
- `git add .` → STOP. Stage selectively. Read the diff.
- `git commit -m "..."` → CHECK: is the message conventional? Is the change atomic?
- `git push` → CHECK: is the branch rebased? Tests pass?
- Create a PR → RUN the PR checklist above.

## Why

Messy git = impossible to debug, revert, or review. Disciplined git = every commit is a rollback point, every PR is reviewable, every message is searchable. This compounds over months into a codebase that's navigable by archaeology.
