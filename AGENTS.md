# NightmareNet Agent Configuration

## Instructions

Read and follow `CLAUDE.md` in this repository for project-specific conventions.

## Skills

The following skills are available and SHOULD be used when relevant:

### Superpowers (via ~/.agents/skills/superpowers/)
- **test-driven-development** — Use when writing any new feature or fixing bugs
- **systematic-debugging** — Use when investigating test failures or runtime errors
- **verification-before-completion** — Use before claiming ANY task is done
- **writing-plans** — Use when the task requires 3+ steps of implementation
- **executing-plans** — Use when executing a previously written plan

### Project Skills (via .claude/skills/)
- **ui-ux-pro-max** — Use for any frontend/UI work on the Next.js frontend

## Commands

Available slash commands in `.claude/commands/`:
- `/check` — Run full quality pipeline (lint + tests + frontend build)
- `/tdd <feature>` — Implement a feature with strict TDD
- `/commit` — Create a conventional commit with pre-flight checks
- `/debug <problem>` — Systematic debugging workflow
- `/prime` — Load project context for a new session

## Hooks

Configured in `.claude/settings.json`:
- **PostToolUse**: Python syntax check on `.py` file edits
- **Stop**: Ruff lint check before session ends
