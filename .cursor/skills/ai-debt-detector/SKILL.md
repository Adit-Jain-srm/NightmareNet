---
name: ai-debt-detector
description: >-
  Use after generating code, after accepting AI suggestions, or when reviewing
  AI-written modules. Also use when code "works" but feels brittle, when error
  handling seems thin, when you notice orphaned resources or missing cleanup,
  or when the agent says "done" but you suspect hidden debt. Catches the specific
  failure patterns AI agents produce that humans wouldn't.
---

## Overview

AI agents generate code that passes the happy path but hides debt: missing error handling, orphaned resources, ignored failure modes, hallucinated packages, silent architectural drift. This skill forces a targeted audit for the exact patterns AI agents get wrong.

## When to Use

- After any AI code generation session (agent wrote 20+ lines)
- Before merging AI-generated PRs
- When code "works" but something feels off
- After vibe-coding sprints — debt accumulates fastest here
- When you find yourself saying "the agent said it's done"

## Process

After ANY code generation, scan for these AI-specific debt patterns:

```
1. FAILURE MODES — What happens when this fails?
   - Network timeout? Disk full? Permission denied? Null input?
   - Is there a try/catch? Does it catch SPECIFIC errors or swallow everything?
   - Are resources cleaned up on failure? (streams closed, connections returned, temp files deleted)

2. ORPHANS — What gets created but never cleaned up?
   - Temp files, event listeners, intervals, subscriptions, connections
   - Are there corresponding cleanup/dispose/close calls for every open/create?
   - In React: does every addEventListener have a removeEventListener in cleanup?

3. EDGE CASES — What inputs break this?
   - Empty array/string? null/undefined? Very large input? Unicode? Concurrent calls?
   - Does the code assume the happy path? (hint: AI almost always does)

4. HALLUCINATED DEPS — Do all imports actually exist?
   - Is every package in package.json/requirements.txt?
   - Are API methods real? (AI invents plausible-sounding methods that don't exist)
   - Check: does this library's latest version still export this function?

5. ARCHITECTURAL DRIFT — Does this match the project's patterns?
   - Same error handling style as existing code?
   - Uses the project's established utilities (not reinventing)?
   - Follows the file structure convention?
   - Matches naming patterns (camelCase vs snake_case, etc.)
```

## Red Flags (stop and fix IMMEDIATELY)

- `catch (e) {}` or `catch (e) { console.log(e) }` — swallowed error
- No `finally` block when resources were opened
- `// TODO: handle error` — AI's way of punting
- Import from a path that doesn't exist in the project
- Timeout set but no abort/cleanup on timeout
- Database connection opened but never released back to pool
- File opened but no close in error path

## Common Mistakes

- Trusting that "it compiled" means "it works" (compilation checks syntax, not logic)
- Reviewing only the diff without checking what the AI DIDN'T generate (missing error paths)
- Assuming the AI used the right library version (it often uses deprecated APIs)
- Skipping the orphan check because "garbage collection handles it" (it doesn't for connections, listeners, timers)

## Rationalizations (countered)

| Excuse | Reality |
|--------|---------|
| "It works in testing" | Testing only covers happy path. Production has failures. |
| "I'll add error handling later" | You won't. And you'll forget which parts need it. |
| "The framework handles cleanup" | Does it? Which framework, which version, which method? Verify. |
| "It's just a prototype" | Prototypes become production. Debt compounds from day one. |
| "The AI generated it, it's probably fine" | AI generates 1.7x more issues than human code (CodeRabbit, 2026). |

## Persistence

ACTIVE after every code generation session. Scan silently. Report only actual findings. Don't generate false positives — only flag patterns that genuinely indicate hidden debt.

## Depth Levels

| Change Size | Audit Depth |
|-------------|-------------|
| < 20 lines | Quick: #1 (failure modes) + #2 (orphans) only |
| 20-100 lines | Standard: all 5 patterns |
| 100+ lines or new module | Deep: all 5 + check integration points + verify deps exist |

## Why This Exists

June 2026: companies spend 44% of AI tokens fixing bugs that AI generated. Developers report 3-day refactoring sessions for 20 minutes of agent-generated code. The agent produces code that "looks polished even when it's wrong." This skill is the countermeasure.

> "It works" is not "it's done." Agent code often passes the happy path and dies on edge cases you haven't thought of yet.
