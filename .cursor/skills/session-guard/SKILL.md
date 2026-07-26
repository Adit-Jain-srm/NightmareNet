---
name: session-guard
description: >-
  Prevents long-session corruption AND context compaction amnesia. Monitors
  session health, detects instruction drift, enforces critical-rule recitation,
  and splits proactively before degradation hits. Use when working on complex
  multi-step tasks, when a session is getting long, when the agent starts
  ignoring rules it followed earlier, when CLAUDE.md conventions drift, when
  output quality seems to degrade, or after any context compaction event.
---

# Session Guard

## Overview

Long sessions corrupt silently. Context compaction drops your instructions without warning. This skill prevents both: monitors health, anchors critical rules through compaction, and splits before damage occurs. No packages, no SQLite, no MCP — pure behavioral enforcement that works in Cursor AND Claude Code.

## Process

Monitor session health passively. At checkpoints, RECITE critical rules to self-anchor. Split BEFORE corruption appears — once it occurs, it's invisible and irreversible.

## Persistence

ACTIVE every session. Monitors passively. Escalates on threshold breach. Recites critical rules at checkpoints.

## The Problem (June 2026, confirmed across platforms)

Two distinct failure modes:

**1. Tool Call Degradation (50+ calls):**
- Hallucinated tool results (content that doesn't match files on disk)
- Tool results duplicated (same read echoed 2-8x)
- False progress (agent claims "done" for incomplete work)

**2. Context Compaction Amnesia:**
- CLAUDE.md instructions silently dropped after compaction
- Naming conventions, architectural rules, file structure constraints forgotten
- Agent operates from "degraded mental model" without knowing it
- Compaction summary creates "narrative momentum" that overrides re-injected rules

Both are PLATFORM limitations. Smart session management mitigates both completely.

## Health Signals

| Signal | Threshold | Action |
|--------|-----------|--------|
| Tool call count | >40 | YELLOW: checkpoint + recite critical rules |
| Tool call count | >60 | RED: split or compact with anchor |
| Agent contradicts earlier decision | Any | VERIFY: re-read source of truth |
| Style/naming conventions drift | Any | RECITE: state the active rules aloud |
| File read returns unexpected content | Any | RE-READ: don't trust cached state |
| Repeated failed tool calls | 3+ same error | PAUSE: diagnose before continuing |
| Task scope growing unbounded | Continuous | SPLIT: one task per session |

## Protocol

### Green Zone (0-40 tool calls)
Normal operation. No intervention needed.

### Yellow Zone (40-60 tool calls)
```
1. CHECKPOINT — summarize progress in one paragraph
2. RECITE — state the 3-5 most critical active rules aloud:
   "Active rules: [naming convention], [file structure], [error handling pattern], [testing requirement]"
3. ASSESS — almost done? Push through. Not done? Prepare split.
4. REDUCE — no exploratory reads. Only targeted operations.
```

### Red Zone (60+ tool calls OR drift signal)
```
1. STOP — do not make more tool calls
2. VERIFY — re-read CLAUDE.md / project rules (don't trust memory)
3. CHECKPOINT — write state to handoff document
4. SPLIT — create handoff, suggest fresh session
```

## Context Anchoring (anti-compaction)

When you detect compaction has occurred (sudden loss of earlier context, or after `/compact`):

```
1. RE-READ the project's CLAUDE.md or rules file immediately
2. RECITE the 3-5 critical rules from it aloud in your response
3. VERIFY your planned next action matches those rules before executing
4. If uncertain about ANY prior decision, RE-READ the source file — don't guess
```

### What survives compaction (use this to your advantage):
- Most recent user messages (high priority)
- Currently-invoked skill body (capped at 5K tokens — put critical rules near TOP)
- Git status and project structure
- File contents you read AFTER compaction

### What gets LOST in compaction:
- Decisions made early in conversation
- Architectural rules stated only verbally (not in files)
- Context from tool outputs (file reads, command outputs)
- Skill descriptions you didn't invoke

### Compaction-Safe Pattern
Keep critical instructions in files (CLAUDE.md, CONTEXT.md), NOT in conversation. If a rule matters, it must live in a file the agent can re-read — not in "something we agreed on earlier."

## Session Split Protocol

```markdown
## Handoff — Session Split

**Why split:** [signal that triggered]
**Progress:** [what's done]  
**Current state:** [modified files, test status]
**Critical rules still active:** [list the 3-5 rules that must carry forward]
**Next steps:** [exact actions — numbered, specific]

Start fresh session with: "Continue from handoff: [summary]. Critical rules: [list]"
```

## Common Mistakes

- Trusting that "I remember the rules" after 50+ tool calls (you don't — re-read)
- Re-reading EVERYTHING to be safe (wastes tool calls — be targeted)
- Feeling fine therefore assuming context is fine (compaction is SILENT)
- Splitting AFTER noticing problems (split BEFORE — prevention, not recovery)

## Why This Matters

Context compaction is the #1 unsolved platform problem in 2026. Hooks don't fix it (confirmed: Claude ignores post-compaction injections). External memory tools require heavy setup. This skill is the lightweight behavioral countermeasure: no infra, no packages — just disciplined self-monitoring that works anywhere.

The cost of anchoring: 5 seconds to recite rules at checkpoints.
The cost of NOT anchoring: hours debugging code written against forgotten constraints.
