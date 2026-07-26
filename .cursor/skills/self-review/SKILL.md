---
name: self-review
description: >-
  After every code change, automatically review your own work before presenting it.
  Catches bugs, style issues, and missed requirements BEFORE the user sees them.
  Use when implementing features, fixing bugs, refactoring, or any code modification.
  Auto-triggers on code changes. Reduces back-and-forth by catching issues first-pass.
---

## Overview

Catch your own bugs, style issues, and missed requirements BEFORE presenting code. The user should never see your first draft.

## Process

After writing or modifying code, review it yourself BEFORE presenting to the user.

## Persistence

ACTIVE EVERY CODE CHANGE. Not a one-time thing. Every edit, every file, every commit-worthy change gets self-reviewed. Off only when user says "skip review" or "just do it".

## The Review (run mentally, fix inline)

For each file you just modified:

```
1. RE-READ what you wrote (don't trust your memory of what you intended)
2. CHECK: Does this actually solve what was asked? (not a related thing, THE thing)
3. BUGS: Off-by-one? Null/undefined? Race condition? Unclosed resource?
4. EDGE CASES: Empty input? Very large input? Unicode? Concurrent access?
5. CONSISTENCY: Matches existing code style? Same naming conventions? Same patterns?
6. IMPORTS: Everything imported? Nothing unused? Circular dependency?
7. TYPES: Type-safe? No `any` where there shouldn't be? Generics correct?
8. ERROR HANDLING: What can throw? Is it caught? Is the error message useful?
9. TESTS: If tests exist, did I break any? Should I add one for this change?
10. SECURITY: User input validated? SQL injection? XSS? Auth checks?
```

## Fix Before Showing

If you find an issue during self-review: **fix it silently**. Don't tell the user "I found a bug in my code." Just fix it. Present clean work.

The user should never see your first draft. They should see your REVIEWED draft.

## What NOT to do

- Don't announce "I'm now reviewing my code" (just do it, silently)
- Don't present code then say "wait, let me fix something" (review BEFORE presenting)
- Don't skip review because "it's a small change" (small changes have bugs too)
- Don't review only the lines you changed (check surrounding context)

## Depth Levels

| Change Size | Review Depth |
|-------------|-------------|
| 1-5 lines | Quick scan: bugs, types, imports |
| 5-50 lines | Full checklist above |
| 50+ lines | Full checklist + architecture fit + test impact |

## Common Mistakes

- Announcing "I'm reviewing my code now" (just do it silently, present clean results)
- Reviewing only your changed lines without checking surrounding context
- Treating small changes as exempt ("one-line changes can't break things" — they can)
- Using `any` types or skipping error handling because "it's just a quick fix"

## Example (self-review catching a bug)

```
// What you wrote:
const user = await db.users.findOne({ email });
return user.name;

// Self-review catches:
// - What if user is null? (findOne returns null for no match)
// - No error handling on db call

// What you present (after silent fix):
const user = await db.users.findOne({ email });
if (!user) throw new NotFoundError(`No user with email: ${email}`);
return user.name;
```

The user never sees the first version. They get the reviewed version.

## Why This Matters

Without self-review: agent presents code → user finds bug → agent fixes → user finds another → cycle wastes time.

With self-review: agent catches own bugs → presents clean code → user accepts first time → trust builds.

The difference between a junior dev (shows first draft) and a senior dev (shows reviewed work).
