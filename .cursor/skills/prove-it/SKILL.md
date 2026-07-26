---
name: prove-it
description: >-
  Use when completing any task, fixing bugs, implementing features, or whenever
  about to claim work is done. Also use when the agent says "fixed", "done",
  "implemented", "all set", or "should work now" without showing evidence.
  Auto-triggers before any completion claim.
---

## Overview

The #1 agent failure: claiming "done" without proof. This skill forces evidence-before-assertion on every completion claim.

## Process

Never say "done" without proof. Never say "fixed" without running it. Never say "implemented" without showing it works.

## Persistence

ACTIVE on EVERY completion claim. Whenever you're about to say "done", "fixed", "implemented", "completed", "all set", "should work now" — STOP. Prove it first.

## The Gate

Before ANY statement claiming work is complete:

```
1. WHAT CLAIM am I about to make? (state it explicitly)
2. WHAT EVIDENCE proves this claim? (name the specific command/test/verification)
3. RUN the evidence NOW (not "I could run..." — actually run it)
4. READ the output (full output, not just exit code)
5. DOES the output CONFIRM the claim? 
   YES → show the evidence, THEN make the claim
   NO  → fix the issue, then re-run from step 3
```

## Examples

**BAD:** "Fixed the authentication bug." ← Where's the proof?

**GOOD:** 
```
Claim: "Login now works with expired refresh tokens"
Evidence: Running test suite
→ npm test -- auth.test.ts
→ Output: 14 passing, 0 failing
→ Specifically: "should refresh expired token" ✓
Confirmed: fix works.
```

**BAD:** "Implemented the search feature." ← Did you try it?

**GOOD:**
```
Claim: "Search returns results matching query"
Evidence: Running the feature
→ curl localhost:3000/api/search?q=test
→ Output: {"results": [{"title": "Test Item", ...}], "total": 3}
Confirmed: feature works.
```

## What Counts as Evidence

| Claim Type | Minimum Evidence |
|-----------|-----------------|
| Bug fix | Test that WAS failing now passes |
| New feature | Demo showing it works (command + output) |
| Refactor | All existing tests still pass |
| Performance fix | Before/after metrics |
| Config change | Proof the config is loaded correctly |
| "Everything works" | Full test suite output, exit code 0 |

## What Does NOT Count

- "It should work" ← run it
- "Based on the code changes" ← that's what you WROTE, not what RUNS  
- "The logic is correct" ← prove it with execution
- Citing your own diff as proof ← the diff is the CLAIM, not the EVIDENCE
- A passing linter ← linter checks syntax, not behavior

## Why

The #1 agent failure pattern (2026): "says done but isn't." Users report spending MORE time verifying false completions than if they'd done it themselves. This single discipline — prove before claiming — eliminates the entire category of wasted cycles.

> "Claiming work is complete without verification is dishonesty, not efficiency."

## Red Flags — STOP if you think any of these

- "The change is so small it obviously works"
- "I'll verify in the next step"
- "Based on my understanding of the code..."
- "It should work because the logic is sound"
- "The tests passed earlier, so this is fine"
- "Let me just confirm it's done" (without running anything)

**All of these mean: you're about to claim without evidence. Run the verification FIRST.**

## Common Mistakes

- Treating your code diff as evidence (the diff is the CLAIM, execution is the EVIDENCE)
- Running tests but not reading output (exit code 0 with skipped tests is not passing)
- "It worked earlier" as justification (prove it works NOW)
- Verifying in your head instead of on the machine

## Rationalizations (countered)

| Excuse | Reality |
|--------|---------|
| "It's a one-line change" | One-line changes break production. Run the test. |
| "I already verified mentally" | Mental verification = no verification. Execute it. |
| "The test suite takes too long" | Run the RELEVANT test, not all of them. No excuse. |
| "It's just a config change" | Config errors are the hardest to debug. Prove it loaded. |
| "I'll run tests after I finish everything" | By then you won't know which change broke what. Prove EACH. |
| "The user can verify" | YOU verify. Don't outsource your responsibility. |
