Debug a failing test or runtime error in NightmareNet using systematic debugging.

**Problem**: $ARGUMENTS

Follow this structured process — do NOT jump to a fix:

## Phase 1: Investigate Root Cause
1. **Reproduce**: Run the failing test/command and capture the exact error output
2. **Trace**: Read the stack trace bottom-up. Identify the exact line and function where the failure occurs
3. **Context**: Read the surrounding code (±30 lines) to understand the logic
4. **Inputs**: What data/state leads to this code path? Check test fixtures, mock data, config

## Phase 2: Form Hypothesis
- Based on the evidence, state your hypothesis for the root cause
- Identify what distinguishes "it works" from "it fails" conditions
- If the error is a type error, check Python 3.9 compatibility constraints

## Phase 3: Verify & Fix
1. Write or modify a test that specifically reproduces the bug (if one doesn't exist)
2. Apply the minimal fix
3. Run the specific test — confirm it passes
4. Run `pytest tests/ -q` — confirm no regressions
5. Run `ruff check .` — confirm lint clean

## Phase 4: Report
- **Root cause**: What actually caused the bug (1-2 sentences)
- **Fix applied**: What changed (file + line)
- **Tests**: Which tests verify the fix
- **Regressions**: Full test suite status

**Rules**:
- NEVER apply a fix without understanding the root cause first
- If 3 fix attempts fail, stop and question your assumptions about the architecture
- Check for related issues in similar code patterns
