Implement a feature using strict Test-Driven Development for NightmareNet.

**Feature**: $ARGUMENTS

Follow this exact sequence:

1. **Understand**: Read relevant existing code and tests to understand the current implementation.

2. **Red** — Write the failing test(s) FIRST:
   - Add tests in the appropriate `tests/test_*.py` file
   - Tests must be specific, testing the exact behavior described
   - Run `pytest <test_file> -v` — confirm the new tests FAIL (red)

3. **Green** — Write minimal implementation:
   - Only enough code to make the failing tests pass
   - No extra features, no premature optimization
   - Run `pytest <test_file> -v` — confirm all tests PASS (green)

4. **Refactor** — Clean up if needed:
   - Run `ruff check .` and fix any lint errors
   - Ensure no regressions: run `pytest tests/ -v --tb=short`

5. **Report**: State the exact tests added and the implementation changes made.

**Rules**:
- NEVER write production code before a failing test
- If you catch yourself writing code first, STOP and write the test
- Each test should test ONE behavior
- Use descriptive test names: `test_<what>_<condition>_<expected>`
