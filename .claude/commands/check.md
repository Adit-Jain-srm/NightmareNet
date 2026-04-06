Run the full NightmareNet quality check pipeline. Execute each step in order and report results:

1. **Lint**: Run `ruff check .` from the project root. Report the number of errors. If any exist, list the first 10.

2. **Tests**: Run `pytest tests/ -v --tb=short -q` from the project root. Report total passed/failed/skipped.

3. **Frontend Build**: Run `cd frontend && npm run build` and verify it compiles without errors.

4. **Summary**: Provide a concise pass/fail summary:
   - ✅ or ❌ Ruff lint (X errors)
   - ✅ or ❌ Tests (X passed, Y failed)
   - ✅ or ❌ Frontend build

If any step fails, diagnose the root cause and suggest fixes. Do NOT auto-fix — report findings only.
