Create a conventional commit for the current staged changes in NightmareNet.

1. Run `git diff --cached --stat` to see what's staged. If nothing is staged, run `git diff --stat` and suggest what to stage.

2. Analyze the changes and determine the commit type:
   - `feat:` — new feature
   - `fix:` — bug fix
   - `docs:` — documentation only
   - `test:` — adding/updating tests
   - `refactor:` — code change that neither fixes a bug nor adds a feature
   - `chore:` — build process, CI, tooling changes

3. Before committing, verify:
   - `ruff check .` passes clean
   - `pytest tests/ -q` passes (run only if Python files changed)
   - `cd frontend && npm run build` passes (run only if frontend/ files changed)

4. Generate the commit command:
   ```
   git commit -m "<type>: <concise description>"
   ```

   Rules:
   - Subject line ≤72 chars
   - Lowercase after the colon
   - No period at the end
   - Imperative mood ("add" not "added")

5. Show the command and ask for confirmation before executing.
