# Code Style & Type Checking Policy

This document covers NightmareNet's type-checking policy for Python and
TypeScript. See [`CONTRIBUTING.md`](../../CONTRIBUTING.md#coding-standards)
for the broader coding standards (line length, import order, docstrings,
etc.).

## Python: mypy, strict mode, and the baseline

`nightmarenet/` is checked with `mypy --strict` (configured in
`pyproject.toml`'s `[tool.mypy]` section). Strict mode means, among other
things: no implicit `Any`, all `def`s must be annotated, no untyped
decorators, no unused `# type: ignore` comments.

The codebase was **not** written under strict mode from day one, so a large
number of pre-existing files don't pass it yet. Rather than blocking every
PR on a full-repo cleanup, we use
[`mypy-baseline`](https://pypi.org/project/mypy-baseline/) to freeze the
current violations:

- **`mypy_baseline.txt`** is a snapshot of every strict-mode error that
  existed in the codebase at the time the baseline was generated. These are
  *known, tracked debt* — they don't fail CI.
- **Any error not in the baseline is new**, whether it's in a brand-new file
  or a new line in a file that already had baselined errors. New errors
  **fail CI**.

In practice this means:

- **New Python files must pass `mypy --strict` outright.** There's nothing
  to grandfather them into, so every error in a new file is a new error.
- **Editing an existing file is fine** as long as you don't introduce
  additional violations. You are not required to fix the file's
  pre-existing errors in the same PR (though it's welcome — see "Paying
  down the baseline" below).

### Running the check locally

```bash
mypy nightmarenet/ --python-version 3.12 | mypy-baseline filter
```

Exit code `0` means no new violations. A non-zero exit means the output
above the summary lists your new violations (pre-existing ones are
suppressed from the detailed listing).

### Paying down the baseline

If you want to fix some pre-existing errors as part of your PR (encouraged,
but never required unless the issue asks for it), regenerate the baseline
afterwards so the fixed errors are removed from the tracked debt:

```bash
mypy nightmarenet/ --python-version 3.12 | mypy-baseline sync --baseline-path mypy_baseline.txt --sort-baseline
```

Commit the updated `mypy_baseline.txt` alongside your fix. `mypy-baseline
sync` will refuse to shrink the baseline silently if it detects unrelated
new errors slipped in — resolve those first.

Do **not** hand-edit `mypy_baseline.txt`, and don't add new
`# type: ignore` comments purely to dodge the baseline check — the
baseline mechanism already gives existing code room to breathe; new code is
expected to be clean.

## TypeScript: strict mode and `no-explicit-any`

- `frontend/tsconfig.json` has `"strict": true`. This is required and
  should not be turned off, repo-wide or per-file, without discussion in
  the PR description.
- `@typescript-eslint/no-explicit-any` is active as a **warning**
  (`frontend/eslint.config.mjs`). It's inherited as an `error` from
  `eslint-config-next/typescript`, but a handful of pre-existing `any`
  usages meant enabling it at `error` level repo-wide would have blocked
  unrelated PRs, so it's explicitly downgraded for now.
  - **New TypeScript code should not introduce `any`.** Treat the warning
    as a hard requirement for anything you write; it's a warning at the
    tooling level only to avoid punishing PRs that don't touch the
    offending files.
  - Once the existing `any` usages are cleaned up, this rule should be
    flipped back to `"error"` in `eslint.config.mjs`.

### Running the checks locally

```bash
cd frontend
npx tsc --noEmit      # type check, matches tsconfig.json's "strict": true
npm run lint          # ESLint, including no-explicit-any as a warning
```

## CI enforcement

See [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml):

- **mypy** runs in strict mode, piped through `mypy-baseline filter`. Only
  new violations fail the build; the job output distinguishes "new" counts
  from the unresolved (baselined) total.
- **tsc --noEmit** runs before the production build, so type errors fail
  fast.
- **ESLint** runs as an informational step (`npm run lint || true`) and does
  not currently fail CI, since some pre-existing rule violations
  (`react-hooks/*`, `react/display-name`) are outside the scope of this
  policy. `no-explicit-any` usages will show up here as warnings.

Total type-checking time (mypy + tsc, cold cache) is a few seconds to low
tens of seconds — well within the 2-minute budget for this step.
