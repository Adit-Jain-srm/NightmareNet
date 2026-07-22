# Releasing NightmareNet

This project uses [Conventional Commits](https://www.conventionalcommits.org/)
and [release-please](https://github.com/googleapis/release-please) to automate
version bumps, changelog updates, and GitHub Releases.

## Version scheme

Semantic Versioning (`MAJOR.MINOR.PATCH`), tracked in:

- `pyproject.toml` (`[project].version`)
- `nightmarenet/__init__.py` (`__version__`)
- `.release-please-manifest.json` (last released version)

Tags look like `v0.2.1` (leading `v`).

| Commit prefix | Version bump |
|---------------|--------------|
| `fix:` | patch |
| `feat:` | minor |
| `feat!:` / `BREAKING CHANGE:` | major |

## Automated flow (default)

1. Merge PRs to `main` using conventional commit titles.
2. On each push to `main`, `.github/workflows/release-please.yml` runs.
3. release-please opens or updates a **Release PR** that:
   - bumps the version
   - updates `CHANGELOG.md`
   - updates `.release-please-manifest.json`
4. Merge the Release PR.
5. release-please creates the git tag and a GitHub Release with auto-generated notes.
6. The tag triggers `.github/workflows/release.yml`, which:
   - builds the wheel and sdist
   - attaches them to the GitHub Release
   - publishes to PyPI (trusted publishing / `pypi` environment)

No manual tagging is required for day-to-day releases after the bootstrap below.

## Bootstrap: first release (`v0.2.1`)

The package is prepared at **0.2.1**. After the release-automation PR is on
`main`, create the first tag once to validate the pipeline:

```bash
git checkout main
git pull
git tag v0.2.1
git push origin v0.2.1
```

That runs `release.yml`: changelog/notes on the GitHub Release, wheel/sdist
assets, and PyPI publish (if the `pypi` environment is configured).

Later releases should go through release-please only (do not hand-edit version
numbers on `main` unless fixing a mistake).

## Manual checklist (fallback)

Use only if Actions are unavailable:

1. Ensure `pyproject.toml` and `nightmarenet/__init__.py` share the same version.
2. Move notes under `## [Unreleased]` in `CHANGELOG.md` into `## [X.Y.Z] - YYYY-MM-DD`.
3. Commit: `chore: release X.Y.Z`
4. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`
5. Confirm the GitHub Release has notes + `dist` assets and that PyPI updated.

## Files

| File | Purpose |
|------|---------|
| `.github/workflows/release-please.yml` | Opens Release PRs / creates tags |
| `release-please-config.json` | Python package release config |
| `.release-please-manifest.json` | Last released version |
| `.github/workflows/release.yml` | Build, GitHub Release assets, PyPI |
| `CHANGELOG.md` | Human-readable history |

## Permissions note

The release-please workflow needs permission for Actions to open PRs (repo
**Settings → Actions → General → Allow GitHub Actions to create and approve
pull requests**). The default `GITHUB_TOKEN` is enough when that is enabled.
