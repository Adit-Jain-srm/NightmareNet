# Tutorial: CI Robustness Check (GitHub Action)

Gate pull requests on NightmareNet robustness scores with a reusable composite
Action: `nightmarenet-robustness-check` / `.github/actions/robustness-check`.

## Minimal integration

Add a workflow with five YAML lines of Action usage:

```yaml
- uses: actions/checkout@v4
- uses: Adit-Jain-srm/NightmareNet/.github/actions/robustness-check@v1
  with:
    model_path: distilbert-base-uncased
    threshold: "0.7"
```

A complete example lives at [`examples/ci-robustness-check.yml`](../../examples/ci-robustness-check.yml).

## Inputs

| Input | Default | Purpose |
| --- | --- | --- |
| `model_path` | — | Hugging Face model ID, or path to `.pt` / `.bin` |
| `config_path` | — | Optional YAML for full `Evaluator` runs |
| `threshold` | `0.7` | Minimum aggregate robustness score |
| `distortion_types` | `dream,nightmare` | Which families to include in the score table |
| `strengths` | `0.1,0.3,0.5,0.7,0.9` | Distortion sweep |
| `post-comment` | `true` | Upsert a PR comment with the markdown table |

## Outputs

| Output | Meaning |
| --- | --- |
| `robustness_score` | Aggregate score in `[0, 1]` |
| `passed` | `true` if score ≥ threshold |
| `report_path` | JSON report on the runner (also uploaded as an artifact) |

## What the PR comment shows

- Overall score and pass/fail
- Per-distortion (dream/nightmare × strength) scores in a markdown table

The job **fails** when the score is below `threshold`, so you can require the
check as a branch protection status.

## Runtime notes

- Without `config_path`, the Action uses the fast CLI path
  (`nightmarenet evaluate --json`) — CPU-only and typically well under 5 minutes
  for small models / probe text.
- With `config_path`, the Action loads the Hugging Face / checkpoint model via
  the full evaluator (heavier; prefer a small DistilBERT-class model in CI).
- Publish / Marketplace: the Action ships with `branding` (`shield` / `purple`).
  Maintainers can publish a tagged release (`v1`) to the GitHub Marketplace.

## Local smoke test

```bash
# From a clone with nightmarenet installed:
export INPUT_MODEL_PATH=distilbert-base-uncased
export INPUT_THRESHOLD=0.5
export INPUT_DISTORTION_TYPES=dream,nightmare
export INPUT_STRENGTHS=0.1,0.5,0.9
export GITHUB_OUTPUT=/tmp/nn-gh-out.txt
python .github/actions/robustness-check/entrypoint.py
```

Optional: validate `action.yml` with [actionlint](https://github.com/rhysd/actionlint)
and run end-to-end with [act](https://github.com/nektos/act).
