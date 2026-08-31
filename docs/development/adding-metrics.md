# Adding an Evaluation Metric

NightmareNet's evaluation metrics are plain functions in [`nightmarenet/evaluation/metrics.py`](../../nightmarenet/evaluation/metrics.py), dispatched by name from the `Evaluator`. Adding a metric is a four-step change: implement the function, export it, wire a dispatch branch, and enable it in config.

## How metrics are selected

The `Evaluator` in [`nightmarenet/evaluation/evaluator/core.py`](../../nightmarenet/evaluation/evaluator/core.py) reads the list of enabled metrics from config and dispatches on the string name:

```python
default_metrics = (
    ["classification", "robustness"]
    if config.get("model", {}).get("type") == "image_classification"
    else ["recall", "generalization", "robustness", "hallucination"]
)
self.enabled_metrics = list(self.eval_config.get("metrics", default_metrics))
```

`Evaluator.evaluate(...)` then runs each enabled metric inside a `try/except`, storing the result (or an `{"error": ...}` dict) under the metric name.

## Existing metric functions

All metric functions live in `metrics.py` and are re-exported from [`nightmarenet/evaluation/__init__.py`](../../nightmarenet/evaluation/__init__.py):

| Function | Signature (abridged) | Purpose |
|---|---|---|
| `recall_score` | `(model, dataloader, tokenizer, device)` | Token accuracy + perplexity on clean data. |
| `generalization_score` | `(model, ood_dataloader, clean_dataloader, device)` | OOD-vs-clean perplexity ratio. |
| `robustness_score` | `(model, base_dataset, tokenizer, distortion_fn, strengths=None, ...)` | AUC over a distortion-strength sweep. |
| `hallucination_rate` | `(model, factual_dataloader, tokenizer, device, confidence_threshold=0.5)` | High-confidence wrong-prediction rate. |
| `classification_metrics` | `(model, dataloader, device, return_per_sample=False)` | Accuracy + weighted/per-class F1. |

Each returns a `dict` that includes a `"metric"` key. NaN/Inf values are guarded with the `_safe_float` helper — reuse it in new metrics.

## Step-by-step: add a metric

Suppose you want to add a `calibration` metric.

### 1. Implement the function in `metrics.py`

```python
def calibration_score(
    model,
    dataloader: DataLoader,
    device="cpu",
) -> dict:
    """Expected Calibration Error (ECE) on a labelled dataset.

    Returns:
        Dict with the metric name and the ECE value.
    """
    model.eval()
    # ... compute ece ...
    ece = 0.0
    return {
        "metric": "calibration",
        "ece": _safe_float(ece),
    }
```

Follow the conventions from `CONTRIBUTING.md`: module logger (no `print`), Google-style docstring on the public function, and wrap suspicious arithmetic with `_safe_float`.

### 2. Export it from `evaluation/__init__.py`

```python
from nightmarenet.evaluation.metrics import (
    calibration_score as calibration_score,
    ...
)
```

### 3. Add a dispatch branch in `evaluator/core.py`

Inside `Evaluator.evaluate(...)`, add a branch mirroring the existing ones:

```python
if "calibration" in self.enabled_metrics:
    logger.info("Evaluating: calibration")
    try:
        results["calibration"] = calibration_score(
            self.model, clean_dataloader, self.device
        )
        if self.tracker:
            self._log_eval("calibration", results["calibration"])
    except Exception as e:
        logger.error("Failed to compute calibration: %s", e)
        results["calibration"] = {"error": str(e)}
```

### 4. Enable it in config

Add the metric name to the `evaluation.metrics` list in your YAML config (the default list lives in `nightmarenet/utils/config.py`):

```yaml
evaluation:
  metrics:
    - recall
    - generalization
    - robustness
    - hallucination
    - calibration
```

## Testing

Add tests under `tests/` (e.g. `tests/test_metrics.py`) covering:

- The happy path returns a dict with the expected `"metric"` key and value range.
- Edge cases (empty dataloader, single class) do not raise and return safe defaults.
- NaN/Inf inputs are neutralized (values pass through `_safe_float`).

```python
def test_calibration_score_returns_metric_key():
    result = calibration_score(model, dataloader, device="cpu")
    assert result["metric"] == "calibration"
    assert 0.0 <= result["ece"] <= 1.0
```

Run with `pytest tests/test_metrics.py` — see the [Testing Guide](testing.md).

## Related Documentation

- [Architecture](architecture.md) — where evaluation sits in the pipeline.
- [Testing Guide](testing.md) — running and writing tests.
- [Code Style](code-style.md) — the `_safe_float` / no-`print` conventions.
