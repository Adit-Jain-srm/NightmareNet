# Cross-Architecture Transfer: DistilBERT to BERT-large

## Objective
The goal of this experiment is to validate the generalizability of the NightmareNet robustness framework. Specifically, we investigate whether robustness improvements achieved by training a smaller model (DistilBERT) transfer to a larger model (BERT-large) on the same suite of distortions.

## Experimental Setup

### Models
- **Trained Model**: `distilbert-base-uncased`
- **Evaluation Model**: `bert-large-uncased`

### Dataset
- **Dataset**: IMDB (subset used for evaluation)
- **Task**: Sequence Classification

### Distortions
The following distortions were evaluated across varying strengths (0.1, 0.3, 0.5):
- `typo`
- `keyboard`
- `swap`
- `deletion`
- `homoglyph`

### Configuration
The experiment was executed using the configuration defined in `configs/examples/cross-arch-eval.yaml`.

## Results

Populate the table by running the cross-arch script (GPU recommended; peak memory stays under 4GB when models run sequentially):

```bash
python scripts/cross_arch_eval.py --config configs/examples/cross-arch-eval.yaml
```

Optional third phase (fine-tune BERT-large with NightmareNet):

```bash
python scripts/cross_arch_eval.py --config configs/examples/cross-arch-eval.yaml --finetune-large
```

Output is written to `cross_arch_eval_results.json`. Timing fields:
- **Inference Time (s/sample)** — single clean forward pass (`inference_time_per_sample`)
- **Eval Sweep (s/sample)** — full robustness sweep (`eval_sweep_time`)

| Model | Robustness Score | Clean Accuracy | Inference Time (s/sample) | Eval Sweep (s/sample) |
|-------|------------------|----------------|---------------------------|----------------------|
| DistilBERT + NightmareNet | run script | run script | run script | run script |
| BERT-large (zero-shot) | run script | run script | run script | run script |
| BERT-large + NightmareNet | run script | run script | run script | run script |

## Discussion
After running the script, compare `robustness_score` and `clean_accuracy` across rows in the JSON output. A smaller gap between DistilBERT and BERT-large suggests transferable robustness gains.

## Limitations
- Default config caps IMDB at 1000 samples for runtime; full-dataset numbers may differ.
- BERT-large zero-shot uses a randomly initialized classification head unless `--large-checkpoint` is supplied.
- Requires a CUDA GPU for practical runtimes; CPU-only runs are supported but slow.

## Conclusions
Run `python scripts/cross_arch_eval.py --config configs/examples/cross-arch-eval.yaml` and paste the JSON metrics into the table above once numbers are available on your hardware.
