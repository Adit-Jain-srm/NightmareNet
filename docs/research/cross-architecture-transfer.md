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
*Note: The following table contains placeholder results pending the final execution of the pipeline.*

| Model | Robustness Score | Clean Accuracy | Inference Time (s/sample) |
|-------|------------------|----------------|---------------------------|
| DistilBERT + NightmareNet | TBD | TBD | TBD |
| BERT-large (zero-shot) | TBD | TBD | TBD |
| BERT-large + NightmareNet | TBD | TBD | TBD |

## Discussion
(To be updated after running the benchmark)

## Limitations
(To be updated after running the benchmark)

## Conclusions
(To be updated after running the benchmark)
