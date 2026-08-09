"""Evaluation metrics and engine."""

from nightmarenet.evaluation.evaluator import Evaluator as Evaluator
from nightmarenet.evaluation.glue import (
    evaluate_glue as evaluate_glue,
)
from nightmarenet.evaluation.glue import (
    evaluate_glue_task as evaluate_glue_task,
)
from nightmarenet.evaluation.metrics import (
    classification_metrics as classification_metrics,
)
from nightmarenet.evaluation.metrics import (
    compute_perplexity as compute_perplexity,
)
from nightmarenet.evaluation.metrics import (
    evaluate_cycle as evaluate_cycle,
)
from nightmarenet.evaluation.metrics import (
    generalization_score as generalization_score,
)
from nightmarenet.evaluation.metrics import (
    hallucination_rate as hallucination_rate,
)
from nightmarenet.evaluation.metrics import (
    quick_robustness_score as quick_robustness_score,
)
from nightmarenet.evaluation.metrics import (
    recall_score as recall_score,
)
from nightmarenet.evaluation.metrics import (
    robustness_score as robustness_score,
)

__all__ = [
    "Evaluator",
    "evaluate_glue",
    "evaluate_glue_task",
    "compute_perplexity",
    "quick_robustness_score",
    "evaluate_cycle",
    "recall_score",
    "generalization_score",
    "robustness_score",
    "hallucination_rate",
    "classification_metrics",
]
