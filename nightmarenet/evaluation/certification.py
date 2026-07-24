"""Certified robustness verification via randomized smoothing (Cohen et al., 2019).

Wraps a base classifier with Gaussian noise sampled in *embedding space* (not token
space) to produce a smoothed classifier with a provable certified L2 radius: for a given
input, no perturbation with embedding-space L2 norm below the certified radius can change
the smoothed classifier's prediction.

This module is a formal complement to NightmareNet's existing empirical robustness_score
(see nightmarenet/evaluation/metrics.py). robustness_score measures how gracefully the
model degrades under specific distortions actually tried; certification here answers a
different question: for THIS input, what radius is provably safe against *any*
perturbation an adversary might choose (within the noise model), not just the ones tested?

Design note -- embedding-space vs. token-space noise:
Randomized smoothing (Cohen et al. 2019) was developed for continuous input domains
(images), where Gaussian noise is added directly to pixel values. Text is discrete --
there's no well-defined way to add continuous noise to token IDs. Following the standard
NLP adaptation, noise is instead injected into the model's continuous embedding vectors,
via a forward hook on `model.get_input_embeddings()`, and the certified radius is
therefore a radius in embedding space, not token/edit-distance space. This is a
meaningfully different guarantee than the vision-domain radius and must be surfaced as
such wherever certified radii are reported downstream (see format_results.py, sub-PR 3).

Statistical note -- two-stage CERTIFY:
Following Cohen et al.'s original procedure, certify_sample uses two *independent*
sample batches: a small n0 to select a candidate class, and a separate n to estimate its
probability. Reusing the same samples for both selection and the confidence bound would
bias the bound upward (the class is chosen *because* it had the most votes in that batch),
understating the true uncertainty -- keeping the batches independent is what makes the
reported (1 - alpha) confidence level valid.

Scope: only randomized smoothing is implemented here. Interval bound propagation (IBP)
is explicitly out of scope -- see parent issue #153. This module has no dependency on
evaluator.py or the config system -- it is standalone.

Reference:
    Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). Certified Adversarial Robustness via
    Randomized Smoothing. ICML 2019. https://arxiv.org/abs/1902.02918
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CertificationResult:
    """Result of certifying a single sample via randomized smoothing.

    Attributes:
        prediction: The smoothed classifier's predicted class index (the plurality vote
            across n noisy copies), regardless of whether certification succeeded.
        certified_radius: Certified L2 radius in embedding space. 0.0 if abstained --
            per issue #160's spec, abstention returns a radius of 0.0 rather than None,
            so aggregate statistics (e.g. mean radius) can treat abstained samples as
            contributing no certified radius without special-casing None.
        p_a_lower: Clopper-Pearson lower confidence bound on the probability that a noisy
            copy predicts `prediction`.
        n_samples_used: Number of noisy forward passes consumed (<= requested n if a
            compute budget clamped it).
        abstained: True if p_a_lower <= 0.5 -- the smoothed classifier's confidence in its
            own top class was insufficient to certify any radius.
        label: The ground-truth label passed in, if any (for certified-accuracy reporting).
        correct: Whether `prediction == label`. None if no label was provided.
    """

    prediction: int
    certified_radius: float
    p_a_lower: float
    n_samples_used: int
    abstained: bool
    label: Optional[int] = None
    correct: Optional[bool] = None


def _empty_certification_result() -> dict[str, Any]:
    """Aggregate result returned when there are zero samples to certify.

    Shared by both emptiness checks in certify_dataset (before and after subset
    selection) so the "no samples" shape is defined in exactly one place.
    """
    return {
        "metric": "certification",
        "n_samples": 0,
        "certified_radius_mean": 0.0,
        "certified_radius_median": 0.0,
        "certification_abstain_rate": 0.0,
        "certified_accuracy": None,
        "results": [],
    }


def clopper_pearson_lower_bound(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound on a binomial proportion.

    Given k successes out of n independent Bernoulli trials, returns the lower bound of a
    (1 - alpha) one-sided confidence interval on the true success probability p. This is
    an exact bound (not a normal approximation), computed via the inverse CDF of the Beta
    distribution -- the standard construction used in Cohen et al. (2019) for CERTIFY:
    `scipy.stats.beta.ppf(alpha, k, n - k + 1)`.

    Args:
        k: Number of trials in which the event of interest occurred.
        n: Total number of trials.
        alpha: Significance level (e.g. 0.001 for 99.9% one-sided confidence).

    Returns:
        Lower bound on the true success probability, in [0, 1]. Exactly 0.0 when k == 0
        (no successes observed -- the honest lower bound is 0).

    Raises:
        ValueError: If n <= 0, k is out of [0, n], or alpha is not in (0, 1).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (0 <= k <= n):
        raise ValueError(f"k ({k}) must be in [0, n] ({n})")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if k == 0:
        return 0.0

    return float(beta_dist.ppf(alpha, k, n - k + 1))


def _make_noise_hook(sigma: float) -> Any:
    """Build a forward hook that adds i.i.d. Gaussian noise to an embedding layer's output.

    Each element of the hooked module's output tensor gets independent noise (torch's RNG
    draws one value per element of the tensor), so when the embedding layer is called on a
    batch of `batch_size` repeated copies of the same input, each copy receives an
    independent noise sample -- exactly the randomized-smoothing noise model.
    """

    def hook(module: Any, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        return output + torch.randn_like(output) * sigma

    return hook


def _run_noisy_forward_passes(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sigma: float,
    n: int,
    num_classes: int,
    batch_size: int,
    device: str,
) -> Any:
    """Run n noisy forward passes and return a histogram of predicted classes.

    Noise is injected via a forward hook on the model's input embedding layer (see
    _make_noise_hook), not by manually recomputing embeddings -- this way the model's own
    forward pass runs exactly as it normally would (including any positional/token-type
    embeddings added after the word-embedding lookup), just with noise added at that one
    point. The hook is always removed before returning, even on error, since a leaked hook
    would silently inject noise into every subsequent normal inference call on this model.

    Batched, not sequential: the input is repeated up to `batch_size` times per chunk
    (each repetition receiving independent noise via the hook), making n=1000 take ~10
    forward passes at batch_size=100 rather than 1000 sequential ones.

    Reproducibility note: results are deterministic for a *fixed* (seed, batch_size) pair
    (see certify_sample's determinism guarantee), but changing batch_size will generally
    change the specific noise values drawn even under the same seed. This is a property of
    PyTorch's CPU torch.randn/normal_ kernel, which is internally vectorized -- the exact
    values it produces depend on the total element count requested per call, not purely on
    seed + sequential position. This doesn't affect certification validity (every
    batch_size still draws n i.i.d. Gaussian samples), only bit-for-bit reproducibility
    across different batch_size settings.

    Args:
        model: A transformers.PreTrainedModel-style classifier exposing
            get_input_embeddings() and accepting input_ids= in its forward signature.
        input_ids: Token IDs for a single example, shape (1, seq_len).
        attention_mask: Optional attention mask for the example, same leading shape.
        sigma: Standard deviation of the Gaussian smoothing noise, in embedding space.
        n: Number of noisy samples to draw.
        num_classes: Number of output classes (histogram size).
        batch_size: Max noisy copies processed in a single forward pass.
        device: Device to run inference on. The model is assumed to already live there.

    Returns:
        Integer array of shape (num_classes,): counts[c] = number of noisy copies whose
        top-1 prediction was class c.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    if n <= 0:
        return counts
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    was_training = model.training
    model.eval()
    embedding_layer = model.get_input_embeddings()
    hook_handle = embedding_layer.register_forward_hook(_make_noise_hook(sigma))

    try:
        remaining = n
        with torch.no_grad():
            while remaining > 0:
                this_batch = min(batch_size, remaining)

                batch_kwargs = {"input_ids": input_ids.repeat(this_batch, 1)}
                if attention_mask is not None:
                    batch_kwargs["attention_mask"] = attention_mask.repeat(this_batch, 1)

                outputs = model(**batch_kwargs)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()

                counts += np.bincount(preds, minlength=num_classes)[:num_classes]
                remaining -= this_batch
    finally:
        # Hook cleanup is not optional: leaving it attached would inject noise into every
        # subsequent normal (non-certification) inference call on this model.
        hook_handle.remove()
        # Certification must not silently leave a model the caller was training stuck in
        # eval mode afterward.
        model.train(was_training)

    return counts


def certify_sample(
    model: Any,
    tokenizer: Any,
    text: str,
    label: Optional[int] = None,
    sigma: float = 0.25,
    n: int = 1000,
    alpha: float = 0.001,
    *,
    n0: int = 100,
    num_classes: Optional[int] = None,
    batch_size: int = 100,
    max_length: int = 128,
    certification_budget: Optional[int] = None,
    device: str = "cpu",
) -> CertificationResult:
    """Certify a single text input via randomized smoothing.

    Tokenizes `text` and runs the two-stage CERTIFY procedure from Cohen et al. (2019):

      1. Selection: draw n0 noisy embedding-space copies, take the plurality-vote class
         as the candidate prediction c_A. This stage is *not* used in the statistical
         bound below -- only to pick which class to test.
      2. Estimation: draw n further noisy copies, *independent* of the selection stage,
         and count how many predict c_A. Compute a (1 - alpha) one-sided Clopper-Pearson
         lower bound p_A on the true probability that a noisy copy predicts c_A.

    Using the same samples for both selection and the confidence bound (a single-stage
    shortcut) biases k upward, since the class was chosen *because* it had the most votes
    in that same batch -- the reported confidence would then overstate the true coverage.
    Keeping the two stages independent (fresh noise for each) is what makes the bound
    statistically valid at the declared (1 - alpha) level.

    If p_A > 0.5, returns a certified L2 (embedding-space) radius of `sigma * Phi^-1(p_A)`;
    otherwise abstains (radius 0.0, abstained=True).

    Args:
        model: PreTrainedModel-style classifier exposing get_input_embeddings() and
            accepting input_ids=/attention_mask= in its forward signature. Assumed to
            already be on `device`.
        tokenizer: Tokenizer used to encode `text`.
        text: The input text to certify.
        label: Optional ground-truth label, for certified-accuracy reporting.
        sigma: Standard deviation of the Gaussian smoothing noise, in embedding space.
        n: Requested number of estimation-stage noisy forward-pass samples.
        alpha: Significance level for the Clopper-Pearson bound (e.g. 0.001 = 99.9%
            one-sided confidence).
        n0: Number of selection-stage samples, independent of `n`. Cheap relative to `n`
            since it only needs to pick a candidate class, not bound its probability.
        num_classes: Number of output classes. Inferred from model.config.num_labels if
            omitted.
        batch_size: Max noisy copies processed in a single forward pass.
        max_length: Max token length for tokenization.
        certification_budget: Optional hard cap on n0 + n for this sample. n0 is
            preserved as long as possible (it's cheap and needed just to pick a candidate
            class); n is reduced first. If the budget doesn't even cover n0, n0 itself is
            reduced and the sample is abstained on without running the estimation stage
            (graceful degradation rather than a silent skip).
        device: Device to run inference on.

    Returns:
        CertificationResult with the smoothed prediction, certified radius, the
        Clopper-Pearson bound used, total samples consumed (n0 + n), and (if `label`
        given) correctness.
    """
    if num_classes is None:
        num_classes = getattr(getattr(model, "config", None), "num_labels", None)
        if num_classes is None:
            raise ValueError(
                "num_classes could not be inferred from model.config.num_labels; "
                "pass it explicitly."
            )

    n0_actual, n_actual = n0, n
    if certification_budget is not None:
        if n0 >= certification_budget:
            n0_actual, n_actual = max(0, certification_budget), 0
        else:
            n0_actual, n_actual = n0, min(n, certification_budget - n0)
        if n0_actual < n0 or n_actual < n:
            logger.warning(
                "Certification budget clamped sample: requested n0=%d n=%d, "
                "using n0=%d n=%d (budget=%d)",
                n0,
                n,
                n0_actual,
                n_actual,
                certification_budget,
            )

    encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Stage 1: selection.
    selection_counts = _run_noisy_forward_passes(
        model, input_ids, attention_mask, sigma, n0_actual, num_classes, batch_size, device
    )
    if selection_counts.sum() == 0:
        # Budget exhausted before selection could even run -- nothing to certify.
        return CertificationResult(
            prediction=-1,
            certified_radius=0.0,
            p_a_lower=0.0,
            n_samples_used=0,
            abstained=True,
            label=label,
            correct=None,
        )
    prediction = int(selection_counts.argmax())

    if n_actual <= 0:
        # Have a candidate class but no (independent) statistical grounds to certify it.
        return CertificationResult(
            prediction=prediction,
            certified_radius=0.0,
            p_a_lower=0.0,
            n_samples_used=n0_actual,
            abstained=True,
            label=label,
            correct=(prediction == label) if label is not None else None,
        )

    # Stage 2: estimation, statistically independent fresh samples.
    estimation_counts = _run_noisy_forward_passes(
        model, input_ids, attention_mask, sigma, n_actual, num_classes, batch_size, device
    )
    k = int(estimation_counts[prediction])
    p_a_lower = clopper_pearson_lower_bound(k, n_actual, alpha)

    if p_a_lower <= 0.5:
        logger.debug(
            "Abstained: p_A=%.4f <= 0.5 (k=%d/%d, sigma=%.3f, alpha=%.4f)",
            p_a_lower,
            k,
            n_actual,
            sigma,
            alpha,
        )
        return CertificationResult(
            prediction=prediction,
            certified_radius=0.0,
            p_a_lower=p_a_lower,
            n_samples_used=n0_actual + n_actual,
            abstained=True,
            label=label,
            correct=(prediction == label) if label is not None else None,
        )

    radius = sigma * float(norm.ppf(p_a_lower))
    return CertificationResult(
        prediction=prediction,
        certified_radius=radius,
        p_a_lower=p_a_lower,
        n_samples_used=n0_actual + n_actual,
        abstained=False,
        label=label,
        correct=(prediction == label) if label is not None else None,
    )


def certify_dataset(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    *,
    text_column: str = "text",
    label_column: Optional[str] = "label",
    sigma: float = 0.25,
    n: int = 1000,
    n0: int = 100,
    alpha: float = 0.001,
    subset_size: Optional[int] = None,
    batch_size: int = 100,
    max_length: int = 128,
    certification_budget_total: Optional[int] = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Certify a subset of a dataset and return aggregate certification statistics.

    Standalone dataset-level wrapper around certify_sample -- no dependency on
    evaluator.py or the config system. Intended to be called directly today, and wired
    into evaluator.py's optional `certify` flag in a later PR.

    Args:
        model: See certify_sample.
        tokenizer: See certify_sample.
        dataset: A HuggingFace Dataset (or anything supporting len() + indexing with
            dict-like examples).
        text_column: Column name containing the input text.
        label_column: Column name containing the ground-truth label, or None if the
            dataset has no labels (correctness won't be reported).
        sigma, n, n0, alpha, batch_size, max_length, device: See certify_sample.
        subset_size: If given, certify only a random (seeded) subset of this size rather
            than the full dataset -- certification is far more expensive per-sample than
            the empirical metrics, so a full-dataset run is rarely appropriate. A
            subset_size of 0 (or one that happens to select zero rows) is treated the
            same as an empty dataset.
        certification_budget_total: Optional hard cap on total forward passes summed
            across every sample in this call. Divided evenly across samples (graceful
            degradation: every sample gets a smaller n rather than some samples being
            skipped entirely). Finer-grained budget allocation strategies are left to the
            evaluator.py integration (sub-PR 2).
        device: Device to run inference on.

    Returns:
        Dict with certified_radius_mean, certified_radius_median,
        certification_abstain_rate, certified_accuracy (if labels provided), n_samples,
        and the full list of per-sample CertificationResult objects under "results".
    """
    if len(dataset) == 0:
        return _empty_certification_result()

    if subset_size is not None:
        dataset = dataset.shuffle(seed=42).select(range(min(subset_size, len(dataset))))
        # subset_size may be 0 (or otherwise leave nothing selected), so re-check
        # emptiness here rather than assuming the original dataset length still
        # applies -- otherwise the divisions below (certification_budget_total //
        # len(dataset), and abstain_count / len(results) further down) would raise
        # ZeroDivisionError instead of returning a clean empty result.
        if len(dataset) == 0:
            return _empty_certification_result()

    per_sample_budgets: list[int | None]

    if certification_budget_total is not None:
        base = certification_budget_total // len(dataset)
        remainder = certification_budget_total % len(dataset)
        per_sample_budgets = [base + 1 if i < remainder else base for i in range(len(dataset))]
        logger.info(
            "certification_budget_total=%d over %d samples -> %d-%d forward passes/sample",
            certification_budget_total,
            len(dataset),
            base,
            base + (1 if remainder else 0),
        )
    else:
        per_sample_budgets = [None] * len(dataset)

    results: list[CertificationResult] = []
    progress = tqdm(dataset, desc="Certifying samples (randomized smoothing)")
    for example, sample_budget in zip(progress, per_sample_budgets):
        label = example.get(label_column) if label_column else None
        result = certify_sample(
            model,
            tokenizer,
            example[text_column],
            label=label,
            sigma=sigma,
            n=n,
            n0=n0,
            alpha=alpha,
            batch_size=batch_size,
            max_length=max_length,
            certification_budget=sample_budget,
            device=device,
        )
        results.append(result)

    # Abstained samples already carry certified_radius=0.0 by construction (see
    # CertificationResult), so they're included directly here rather than filtered out --
    # excluding them would inflate the mean/median radius by only counting the samples
    # that happened to succeed. Same for accuracy: a sample the classifier couldn't
    # certify is not a certified-correct prediction, so abstained samples with a label
    # count as incorrect (0) rather than being dropped from the denominator entirely.
    radii = [r.certified_radius for r in results]
    abstain_count = sum(1 for r in results if r.abstained)
    correctness = [
        (r.correct is True) if not r.abstained else False for r in results if r.label is not None
    ]

    return {
        "metric": "certification",
        "n_samples": len(results),
        "certified_radius_mean": float(np.mean(radii)) if radii else 0.0,
        "certified_radius_median": float(np.median(radii)) if radii else 0.0,
        "certification_abstain_rate": abstain_count / len(results),
        "certified_accuracy": (sum(correctness) / len(correctness)) if correctness else None,
        "results": results,
    }
