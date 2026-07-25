"""IBM Adversarial Robustness Toolbox (ART) integration for NightmareNet.

Provides an adapter layer that wraps NightmareNet models as ART
``PyTorchClassifier`` instances, enabling standardized adversarial
attacks (PGD, FGSM, C&W) and comparable robustness metrics.

Install the optional dependency with::

    pip install 'nightmarenet[art]'
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported attack registry
# ---------------------------------------------------------------------------

SUPPORTED_ATTACKS: Dict[str, str] = {
    "pgd": "ProjectedGradientDescent",
    "fgsm": "FastGradientMethod",
    "cw": "CarliniL2Method",
}

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


def _check_art_available() -> None:
    """Raise ``ImportError`` with install instructions if ART is missing."""
    try:
        import art  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The Adversarial Robustness Toolbox (ART) is not installed. "
            "Install it with:  pip install 'nightmarenet[art]'"
        ) from exc


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ARTAttackResult:
    """Container for the outcome of a single ART attack run.

    Attributes:
        attack_name: Human-readable attack identifier (e.g. ``"pgd"``).
        success_rate: Fraction of samples where the attack changed the prediction.
        mean_perturbation: Mean L2 perturbation magnitude across all adversarial examples.
        median_perturbation: Median L2 perturbation magnitude.
        query_count: Total forward-pass queries made by the attack.
        elapsed_seconds: Wall-clock time taken for the attack.
        adversarial_examples: The generated adversarial inputs (numpy array).
        original_predictions: Model predictions on clean inputs.
        adversarial_predictions: Model predictions on adversarial inputs.
    """

    attack_name: str
    success_rate: float
    mean_perturbation: float
    median_perturbation: float
    query_count: int
    elapsed_seconds: float
    adversarial_examples: Optional[Any] = field(default=None, repr=False)
    original_predictions: Optional[Any] = field(default=None, repr=False)
    adversarial_predictions: Optional[Any] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class NightmareNetARTClassifier:
    """Wrapper that adapts a ``torch.nn.Module`` into an ART ``PyTorchClassifier``.

    Example::

        from nightmarenet.evaluation.art_adapter import NightmareNetARTClassifier

        wrapper = NightmareNetARTClassifier(
            model=my_torch_model,
            nb_classes=10,
            input_shape=(3, 32, 32),
        )
        art_classifier = wrapper.classifier  # ready for ART attacks
    """

    def __init__(
        self,
        model: Any,
        nb_classes: int,
        input_shape: Tuple[int, ...],
        clip_values: Tuple[float, float] = (0.0, 1.0),
        loss: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        device_type: str = "cpu",
    ) -> None:
        _check_art_available()

        import torch
        from art.estimators.classification import PyTorchClassifier

        if loss is None:
            loss = torch.nn.CrossEntropyLoss()

        self._model = model
        self.classifier = PyTorchClassifier(
            model=model,
            loss=loss,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=clip_values,
            device_type=device_type,
        )
        logger.info(
            "Wrapped model as ART PyTorchClassifier (classes=%d, shape=%s)",
            nb_classes,
            input_shape,
        )

    @classmethod
    def from_nightmarenet_model(
        cls,
        model: Any,
        nb_classes: int,
        input_shape: Tuple[int, ...],
        **kwargs: Any,
    ) -> NightmareNetARTClassifier:
        """Factory constructor for convenience."""
        return cls(model=model, nb_classes=nb_classes, input_shape=input_shape, **kwargs)


# ---------------------------------------------------------------------------
# Attack runners
# ---------------------------------------------------------------------------


def _build_attack(classifier: Any, attack_name: str, **kwargs: Any) -> Any:
    """Instantiate an ART attack object by name."""
    name = attack_name.lower()
    if name not in SUPPORTED_ATTACKS:
        raise ValueError(
            f"Unsupported attack: {attack_name!r}. "
            f"Supported: {list(SUPPORTED_ATTACKS.keys())}"
        )

    _check_art_available()

    if name == "pgd":
        from art.attacks.evasion import ProjectedGradientDescent

        return ProjectedGradientDescent(
            estimator=classifier,
            eps=kwargs.get("eps", 0.3),
            eps_step=kwargs.get("eps_step", 0.01),
            max_iter=kwargs.get("max_iter", 40),
            targeted=kwargs.get("targeted", False),
            batch_size=kwargs.get("batch_size", 32),
        )
    elif name == "fgsm":
        from art.attacks.evasion import FastGradientMethod

        return FastGradientMethod(
            estimator=classifier,
            eps=kwargs.get("eps", 0.3),
            targeted=kwargs.get("targeted", False),
            batch_size=kwargs.get("batch_size", 32),
        )
    elif name == "cw":
        from art.attacks.evasion import CarliniL2Method

        return CarliniL2Method(
            classifier=classifier,
            confidence=kwargs.get("confidence", 0.0),
            max_iter=kwargs.get("max_iter", 10),
            batch_size=kwargs.get("batch_size", 32),
            learning_rate=kwargs.get("learning_rate", 0.01),
        )

    # Should not reach here due to the check above, but satisfy type checkers.
    raise ValueError(f"Unsupported attack: {attack_name!r}")  # pragma: no cover


def run_art_attack(
    classifier: Any,
    attack_name: str,
    x: Any,
    y: Any,
    **attack_kwargs: Any,
) -> ARTAttackResult:
    """Run a single ART attack against a wrapped model.

    Args:
        classifier: An ART-compatible classifier (e.g. from
            ``NightmareNetARTClassifier.classifier``).
        attack_name: One of ``"pgd"``, ``"fgsm"``, ``"cw"``.
        x: Clean input samples as a numpy array of shape ``(N, *input_shape)``.
        y: True labels as a numpy array of shape ``(N,)`` or one-hot ``(N, C)``.
        **attack_kwargs: Extra keyword arguments forwarded to the ART attack constructor.

    Returns:
        An ``ARTAttackResult`` with metrics and adversarial examples.
    """
    _check_art_available()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)

    attack = _build_attack(classifier, attack_name, **attack_kwargs)

    # Run the attack
    t0 = time.perf_counter()
    x_adv = attack.generate(x=x)
    elapsed = time.perf_counter() - t0

    # Predictions
    original_preds = np.argmax(classifier.predict(x), axis=1)
    adversarial_preds = np.argmax(classifier.predict(x_adv), axis=1)

    # True labels (handle one-hot)
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y

    # Success rate = fraction where adversarial prediction differs from original correct prediction
    correctly_classified = original_preds == true_labels
    if correctly_classified.sum() == 0:
        success_rate = 0.0
    else:
        flipped = adversarial_preds[correctly_classified] != true_labels[correctly_classified]
        success_rate = float(flipped.mean())

    # Perturbation magnitudes (L2)
    diff = (x_adv - x).reshape(len(x), -1)
    l2_norms = np.linalg.norm(diff, axis=1)

    # Query count heuristic — ART does not expose this directly.
    # PGD: max_iter forward + backward per sample; FGSM: 1; C&W: max_iter.
    name = attack_name.lower()
    max_iter = attack_kwargs.get("max_iter", 40 if name == "pgd" else (10 if name == "cw" else 1))
    query_count = int(len(x) * max_iter)

    return ARTAttackResult(
        attack_name=attack_name.lower(),
        success_rate=round(success_rate, 4),
        mean_perturbation=round(float(np.mean(l2_norms)), 6),
        median_perturbation=round(float(np.median(l2_norms)), 6),
        query_count=query_count,
        elapsed_seconds=round(elapsed, 3),
        adversarial_examples=x_adv,
        original_predictions=original_preds,
        adversarial_predictions=adversarial_preds,
    )


def run_art_benchmark(
    classifier: Any,
    x: Any,
    y: Any,
    attacks: Optional[Sequence[str]] = None,
    **attack_kwargs: Any,
) -> List[ARTAttackResult]:
    """Run multiple ART attacks and return aggregated results.

    Args:
        classifier: ART-compatible classifier.
        x: Clean input samples.
        y: True labels.
        attacks: Attack names to run. Defaults to ``["pgd", "fgsm", "cw"]``.
        **attack_kwargs: Extra keyword arguments forwarded to each attack constructor.

    Returns:
        List of ``ARTAttackResult``, one per attack.
    """
    if attacks is None:
        attacks = list(SUPPORTED_ATTACKS.keys())

    results: List[ARTAttackResult] = []
    for attack_name in attacks:
        logger.info("Running ART attack: %s", attack_name)
        result = run_art_attack(classifier, attack_name, x, y, **attack_kwargs)
        logger.info(
            "  %s — success_rate=%.2f%%, mean_L2=%.4f, time=%.1fs",
            result.attack_name,
            result.success_rate * 100,
            result.mean_perturbation,
            result.elapsed_seconds,
        )
        results.append(result)

    return results
