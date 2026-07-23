"""Dream and nightmare data generation.

Applies distortions to a base dataset to produce dream (mildly distorted)
and nightmare (extremely perturbed) training splits.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional

import torch
from datasets import Dataset, IterableDataset

from nightmarenet.distortions.adversarial import apply_adversarial_distortions
from nightmarenet.distortions.loader import load_custom_engine
from nightmarenet.distortions.registry import get_registry
from nightmarenet.distortions.semantic import apply_semantic_distortions
from nightmarenet.distortions.text import apply_text_distortions
from nightmarenet.utils.validation import (
    validate_dataset_columns,
    validate_non_empty_dataset,
    validate_strength,
)

logger = logging.getLogger(__name__)


class DreamDatasetGenerator:
    """Generates mildly distorted dream data from a base dataset.

    Applies text-level and light semantic distortions to encourage
    the model to learn abstract, invariant representations.

    Args:
        strength: Distortion strength (0–1). Typical dream range: 0.2–0.3.
        text_column: Name of the text column in the dataset.
        config: Optional distortion config dict with per-type weights.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        strength: float = 0.25,
        text_column: str = "text",
        config: Optional[dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        self.strength = validate_strength(strength, "strength")
        self.text_column = text_column
        self.config = config or {}
        self.seed = seed

    def _distort(self, example: dict[str, Any]) -> dict[str, Any]:
        """Apply dream-level distortions to a single example."""
        text = example[self.text_column]
        if not text or not text.strip():
            return example

        result = text

        # Apply custom engines from config if specified
        custom_engines = self.config.get("custom_engines", [])
        if custom_engines:
            registry = get_registry()
            for engine_config in custom_engines:
                engine_name = engine_config.get("engine")
                engine_strength = engine_config.get("strength", self.strength)

                # Handle custom: prefix for file-based engines
                if engine_name and engine_name.startswith("custom:"):
                    loaded_name = load_custom_engine(engine_name, registry)
                    if loaded_name:
                        engine_name = loaded_name

                if engine_name and engine_name in registry:
                    result = registry.apply(
                        engine_name, result, strength=engine_strength, seed=self.seed
                    )

        # Apply text-level corruptions (primary for dream phase)
        text_config = self.config.get("text", None)
        result = apply_text_distortions(result, strength=self.strength, config=text_config)

        # Apply light semantic distortions
        semantic_config = self.config.get("semantic", None)
        result = apply_semantic_distortions(
            result, strength=self.strength * 0.5, config=semantic_config
        )

        return {**example, self.text_column: result}

    def generate(self, dataset: Any) -> Any:
        """Generate a dream dataset by applying mild distortions.

        Args:
            dataset: Base HuggingFace Dataset or IterableDataset to distort.

        Returns:
            A new Dataset/IterableDataset with mildly distorted text.
        """
        if not hasattr(dataset, "map"):
            return DistortedVisionDataset(dataset, self, phase="dream")

        import random

        random.seed(self.seed)

        # Streaming: lazily map distortions
        if isinstance(dataset, IterableDataset):
            logger.info(
                "Generating dream data (strength=%.2f) in streaming mode...",
                self.strength,
            )
            # Validate column when metadata is available
            features = getattr(dataset, "features", None)
            if features is not None and self.text_column not in features:
                raise ValueError(
                    f"Text column '{self.text_column}' not found in streaming dataset. "
                    f"Available columns: {list(features)}"
                )
            return dataset.map(self._distort)

        validate_dataset_columns(dataset, [self.text_column])
        validate_non_empty_dataset(dataset, "dataset")

        logger.info(
            "Generating dream data (strength=%.2f) from %d samples...",
            self.strength,
            len(dataset),
        )

        original_texts = dataset[self.text_column]

        dream_data = dataset.map(
            self._distort,
            desc="Generating dream data",
        )

        modified_count = sum(
            1 for o, g in zip(original_texts, dream_data[self.text_column]) if o != g
        )
        logger.info(
            "Dream data generation complete. %d samples produced, %d texts modified.",
            len(dream_data),
            modified_count,
        )
        return dream_data

    def generate_and_save(self, dataset: Dataset, output_dir: str) -> Dataset:
        """Generate dream data and save to disk.

        Args:
            dataset: Base dataset to distort.
            output_dir: Directory to save the generated dataset.

        Returns:
            The generated dream Dataset.
        """
        dream_data = self.generate(dataset)
        save_path = os.path.join(output_dir, "dream")
        try:
            os.makedirs(save_path, exist_ok=True)
            dream_data.save_to_disk(save_path)
        except OSError as exc:
            raise OSError(f"Failed to save dream data to '{save_path}': {exc}") from exc
        logger.info("Dream data saved to %s", save_path)
        return dream_data


class NightmareDatasetGenerator:
    """Generates extremely perturbed nightmare data from a base dataset.

    Applies aggressive text, semantic, and adversarial distortions to
    stress-test the model's learned representations.

    Args:
        strength: Distortion strength (0–1). Typical nightmare range: 0.7–0.9.
        text_column: Name of the text column in the dataset.
        config: Optional distortion config dict with per-type weights.
        seed: Random seed for reproducibility.
        strength_schedule: Scheduling strategy for distortion strength within batch.
            Options: "uniform" (default), "linear", "cosine", "step".
        strength_min: Minimum strength for scheduled variants (0–1).
        strength_max: Maximum strength for scheduled variants (0–1).
    """

    def __init__(
        self,
        strength: float = 0.8,
        text_column: str = "text",
        config: Optional[dict[str, Any]] = None,
        seed: int = 42,
        strength_schedule: str = "uniform",
        strength_min: float = 0.3,
        strength_max: float = 0.9,
        target_model: Optional[Any] = None,
        target_tokenizer: Optional[Any] = None,
        cycle_id: int = 0,
    ) -> None:
        self.strength = validate_strength(strength, "strength")
        self.text_column = text_column
        self.config = config or {}
        self.seed = seed
        self.strength_schedule = strength_schedule
        self.strength_min = validate_strength(strength_min, "strength_min")
        self.strength_max = validate_strength(strength_max, "strength_max")
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.cycle_id = int(cycle_id)

        if self.strength_schedule not in ("uniform", "linear", "cosine", "step"):
            raise ValueError(
                f"Invalid strength_schedule: {self.strength_schedule}. "
                "Must be one of: uniform, linear, cosine, step"
            )

        if self.strength_schedule != "uniform":
            logger.warning(
                "strength_schedule is non-uniform; nightmare_strength config will be ignored. "
                "Only strength_min and strength_max are used."
            )

        if self.strength_min > self.strength_max:
            logger.warning("strength_min > strength_max; schedule will decrease over batch.")

    @property
    def uses_gradient_learned(self) -> bool:
        """Return whether model-aware learned distortion is enabled."""
        adversarial = self.config.get("adversarial", {})
        return bool(
            adversarial.get("learned", 0.0) > 0.0
            and adversarial.get("learned_strategy", "attention") == "gradient"
        )

    def set_target_model(
        self,
        target_model: Optional[Any],
        target_tokenizer: Optional[Any] = None,
    ) -> None:
        """Set the current target model used by model-aware distortions."""
        self.target_model = target_model
        if target_tokenizer is not None:
            self.target_tokenizer = target_tokenizer

    def set_cycle(self, cycle_id: int) -> None:
        """Set the current training cycle for learned-example caching."""
        self.cycle_id = int(cycle_id)

    def _compute_strengths(self, num_samples: int) -> list[float]:
        """Compute per-sample distortion strengths based on schedule.

        Args:
            num_samples: Number of samples in the batch.

        Returns:
            List of strength values (0–1) for each sample.
        """
        if self.strength_schedule == "uniform":
            return [self.strength] * num_samples

        strengths = []
        for i in range(num_samples):
            # Normalized position in batch [0, 1]
            t = i / max(1, num_samples - 1)

            if self.strength_schedule == "linear":
                # Linear interpolation from min to max
                strength = self.strength_min + t * (self.strength_max - self.strength_min)
            elif self.strength_schedule == "cosine":
                # Cosine annealing from min to max
                strength = self.strength_min + (self.strength_max - self.strength_min) * (
                    0.5 * (1 - math.cos(math.pi * t))
                )
            elif self.strength_schedule == "step":
                # Step function: first half at min, second half at max
                strength = self.strength_min if t < 0.5 else self.strength_max
            else:
                # Fallback to uniform (should not reach here due to validation)
                strength = self.strength

            strengths.append(strength)

        return strengths

    def _distort(self, example: dict[str, Any], strength: Optional[float] = None) -> dict[str, Any]:
        """Apply nightmare-level distortions to a single example.

        Args:
            example: Dataset example dict.
            strength: Optional per-sample strength. If None, uses self.strength.

        Returns:
            Distorted example dict.
        """
        text = example[self.text_column]
        if not text or not text.strip():
            return example

        # Use provided strength or fall back to default
        actual_strength = strength if strength is not None else self.strength
        result = text

        # Apply custom engines from config if specified
        custom_engines = self.config.get("custom_engines", [])
        if custom_engines:
            registry = get_registry()
            for engine_config in custom_engines:
                engine_name = engine_config.get("engine")
                engine_strength = engine_config.get("strength", actual_strength)

                # Handle custom: prefix for file-based engines
                if engine_name and engine_name.startswith("custom:"):
                    loaded_name = load_custom_engine(engine_name, registry)
                    if loaded_name:
                        engine_name = loaded_name

                if engine_name and engine_name in registry:
                    result = registry.apply(
                        engine_name, result, strength=engine_strength, seed=self.seed
                    )

        # Apply aggressive text-level corruptions
        text_config = self.config.get("text", None)
        result = apply_text_distortions(result, strength=actual_strength, config=text_config)

        # Apply strong semantic distortions
        semantic_config = self.config.get("semantic", None)
        result = apply_semantic_distortions(
            result, strength=actual_strength, config=semantic_config
        )

        # Apply adversarial distortions (unique to nightmare phase)
        adversarial_config = self.config.get("adversarial", None)
        result = apply_adversarial_distortions(
            result,
            strength=actual_strength,
            config=adversarial_config,
            target_model=self.target_model,
            target_tokenizer=self.target_tokenizer,
            cycle_id=self.cycle_id,
        )

        return {**example, self.text_column: result}

    def generate(self, dataset: Any) -> Any:
        """Generate a nightmare dataset by applying extreme distortions.

        Args:
            dataset: Base HuggingFace Dataset or IterableDataset to distort.

        Returns:
            A new Dataset/IterableDataset with extremely perturbed text.
        """
        if not hasattr(dataset, "map"):
            return DistortedVisionDataset(dataset, self, phase="nightmare")

        import random

        random.seed(self.seed)

        # Streaming: lazily map distortions
        if isinstance(dataset, IterableDataset):
            logger.info(
                "Generating nightmare data (strength=%.2f, schedule=%s) in streaming mode...",
                self.strength,
                self.strength_schedule,
            )
            # Validate column when metadata is available
            features = getattr(dataset, "features", None)
            if features is not None and self.text_column not in features:
                raise ValueError(
                    f"Text column '{self.text_column}' not found in streaming dataset. "
                    f"Available columns: {list(features)}"
                )
            # For streaming, fall back to uniform strength (cannot pre-compute batch sizes)
            if self.strength_schedule != "uniform":
                logger.warning(
                    "Strength scheduling not supported for streaming datasets. "
                    "Falling back to uniform strength."
                )
            return dataset.map(self._distort)

        validate_dataset_columns(dataset, [self.text_column])
        validate_non_empty_dataset(dataset, "dataset")

        logger.info(
            "Generating nightmare data (strength=%.2f, schedule=%s) from %d samples...",
            self.strength,
            self.strength_schedule,
            len(dataset),
        )

        original_texts = dataset[self.text_column]

        # Pre-compute strengths for non-uniform schedules
        if self.strength_schedule == "uniform":
            nightmare_data = dataset.map(
                self._distort,
                desc="Generating nightmare data",
            )
        else:
            strengths = self._compute_strengths(len(dataset))

            def _distort_with_strength(example: dict[str, Any], idx: int) -> dict[str, Any]:
                return self._distort(example, strength=strengths[idx])

            nightmare_data = dataset.map(
                _distort_with_strength,
                with_indices=True,
                desc="Generating nightmare data",
            )

        modified_count = sum(
            1 for o, g in zip(original_texts, nightmare_data[self.text_column]) if o != g
        )
        logger.info(
            "Nightmare data generation complete. %d samples produced, %d texts modified.",
            len(nightmare_data),
            modified_count,
        )
        return nightmare_data

    def generate_and_save(self, dataset: Dataset, output_dir: str) -> Dataset:
        """Generate nightmare data and save to disk.

        Args:
            dataset: Base dataset to distort.
            output_dir: Directory to save the generated dataset.

        Returns:
            The generated nightmare Dataset.
        """
        nightmare_data = self.generate(dataset)
        save_path = os.path.join(output_dir, "nightmare")
        try:
            os.makedirs(save_path, exist_ok=True)
            nightmare_data.save_to_disk(save_path)
        except OSError as exc:
            raise OSError(f"Failed to save nightmare data to '{save_path}': {exc}") from exc
        logger.info("Nightmare data saved to %s", save_path)
        return nightmare_data


def create_generators_from_config(
    config: dict[str, Any],
) -> tuple[DreamDatasetGenerator, NightmareDatasetGenerator]:
    """Create dream and nightmare generators from a config dictionary.

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (DreamDatasetGenerator, NightmareDatasetGenerator).
    """
    distortion_config = config.get("distortion", {})
    dataset_config = config.get("dataset", {})
    seed = config.get("seed", 42)

    dream_gen = DreamDatasetGenerator(
        strength=distortion_config.get("dream_strength", 0.25),
        text_column=dataset_config.get("text_column", "text"),
        config=distortion_config,
        seed=seed,
    )

    nightmare_gen = NightmareDatasetGenerator(
        strength=distortion_config.get("nightmare_strength", 0.8),
        text_column=dataset_config.get("text_column", "text"),
        config=distortion_config,
        seed=seed,
        strength_schedule=distortion_config.get("strength_schedule", "uniform"),
        strength_min=distortion_config.get("strength_min", 0.3),
        strength_max=distortion_config.get("strength_max", 0.9),
    )

    return dream_gen, nightmare_gen


class DistortedVisionDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(
        self,
        dataset: Any,
        generator: Any,
        phase: str = "dream",
    ) -> None:
        self.dataset: Any = dataset
        self.generator: DreamDatasetGenerator | NightmareDatasetGenerator = generator
        self.phase: str = phase
        self._cached_strengths: Optional[list[float]] = None
        self._cached_strengths_len: Optional[int] = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        pixel_values = item["pixel_values"]
        labels = item["labels"]

        from nightmarenet.distortions.registry import get_vision_registry

        registry = get_vision_registry()

        # Retrieve engines matching the current phase
        engines: list[str] = []
        vision_config = self.generator.config.get("vision", {})
        if vision_config:
            for name, prob in vision_config.items():
                import random

                meta = registry.get_engine_metadata(name)
                if meta.get("phase") == self.phase and random.random() < prob:
                    engines.append(name)
        else:
            for name in registry.engine_names:
                meta = registry.get_engine_metadata(name)
                if meta.get("phase") == self.phase:
                    engines.append(name)

        actual_strength = self.generator.strength
        sched = getattr(self.generator, "strength_schedule", "uniform")
        if self.phase == "nightmare" and sched != "uniform":
            ds_len = len(self.dataset)
            if self._cached_strengths is None or self._cached_strengths_len != ds_len:
                if hasattr(self.generator, "_compute_strengths"):
                    self._cached_strengths = self.generator._compute_strengths(ds_len)
                else:
                    self._cached_strengths = [actual_strength] * ds_len
                self._cached_strengths_len = ds_len
            if idx < len(self._cached_strengths):
                actual_strength = self._cached_strengths[idx]

        distorted = pixel_values
        for name in engines:
            if name in ["vision_fgsm", "vision_pgd"]:
                fn = registry._engines.get(name)
                if fn is not None and hasattr(fn, "__self__"):
                    fn.__self__.model = getattr(self.generator, "target_model", None)
            distorted = registry.apply(
                name, distorted, strength=actual_strength, seed=self.generator.seed
            )

        return {"pixel_values": distorted, "labels": labels}
