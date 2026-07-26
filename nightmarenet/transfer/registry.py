"""Foundation model registry for robustness transfer learning.

Saves and loads adversarially hardened backbone representations.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

DEFAULT_FOUNDATION_DIR = Path(
    os.getenv("NIGHTMARENET_FOUNDATION_DIR", str(Path.home() / ".nightmarenet" / "foundation"))
)


class FoundationRegistry:
    """Manages the storage and retrieval of robust foundation models."""

    def __init__(self, cache_dir: Optional[str | Path] = None) -> None:
        if cache_dir is None:
            self.cache_dir = DEFAULT_FOUNDATION_DIR
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def register(
        self, model_path: str | Path, name: str, metadata: Optional[dict[str, Any]] = None
    ) -> Path:
        """Register a fine-tuned model as a foundation backbone.

        This extracts ONLY the base model (no classification head) and saves it.

        Args:
            model_path: Path to the fully-trained model.
            name: Identifier for the foundation model.
            metadata: Optional dictionary with info like 'robustness_score', 'training_task'.

        Returns:
            Path to the saved foundation model.
        """
        model_path_str = str(model_path)
        dest_path = self.cache_dir / name

        if dest_path.exists():
            logger.warning("Foundation model '%s' already exists. Overwriting.", name)

        dest_path.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting backbone from %s", model_path_str)
        # Using AutoModel directly loads the backbone without task-specific heads.
        backbone = AutoModel.from_pretrained(model_path_str)
        tokenizer = AutoTokenizer.from_pretrained(model_path_str)

        backbone.save_pretrained(dest_path)
        tokenizer.save_pretrained(dest_path)

        # Save metadata
        if metadata is None:
            metadata = {}

        meta_path = dest_path / "nightmarenet_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Foundation model '%s' successfully registered at %s", name, dest_path)
        return dest_path

    def load(self, name: str) -> tuple[PreTrainedModel, PreTrainedTokenizer, dict[str, Any]]:
        """Load a foundation model by name.

        Args:
            name: Identifier for the foundation model.

        Returns:
            Tuple of (backbone, tokenizer, metadata).
        """
        model_path = self.cache_dir / name
        if not model_path.exists():
            raise FileNotFoundError(
                f"Foundation model '{name}' not found in registry at {model_path}."
            )

        logger.info("Loading foundation model '%s'", name)
        backbone = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        meta_path = model_path / "nightmarenet_meta.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        return backbone, tokenizer, metadata

    def list_models(self) -> list[str]:
        """List all registered foundation models."""
        if not self.cache_dir.exists():
            return []
        models = []
        for item in self.cache_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                models.append(item.name)
        return sorted(models)


_default_registry = None


def get_registry(cache_dir: Optional[str | Path] = None) -> FoundationRegistry:
    """Get the singleton registry instance."""
    global _default_registry
    if _default_registry is None or (
        cache_dir is not None and Path(cache_dir) != _default_registry.cache_dir
    ):
        _default_registry = FoundationRegistry(cache_dir)
    return _default_registry
