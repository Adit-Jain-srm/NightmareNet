"""HuggingFace Hub integration for NightmareNet models.

Provides ``NightmareNetHubMixin`` — a drop-in mixin that adds
``push_to_hub`` / ``from_pretrained`` to any ``torch.nn.Module``.

Usage::

    from nightmarenet.hub import NightmareNetHubMixin

    class MyModel(NightmareNetHubMixin, torch.nn.Module):
        ...

    model = MyModel(...)
    model.push_to_hub("user/my-model")
    loaded = MyModel.from_pretrained("user/my-model")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml

try:
    from huggingface_hub import PyTorchModelHubMixin

    _HF_AVAILABLE = True
except ImportError:
    PyTorchModelHubMixin = object  # type: ignore[assignment,misc]
    _HF_AVAILABLE = False

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "nightmarenet_config.json"


def _require_hf_hub() -> None:
    """Raise a clear error when *huggingface_hub* is not installed."""
    if not _HF_AVAILABLE:
        raise ImportError(
            "The 'huggingface-hub' package is required for Hub integration. "
            "Install it with:  pip install nightmarenet[hub]"
        )


class NightmareNetHubMixin(PyTorchModelHubMixin):  # type: ignore[misc]
    """Mixin that adds Hub push/pull with NightmareNet metadata.

    Subclasses **must** be ``torch.nn.Module`` subclasses.  The mixin
    delegates weight serialisation to ``state_dict()`` / ``load_state_dict()``
    and stores NightmareNet-specific config (training params, robustness
    scores, distortion families) alongside the weights.

    Override ``get_nightmarenet_config()`` to attach custom metadata.
    """

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_nightmarenet_config(self) -> Dict[str, Any]:
        """Return NightmareNet-specific metadata to persist.

        Override in subclasses to include training history, robustness
        scores, distortion families, etc.  The base implementation
        returns an empty dict.
        """
        return {}

    # ------------------------------------------------------------------
    # PyTorchModelHubMixin interface
    # ------------------------------------------------------------------

    def _save_pretrained(self, save_directory: Union[str, Path]) -> None:  # type: ignore[override]
        """Save model weights **and** NightmareNet config to *save_directory*.

        This is called by ``PyTorchModelHubMixin.push_to_hub`` and by
        ``save_pretrained``.  We intentionally do **not** call
        ``super()._save_pretrained`` because the base implementation
        expects a very specific file layout that conflicts with our
        config serialisation.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # 1. Save PyTorch weights
        weights_path = save_directory / "pytorch_model.bin"
        torch.save(self.state_dict(), weights_path)
        logger.info("Saved model weights to %s", weights_path)

        # 2. Save NightmareNet config as JSON (Hub-compatible)
        config = self.get_nightmarenet_config()
        config_path = save_directory / _CONFIG_FILENAME
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved NightmareNet config to %s", config_path)

    @classmethod
    def _from_pretrained(  # type: ignore[override]
        cls,
        *,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs: Any,
    ) -> "NightmareNetHubMixin":
        """Load a NightmareNet model from the Hub or a local directory.

        Steps:
        1. Resolve the model directory (Hub download or local path).
        2. Load ``nightmarenet_config.json`` if present.
        3. Instantiate the class with ``**model_kwargs`` + config values.
        4. Load ``pytorch_model.bin`` into the instance.
        """
        _require_hf_hub()
        from huggingface_hub import hf_hub_download

        # Resolve local vs Hub
        if os.path.isdir(model_id):
            model_dir = Path(model_id)
        else:
            model_dir = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=_CONFIG_FILENAME,
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                )
            ).parent

        # Load config
        config: Dict[str, Any] = {}
        config_path = model_dir / _CONFIG_FILENAME
        if config_path.exists():
            with open(config_path, encoding="utf-8") as fh:
                config = json.load(fh)

        # Merge config into model_kwargs (explicit kwargs take precedence)
        merged = {**config, **model_kwargs}

        # Instantiate
        instance = cls(**merged)  # type: ignore[call-arg]

        # Load weights
        weights_path = model_dir / "pytorch_model.bin"
        if not weights_path.exists():
            # Fallback: try HuggingFace's safetensors convention
            weights_path = model_dir / "model.safetensors"
            if weights_path.exists():
                from safetensors.torch import load_file

                state = load_file(str(weights_path))
            else:
                raise FileNotFoundError(
                    f"No model weights found in {model_dir}. "
                    f"Expected pytorch_model.bin or model.safetensors."
                )
        else:
            state = torch.load(weights_path, map_location=map_location)

        instance.load_state_dict(state, strict=strict)
        logger.info("Loaded NightmareNet model from %s", model_dir)
        return instance

    # ------------------------------------------------------------------
    # Convenience wrappers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def push_to_hub(self, repo_id: str, **kwargs: Any) -> Any:
        """Push model to HuggingFace Hub.

        Delegates to ``PyTorchModelHubMixin.push_to_hub`` after ensuring
        ``_save_pretrained`` serialises NightmareNet metadata.
        """
        _require_hf_hub()
        return super().push_to_hub(repo_id, **kwargs)  # type: ignore[misc]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "NightmareNetHubMixin":
        """Load a pretrained model from the Hub or local path."""
        _require_hf_hub()
        return super().from_pretrained(  # type: ignore[misc]
            pretrained_model_name_or_path, **kwargs
        )


__all__ = ["NightmareNetHubMixin", "_HF_AVAILABLE"]
