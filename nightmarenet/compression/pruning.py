"""Model compression utilities: pruning and bottleneck.

Forces information retention efficiency by removing low-magnitude weights
or applying bottleneck layers that reduce the model's capacity.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MagnitudePruner:
    """Prunes weights below a magnitude percentile threshold.

    Zeros out the smallest weights in each layer's parameters,
    then optionally allows fine-tuning of the remaining weights.

    Args:
        pruning_ratio: Fraction of weights to prune (0–1). E.g., 0.2 prunes 20%.
    """

    def __init__(self, pruning_ratio: float = 0.2) -> None:
        if not 0.0 <= pruning_ratio < 1.0:
            raise ValueError(f"pruning_ratio must be in [0, 1), got {pruning_ratio}")
        self.pruning_ratio = pruning_ratio

    def apply(self, model: nn.Module) -> dict:
        """Apply magnitude-based pruning to the model.

        Args:
            model: PyTorch model to prune.

        Returns:
            Dict with pruning statistics (pruned_params, total_params, sparsity).
        """
        total_params = 0
        pruned_params = 0

        for name, param in model.named_parameters():
            if not param.requires_grad or param.dim() < 2:
                continue

            with torch.no_grad():
                # Compute the magnitude threshold for this parameter
                flat = param.abs().flatten()
                total_params += flat.numel()

                k = int(flat.numel() * self.pruning_ratio)
                if k == 0:
                    continue

                # Handle edge case where layer has all-zero weights
                if flat.max() == 0:
                    logger.info("Layer '%s': all-zero weights, skipping", name)
                    continue

                threshold = torch.kthvalue(flat, k).values

                # Create mask: 1 for weights above threshold, 0 for below
                mask = (param.abs() > threshold).float()
                param.mul_(mask)

                # Count actual zeros introduced (may differ from k due to ties)
                actual_pruned = int((mask == 0).sum().item())
                pruned_params += actual_pruned
                logger.debug(
                    "Layer '%s': pruned %d/%d params",
                    name,
                    actual_pruned,
                    flat.numel(),
                )

        sparsity = pruned_params / max(total_params, 1)
        logger.info(
            "Magnitude pruning complete: %d/%d params pruned (%.2f%% sparsity)",
            pruned_params,
            total_params,
            sparsity * 100,
        )

        return {
            "pruned_params": pruned_params,
            "total_params": total_params,
            "sparsity": sparsity,
        }


class BottleneckWrapper(nn.Module):
    """Wraps a transformer layer with a lower-rank projection.

    Inserts a bottleneck (down-project → up-project) around the original
    layer to force information compression through a narrower channel.

    Args:
        original_layer: The transformer layer to wrap.
        bottleneck_dim: Dimension of the bottleneck. If None, computed from rank_ratio.
        rank_ratio: Ratio of the bottleneck dim to the original hidden dim.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        bottleneck_dim: Optional[int] = None,
        rank_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError(f"rank_ratio must be in (0, 1], got {rank_ratio}")
        self.original_layer = original_layer

        # Infer hidden dimension from the layer
        hidden_dim = self._infer_hidden_dim(original_layer)
        if bottleneck_dim is None:
            bottleneck_dim = max(1, int(hidden_dim * rank_ratio))

        self.down_project = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_project = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        # Initialize near-identity
        nn.init.kaiming_uniform_(self.down_project.weight)
        nn.init.kaiming_uniform_(self.up_project.weight)

        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        logger.info(
            "BottleneckWrapper: %d → %d → %d",
            hidden_dim,
            bottleneck_dim,
            hidden_dim,
        )

    @staticmethod
    def _infer_hidden_dim(layer: nn.Module) -> int:
        """Try to infer the hidden dimension from the layer's parameters."""
        for name, param in layer.named_parameters():
            if param.dim() >= 2:
                return param.shape[-1]
        raise ValueError("Could not infer hidden dimension from layer parameters.")

    def forward(self, hidden_states, **kwargs):
        """Forward pass: compress through bottleneck, then pass to original layer."""
        compressed = self.down_project(hidden_states)
        expanded = self.up_project(compressed)
        return self.original_layer(expanded, **kwargs)


def apply_bottleneck_to_model(
    model: nn.Module,
    rank_ratio: float = 0.5,
    target_modules: Optional[list] = None,
) -> dict:
    """Apply bottleneck wrappers to selected layers of a model.

    Args:
        model: The model to modify.
        rank_ratio: Ratio of bottleneck dimension to original hidden dimension.
        target_modules: Optional list of module name patterns to target.
            Defaults to transformer attention/MLP layers.

    Returns:
        Dict with statistics about the applied bottlenecks.
    """
    if target_modules is None:
        target_modules = ["mlp", "attn"]

    wrapped_count = 0
    for name, module in list(model.named_children()):
        if any(target in name.lower() for target in target_modules):
            try:
                wrapper = BottleneckWrapper(module, rank_ratio=rank_ratio)
                setattr(model, name, wrapper)
                wrapped_count += 1
            except (ValueError, RuntimeError) as e:
                logger.warning("Could not wrap module '%s': %s", name, e)
        elif len(list(module.children())) > 0:
            # Recurse into child modules
            child_stats = apply_bottleneck_to_model(
                module, rank_ratio=rank_ratio, target_modules=target_modules
            )
            wrapped_count += child_stats.get("wrapped_count", 0)

    logger.info(
        "Bottleneck application complete: %d modules wrapped with rank_ratio=%.2f",
        wrapped_count,
        rank_ratio,
    )
    return {"wrapped_count": wrapped_count, "rank_ratio": rank_ratio}
