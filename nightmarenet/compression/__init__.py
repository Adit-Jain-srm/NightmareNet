"""Model compression utilities (pruning and bottleneck)."""

from nightmarenet.compression.pruning import (
    BottleneckWrapper as BottleneckWrapper,
)
from nightmarenet.compression.pruning import (
    MagnitudePruner as MagnitudePruner,
)
from nightmarenet.compression.pruning import (
    apply_bottleneck_to_model as apply_bottleneck_to_model,
)

__all__ = ["MagnitudePruner", "BottleneckWrapper", "apply_bottleneck_to_model"]
