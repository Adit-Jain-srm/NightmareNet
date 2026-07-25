"""Utilities for vision distortions."""

import torch
import torchvision.transforms.functional as functional
from PIL import Image


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a PyTorch tensor (C, H, W) in [0, 1]."""
    return functional.to_tensor(img)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a PyTorch tensor (C, H, W) in [0, 1] to a PIL Image."""
    return functional.to_pil_image(tensor)
