"""Dream-phase vision distortion engines."""

import io
import math
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from nightmarenet.distortions.vision.base import ImageDistortion
from nightmarenet.distortions.vision.utils import to_pil, to_tensor


class ColorJitter(ImageDistortion):
    """Randomly adjusts brightness, contrast, saturation, and hue of an image."""

    name = "vision_color_jitter"
    phase = "dream"
    description = "Random brightness, contrast, saturation, and hue shifts scaled by strength"

    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image.clone()

        # Input must be in [0, 1]
        distorted = image.clone()

        # Seeded generator
        gen = None
        if seed is not None:
            gen = torch.Generator(device=image.device)
            gen.manual_seed(seed)

        # Helper to get uniform random value in [min_val, max_val]
        def rand_uniform(min_val: float, max_val: float) -> float:
            if gen is not None:
                val = torch.rand(1, generator=gen, device=image.device).item()
            else:
                val = torch.rand(1, device=image.device).item()
            return min_val + val * (max_val - min_val)

        # 1. Brightness adjustment
        # Factor in [max(0.0, 1.0 - strength), 1.0 + strength]
        b_factor = rand_uniform(max(0.0, 1.0 - strength), 1.0 + strength)
        distorted = distorted * b_factor

        # 2. Contrast adjustment
        # Factor in [max(0.0, 1.0 - strength), 1.0 + strength]
        c_factor = rand_uniform(max(0.0, 1.0 - strength), 1.0 + strength)
        # Using simple mean across spatial dimensions
        mean = distorted.mean(dim=(-2, -1), keepdim=True)
        distorted = (distorted - mean) * c_factor + mean

        # 3. Saturation (only for 3-channel RGB images)
        if distorted.shape[0] == 3:
            s_factor = rand_uniform(max(0.0, 1.0 - strength), 1.0 + strength)
            # Grayscale using standard luma weights
            luma = (0.299 * distorted[0] + 0.587 * distorted[1] + 0.114 * distorted[2]).unsqueeze(0)
            distorted = luma + s_factor * (distorted - luma)

        # 4. Hue shift (only for 3-channel RGB images)
        if distorted.shape[0] == 3:
            # Hue shift in radians: [-strength * 0.5, strength * 0.5]
            hue_shift = rand_uniform(-strength * 0.5, strength * 0.5)

            r, g, b = distorted[0], distorted[1], distorted[2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.14713 * r - 0.28886 * g + 0.436 * b
            v = 0.615 * r - 0.51499 * g - 0.10001 * b

            cos_a = math.cos(hue_shift)
            sin_a = math.sin(hue_shift)

            u_new = u * cos_a - v * sin_a
            v_new = u * sin_a + v * cos_a

            r_new = y + 1.13983 * v_new
            g_new = y - 0.39465 * u_new - 0.58060 * v_new
            b_new = y + 2.03211 * u_new

            distorted = torch.stack([r_new, g_new, b_new], dim=0)

        return torch.clamp(distorted, 0.0, 1.0)


class GeometricTransform(ImageDistortion):
    """Applies random rotation, translation, and scale transformations to an image."""

    name = "vision_geometric_transform"
    phase = "dream"
    description = "Random rotation, translation, and scale scaled by strength"

    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image.clone()

        # Seeded generator
        gen = None
        if seed is not None:
            gen = torch.Generator(device=image.device)
            gen.manual_seed(seed)

        # Helper to get uniform random value in [min_val, max_val]
        def rand_uniform(min_val: float, max_val: float) -> float:
            if gen is not None:
                val = torch.rand(1, generator=gen, device=image.device).item()
            else:
                val = torch.rand(1, device=image.device).item()
            return min_val + val * (max_val - min_val)

        # Rotation angle in radians: max 30 degrees (pi/6) at strength 1.0
        max_angle = strength * (math.pi / 6.0)
        angle = rand_uniform(-max_angle, max_angle)

        # Translation: max 20% shift at strength 1.0
        max_trans = strength * 0.2
        tx = rand_uniform(-max_trans, max_trans)
        ty = rand_uniform(-max_trans, max_trans)

        # Scale: max 20% zoom in/out at strength 1.0
        max_scale = strength * 0.2
        scale = rand_uniform(1.0 - max_scale, 1.0 + max_scale)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # PyTorch affine grid theta matrix (1, 2, 3)
        # T = [scale * cos(a), -scale * sin(a), tx]
        #     [scale * sin(a),  scale * cos(a), ty]
        theta = torch.tensor(
            [
                [
                    [scale * cos_a, -scale * sin_a, tx],
                    [scale * sin_a, scale * cos_a, ty],
                ]
            ],
            dtype=image.dtype,
            device=image.device,
        )

        # Reshape to (1, C, H, W) for grid_sample
        x = image.unsqueeze(0)

        # Create affine grid and sample image
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        distorted = F.grid_sample(x, grid, align_corners=False).squeeze(0)

        return torch.clamp(distorted, 0.0, 1.0)


class GaussianBlur(ImageDistortion):
    """Applies Gaussian blur to an image with kernel size scaled by strength."""

    name = "vision_gaussian_blur"
    phase = "dream"
    description = "Gaussian blur with kernel size scaled by strength"

    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image.clone()

        # Kernel size must be an odd integer
        # Max radius is 5 (kernel size 11) at strength 1.0
        max_radius = 5
        radius = int(strength * max_radius)
        if radius < 1:
            radius = 1
        kernel_size = 2 * radius + 1

        # Sigma: max 3.0 at strength 1.0
        sigma = max(strength * 3.0, 1e-5)

        # Construct 1D Gaussian kernel
        x = torch.arange(
            -kernel_size // 2 + 1,
            kernel_size // 2 + 1,
            dtype=torch.float32,
            device=image.device,
        )
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 2D Gaussian kernel
        kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)

        # Prepare image: unsqueeze to (1, C, H, W)
        x_img = image.unsqueeze(0)
        channels = image.shape[0]

        # Reshape 2D kernel for depthwise conv2d: (channels, 1, K_H, K_W)
        kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size).to(dtype=image.dtype)

        padding = kernel_size // 2
        distorted = F.conv2d(
            x_img,
            kernel_2d,
            padding=padding,
            groups=channels,
        ).squeeze(0)

        return torch.clamp(distorted, 0.0, 1.0)


class JPEGCompression(ImageDistortion):
    """Simulates JPEG compression artifacts on an image."""

    name = "vision_jpeg_compression"
    phase = "dream"
    description = "Simulates JPEG compression artifacts scaled by strength"

    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image.clone()

        # Quality ranges from 100 down to 20 at strength 1.0
        quality = int(100 - (strength * 80))
        quality = max(1, min(100, quality))

        # Convert tensor to PIL image
        pil_img = to_pil(image)

        # Save to buffer with JPEG format and given quality
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)

        # Load back
        compressed_pil = Image.open(buf)

        # Convert back to tensor
        distorted = to_tensor(compressed_pil).to(device=image.device, dtype=image.dtype)

        # Ensure correct channel dimension (e.g. if conversion dropped channels)
        if distorted.shape[0] != image.shape[0]:
            if distorted.shape[0] == 1 and image.shape[0] == 3:
                # Grayscale to RGB
                distorted = distorted.repeat(3, 1, 1)
            elif distorted.shape[0] == 3 and image.shape[0] == 1:
                # RGB to Grayscale
                distorted = (
                    0.299 * distorted[0] + 0.587 * distorted[1] + 0.114 * distorted[2]
                ).unsqueeze(0)

        return torch.clamp(distorted, 0.0, 1.0)
