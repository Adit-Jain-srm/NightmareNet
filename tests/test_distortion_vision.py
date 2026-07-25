import pytest
import torch
from PIL import Image

from nightmarenet.distortions.registry import get_vision_registry
from nightmarenet.distortions.vision.dream import (
    ColorJitter,
    GaussianBlur,
    GeometricTransform,
    JPEGCompression,
)
from nightmarenet.distortions.vision.gaussian_noise import GaussianNoise
from nightmarenet.distortions.vision.nightmare import (
    FGSM,
    PGD,
    AdversarialPatch,
    PixelPerturbation,
)
from nightmarenet.distortions.vision.utils import to_pil, to_tensor


def test_vision_registry_singleton():
    """Test that get_vision_registry returns a singleton."""
    r1 = get_vision_registry()
    r2 = get_vision_registry()
    assert r1 is r2


def test_vision_registry_builtin_loaded():
    """Test that builtin gaussian noise is loaded."""
    registry = get_vision_registry()
    assert "vision_gaussian_noise" in registry


def test_gaussian_noise_no_op():
    """Test that strength 0.0 is a no-op."""
    noise = GaussianNoise()
    img = torch.zeros(3, 32, 32)
    distorted = noise.distort(img, strength=0.0)
    assert torch.all(img == distorted)


def test_gaussian_noise_determinism():
    """Test that the same seed produces the same output."""
    noise = GaussianNoise()
    img = torch.zeros(3, 32, 32)

    distorted1 = noise.distort(img, strength=0.5, seed=42)
    distorted2 = noise.distort(img, strength=0.5, seed=42)

    assert torch.all(distorted1 == distorted2)


def test_gaussian_noise_different_seeds():
    """Test that different seeds produce different outputs."""
    noise = GaussianNoise()
    img = torch.zeros(3, 32, 32)

    distorted1 = noise.distort(img, strength=0.5, seed=42)
    distorted2 = noise.distort(img, strength=0.5, seed=43)

    assert not torch.all(distorted1 == distorted2)


def test_gaussian_noise_bounds():
    """Test that gaussian noise output is bounded between 0 and 1."""
    noise = GaussianNoise()
    # image full of 0.5
    img = torch.ones(3, 32, 32) * 0.5
    distorted = noise.distort(img, strength=1.0)

    assert torch.all(distorted >= 0.0)
    assert torch.all(distorted <= 1.0)


def test_utils_conversions():
    """Test PIL to Tensor and Tensor to PIL conversions."""
    # Create a random RGB image
    img = Image.new("RGB", (32, 32), color="red")

    tensor = to_tensor(img)
    assert tensor.shape == (3, 32, 32)
    assert isinstance(tensor, torch.Tensor)

    img_back = to_pil(tensor)
    assert isinstance(img_back, Image.Image)
    assert img_back.size == (32, 32)


def test_vision_registry_apply():
    """Test applying a vision distortion via the registry."""
    registry = get_vision_registry()
    img = torch.ones(3, 32, 32) * 0.5

    distorted = registry.apply("vision_gaussian_noise", img, strength=0.5, seed=123)
    assert distorted.shape == (3, 32, 32)
    assert torch.all(distorted >= 0.0) and torch.all(distorted <= 1.0)


def test_gaussian_noise_empty_tensor():
    """Test empty tensor handling (e.g. shape with 0 dimension)."""
    noise = GaussianNoise()
    img = torch.zeros(3, 0, 0)
    distorted = noise.distort(img, strength=0.5, seed=42)
    assert distorted.shape == (3, 0, 0)


@pytest.mark.parametrize(
    "distortion_cls,name",
    [
        (ColorJitter, "vision_color_jitter"),
        (GeometricTransform, "vision_geometric_transform"),
        (GaussianBlur, "vision_gaussian_blur"),
        (JPEGCompression, "vision_jpeg_compression"),
    ],
)
def test_dream_distortions_properties(distortion_cls, name):
    """Test standard distortion properties: registered, no-op, bounds, and determinism."""
    # 1. Test Registry integration
    registry = get_vision_registry()
    assert name in registry

    distortion = distortion_cls()
    img = torch.ones(3, 32, 32) * 0.5

    # 2. Test strength 0.0 is no-op
    distorted_noop = distortion.distort(img, strength=0.0)
    assert torch.all(img == distorted_noop)

    # 3. Test bounds [0, 1] at max strength 1.0
    distorted_max = distortion.distort(img, strength=1.0)
    assert distorted_max.shape == (3, 32, 32)
    assert torch.all(distorted_max >= 0.0)
    assert torch.all(distorted_max <= 1.0)

    # 4. Test determinism with seed
    distorted_seed1 = distortion.distort(img, strength=0.8, seed=42)
    distorted_seed2 = distortion.distort(img, strength=0.8, seed=42)
    assert torch.all(distorted_seed1 == distorted_seed2)


def test_dream_distortions_single_channel():
    """Test that all dream distortions handle single channel grayscale images correctly."""
    img = torch.ones(1, 32, 32) * 0.5

    for distortion_cls in [ColorJitter, GeometricTransform, GaussianBlur, JPEGCompression]:
        distortion = distortion_cls()
        distorted = distortion.distort(img, strength=0.5, seed=42)
        assert distorted.shape == (1, 32, 32)
        assert torch.all(distorted >= 0.0) and torch.all(distorted <= 1.0)


class MockClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 32 * 32, 2)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.linear(x_flat)


def test_nightmare_distortions_registration():
    """Test that all nightmare distortions are loaded in the registry."""
    registry = get_vision_registry()
    expected_names = [
        "vision_pixel_perturbation",
        "vision_fgsm",
        "vision_pgd",
        "vision_adversarial_patch",
    ]
    for name in expected_names:
        assert name in registry


def test_pixel_perturbation_properties():
    """Test basic properties of PixelPerturbation."""
    distortion = PixelPerturbation(epsilon=0.1)
    img = torch.ones(3, 32, 32) * 0.5

    # Strength 0.0 is no-op
    assert torch.all(distortion.distort(img, strength=0.0) == img)

    # Determinism
    res1 = distortion.distort(img, strength=0.8, seed=42)
    res2 = distortion.distort(img, strength=0.8, seed=42)
    assert torch.all(res1 == res2)

    # Different seeds produce different outputs
    res3 = distortion.distort(img, strength=0.8, seed=43)
    assert not torch.all(res1 == res3)

    # L-inf bounds respected: perturbed - img <= eps
    # eps = strength * epsilon = 0.8 * 0.1 = 0.08
    diff = torch.abs(res1 - img)
    assert torch.all(diff <= 0.080001)  # allow tiny float tolerance
    assert torch.all(res1 >= 0.0) and torch.all(res1 <= 1.0)


def test_adversarial_patch_properties():
    """Test properties of AdversarialPatch."""
    distortion = AdversarialPatch(patch_size=8)
    img = torch.ones(3, 32, 32) * 0.5

    # Strength 0.0 is no-op
    assert torch.all(distortion.distort(img, strength=0.0) == img)

    # Determinism
    res1 = distortion.distort(img, strength=0.8, seed=42)
    res2 = distortion.distort(img, strength=0.8, seed=42)
    assert torch.all(res1 == res2)

    # Check bounds
    assert torch.all(res1 >= 0.0) and torch.all(res1 <= 1.0)

    # Grayscale handling
    gray_img = torch.ones(1, 32, 32) * 0.5
    res_gray = distortion.distort(gray_img, strength=0.8, seed=42)
    assert res_gray.shape == (1, 32, 32)


def test_adversarial_patch_oversized():
    """Test that patch sizes larger than the image are clamped correctly."""
    distortion = AdversarialPatch(patch_size=64)
    img = torch.ones(3, 32, 32) * 0.5
    res = distortion.distort(img, strength=1.0, seed=42)
    # Since patch size is clamped to 32x32, the entire image becomes noise
    assert res.shape == (3, 32, 32)


def test_gradient_attacks_fallback():
    """Test that FGSM and PGD fall back to PixelPerturbation if no model is provided."""
    img = torch.ones(3, 32, 32) * 0.5

    fgsm = FGSM(model=None, epsilon=0.1)
    pgd = PGD(model=None, epsilon=0.1)
    fallback = PixelPerturbation(epsilon=0.1)

    fgsm_res = fgsm.distort(img, strength=0.5, seed=42)
    pgd_res = pgd.distort(img, strength=0.5, seed=42)
    fallback_res = fallback.distort(img, strength=0.5, seed=42)

    assert torch.all(fgsm_res == fallback_res)
    assert torch.all(pgd_res == fallback_res)


def test_gradient_attacks_with_model():
    """Test FGSM and PGD execution with a mock model."""
    model = MockClassifier()
    # Set to training mode initially to verify restoration
    model.train()

    img = torch.ones(3, 32, 32) * 0.5
    fgsm = FGSM(model=model, epsilon=0.1)
    pgd = PGD(model=model, epsilon=0.1, steps=5)

    fgsm_res = fgsm.distort(img, strength=1.0)
    pgd_res = pgd.distort(img, strength=1.0)

    # Output shapes
    assert fgsm_res.shape == (3, 32, 32)
    assert pgd_res.shape == (3, 32, 32)

    # Output values bounded
    assert torch.all(fgsm_res >= 0.0) and torch.all(fgsm_res <= 1.0)
    assert torch.all(pgd_res >= 0.0) and torch.all(pgd_res <= 1.0)

    # L-inf limits respected (at strength 1.0, eps = 0.1)
    assert torch.all(torch.abs(fgsm_res - img) <= 0.10001)
    assert torch.all(torch.abs(pgd_res - img) <= 0.10001)

    # Attacks should modify the image
    assert not torch.all(fgsm_res == img)
    assert not torch.all(pgd_res == img)

    # Verify model mode is restored
    assert model.training
