"""Property-based fuzz tests for distortion engine determinism and contracts.

Verifies Paper Section 4.4 claim: "All distortion operations are deterministic
given (text, strength, seed)." Tests run against every engine registered in the
default registry so new engines are covered automatically.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import nightmarenet.distortions.adversarial as _adv_mod
from nightmarenet.distortions.registry import get_registry

pytestmark = pytest.mark.slow


class _NoOpLearned:
    """Drop-in replacement for LearnedAdversarialGenerator that does no inference."""

    def generate(self, text: str, strength: float = 0.3) -> str:  # noqa: ARG002
        return text


@pytest.fixture(autouse=True)
def _patch_learned_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the learned-model cache with a no-op so fuzz tests need no GPU/network."""
    monkeypatch.setattr(_adv_mod, "_LEARNED_CACHE", {"distilbert-base-uncased": _NoOpLearned()})


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Printable ASCII + common Unicode blocks (Latin extended, CJK, Arabic, emoji)
_unicode_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        whitelist_characters="\n\t",
    ),
    min_size=0,
    max_size=300,
)

_strength = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_seed = st.integers(min_value=0, max_value=2**31 - 1)

# Parametrize over engine names at collection time so failures name the engine.
_ENGINE_NAMES = get_registry().engine_names


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
@settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(text=_unicode_text, strength=_strength, seed=_seed)
def test_determinism(engine_name: str, text: str, strength: float, seed: int) -> None:
    """Same (text, strength, seed) must always produce the same output."""
    registry = get_registry()
    first = registry.apply(engine_name, text, strength=strength, seed=seed)
    second = registry.apply(engine_name, text, strength=strength, seed=seed)
    assert first == second, (
        f"Engine '{engine_name}' is non-deterministic: "
        f"got different outputs for seed={seed}, strength={strength!r}, text={text!r}"
    )


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
@settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(strength=_strength, seed=_seed)
def test_empty_input_returns_empty(engine_name: str, strength: float, seed: int) -> None:
    """Empty string input must return empty string without raising."""
    result = get_registry().apply(engine_name, "", strength=strength, seed=seed)
    assert result == "", (
        f"Engine '{engine_name}' returned {result!r} for empty input "
        f"(strength={strength!r}, seed={seed})"
    )


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(text=_unicode_text, seed=_seed)
def test_strength_boundaries_do_not_crash(engine_name: str, text: str, seed: int) -> None:
    """strength=0.0 and strength=1.0 must not raise for any input."""
    registry = get_registry()
    result_low = registry.apply(engine_name, text, strength=0.0, seed=seed)
    result_high = registry.apply(engine_name, text, strength=1.0, seed=seed)
    assert isinstance(result_low, str)
    assert isinstance(result_high, str)


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
@settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    text=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=1,
        max_size=200,
    ),
    strength=_strength,
    seed=_seed,
)
def test_unicode_does_not_crash(engine_name: str, text: str, strength: float, seed: int) -> None:
    """Arbitrary Unicode text must not raise and must return a string."""
    result = get_registry().apply(engine_name, text, strength=strength, seed=seed)
    assert isinstance(result, str)


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(strength=_strength, seed=_seed)
def test_registry_roundtrip_matches_direct_call(
    engine_name: str, strength: float, seed: int
) -> None:
    """registry.apply(name, ...) must equal calling the engine function directly."""
    registry = get_registry()
    sample = "The quick brown fox jumps over the lazy dog."
    via_registry = registry.apply(engine_name, sample, strength=strength, seed=seed)
    direct = registry._engines[engine_name](sample, strength, seed)
    assert via_registry == direct
