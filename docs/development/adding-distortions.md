# Adding a Distortion Engine

This guide walks through adding a **built-in (in-tree) distortion engine** to the NightmareNet core. If you instead want to ship a distortion as a separately-installed package or a single-file/decorator plugin, see the [Plugin Development Guide](../plugin_development.md) — this document intentionally focuses on the in-repo workflow and links out to avoid duplication.

## The contract

Every distortion — built-in or plugin — implements the same signature, defined in [`nightmarenet/distortions/registry.py`](../../nightmarenet/distortions/registry.py):

```python
from typing import Callable, Optional

DistortionFn = Callable[[str, float, Optional[int]], str]
```

That is: take `text`, a `strength` in `[0.0, 1.0]`, and an optional `seed`; return the distorted string. The behavioral contract (from [`nightmarenet/distortions/base.py`](../../nightmarenet/distortions/base.py)) is:

- `strength=0.0` should be approximately a no-op (return the text unchanged).
- `strength=1.0` should produce maximum distortion.
- The same `(text, strength, seed)` must produce identical output (determinism).
- Empty input must return an empty string without raising.
- The return type must be `str`.

## Two ways to implement an engine

### Option A — a plain function

The registry accepts any callable matching `DistortionFn`. This is the simplest form for a built-in engine.

```python
# nightmarenet/distortions/your_engine.py
import random
from typing import Optional


def distort(text: str, strength: float, seed: Optional[int] = None) -> str:
    """Swap adjacent characters with probability proportional to strength."""
    if not text:
        return ""
    rng = random.Random(seed)
    chars = list(text)
    for i in range(len(chars) - 1):
        if rng.random() < strength:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)
```

### Option B — a `BaseDistortion` subclass

For engines that carry metadata (`name`, `phase`, `description`), subclass `BaseDistortion` from [`nightmarenet/distortions/base.py`](../../nightmarenet/distortions/base.py):

```python
from typing import Optional

from nightmarenet.distortions.base import BaseDistortion


class CharSwap(BaseDistortion):
    name = "char_swap"
    phase = "nightmare"          # "dream", "nightmare", or "custom"
    description = "Swap adjacent characters"

    def distort(self, text: str, strength: float, seed: Optional[int] = None) -> str:
        ...
```

`BaseDistortion.validate()` returns `True` when `name` is set, and the abstract `distort` method enforces the signature.

## Registering the engine

Built-in engines are registered in `DistortionRegistry._register_builtins()` inside [`nightmarenet/distortions/registry.py`](../../nightmarenet/distortions/registry.py), alongside the existing `dream`, `nightmare`, and `keyboard_typo` engines:

```python
def _register_builtins(self) -> None:
    from nightmarenet.distortions import your_engine

    self.register(
        "char_swap",
        your_engine.distort,
        metadata={
            "phase": "nightmare",
            "description": "Swap adjacent characters",
            "source": "builtin",
        },
    )
```

The registry is a lazy singleton — retrieve it with `get_registry()` and apply an engine by name:

```python
from nightmarenet.distortions.registry import get_registry

registry = get_registry()
result = registry.apply("char_swap", "hello world", strength=0.5, seed=42)
print(registry.engine_names)          # sorted list of registered engines
print(registry.list_engines_by_source())  # grouped into builtin / plugin / custom
```

Relevant `DistortionRegistry` methods:

| Method | Description |
|---|---|
| `register(name, fn, metadata=None)` | Register an engine (raises `TypeError` if `fn` is not callable). |
| `register_decorator(name, phase="custom", description="")` | Decorator form of `register`. |
| `apply(name, text, strength=0.3, seed=None, **kwargs)` | Apply an engine; extra kwargs are forwarded only if the function accepts them. |
| `list_engines()` / `list_engines_by_source()` | Enumerate engines with metadata. |
| `unregister(name)` | Remove an engine. |
| `engine_names` | Property — sorted engine names. |

> [!NOTE]
> Vision distortions use a parallel `VisionDistortionRegistry` (retrieved via `get_vision_registry()`) with the signature `(image: torch.Tensor, strength, seed) -> torch.Tensor` and the entry-point group `nightmarenet.distortions.vision`.

## Validating your engine

Use the helpers in [`nightmarenet/distortions/testing.py`](../../nightmarenet/distortions/testing.py) to check the contract before writing full tests:

```python
from nightmarenet.distortions.testing import (
    validate_distortion_function,
    validate_distortion_plugin,
)

# For a plain function
failures = validate_distortion_function(distort)

# For a BaseDistortion subclass
failures = validate_distortion_plugin(CharSwap)

assert not failures, failures
```

These delegate to `validate_distortion_contract` and `validate_base_distortion` in [`nightmarenet/distortions/validators.py`](../../nightmarenet/distortions/validators.py), which check determinism, the strength-0 no-op, empty input, and type correctness.

## Testing

Add tests under `tests/` (mirroring the package layout). At minimum cover:

1. **Determinism** — same `(text, strength, seed)` produces the same output.
2. **Strength 0** — approximately a no-op.
3. **Strength 1** — produces a measurable change.
4. **Empty input** — returns `""` without raising.
5. **Registry round-trip** — `get_registry().apply("char_swap", ...)` matches calling the function directly.

```python
from nightmarenet.distortions.registry import get_registry
from nightmarenet.distortions.your_engine import distort


def test_determinism():
    assert distort("hello", 0.5, seed=42) == distort("hello", 0.5, seed=42)


def test_strength_zero_is_noop():
    assert distort("hello", 0.0, seed=42) == "hello"


def test_empty_input():
    assert distort("", 0.5, seed=42) == ""


def test_registry_round_trip():
    registry = get_registry()
    assert registry.apply("char_swap", "hello", strength=0.5, seed=42) == distort(
        "hello", 0.5, seed=42
    )
```

Run them with `pytest tests/test_your_engine.py` (see the [Testing Guide](testing.md)).

## Documentation

Per `CONTRIBUTING.md`, update the README distortion table when adding an engine, and if your distortion implements a published adversarial attack, cite the paper in the module docstring and in `docs/research/paper-draft.md`.

## Related Documentation

- [Plugin Development Guide](../plugin_development.md) — entry-point, decorator, and file-based plugins.
- [Testing Guide](testing.md) — running and writing tests.
- [Architecture](architecture.md) — where distortions sit in the pipeline.
