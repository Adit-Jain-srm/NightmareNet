### Assignment Request (L3 / Core Performance Optimization)
I would like to take on this performance fix. Below is my proposed technical implementation plan to resolve the VRAM leaks.

**Step-by-Step Implementation Plan:**
1. **Payload Inspection:** Review the `CallbackManager` logic to identify where model outputs (which are attached to the computation graph) are being passed to callbacks (like logging, metric computation, or visualization).
2. **Tensor Detachment:** Introduce a utility function that recursively traverses the payload dictionary/list and calls `.detach().cpu()` on any PyTorch tensors before they are handed off to the callbacks.
3. **Memory Profiling Test:** Add a unit test that runs a mini training loop with mock callbacks, measuring VRAM usage before and after to assert that memory does not monotonically increase over epochs.

**Files to be modified:**
- `nightmarenet/callbacks/manager.py` (or equivalent callback handler)
- `nightmarenet/utils/tensor_utils.py` (New: utility for recursive detach)
- `tests/test_callbacks_memory.py`

I'll ensure all code passes `ruff` and `mypy` as required by the contribution rules. Let me know if I can start!