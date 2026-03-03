# Causal Conv1d Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify causal_conv1d GPU Mode task to match trimul structure with minimal PyTorch reference

**Architecture:** Replace FLA Triton kernel reference with pure PyTorch `causal_conv1d_ref_torch`, reduce interface to 3-tuple `(x, weight, config)`, hardcode SiLU activation, remove all optional features (bias, residual, cu_seqlens).

**Tech Stack:** PyTorch (F.conv1d, F.silu), pytest, bfloat16

---

## Task 1: Update Type Definitions

**Files:**
- Modify: `k_search/tasks/gpu_mode/causal_conv1d/task.py:1-18`

**Step 1: Write failing test for new type signature**

Update `tests/test_gpu_mode_causal_conv1d.py` to expect 3-tuple:

```python
@pytest.mark.cuda
class TestGenerateInput:
    def test_generate_input_returns_three_tuple(self):
        """Verify generate_input returns (x, weight, config) tuple."""
        data = generate_input(B=2, T=64, D=32, W=4, seed=42)

        assert isinstance(data, tuple)
        assert len(data) == 3
        x, weight, config = data
        assert isinstance(x, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        assert isinstance(config, dict)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestGenerateInput::test_generate_input_returns_three_tuple -xvs`

Expected: FAIL - generate_input currently returns 5-tuple

**Step 3: Update task.py type definitions**

```python
from typing import TypeAlias, TypedDict
import torch

input_t: TypeAlias = tuple[torch.Tensor, torch.Tensor, dict[str, str]]
output_t: TypeAlias = torch.Tensor


class TestSpec(TypedDict):
    B: int
    T: int
    D: int
    W: int
    seed: int
```

**Step 4: Run test (will still fail until we update generate_input)**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestGenerateInput::test_generate_input_returns_three_tuple -xvs`

Expected: Still FAIL - generate_input not yet updated

**Step 5: Commit type definition changes**

```bash
git add k_search/tasks/gpu_mode/causal_conv1d/task.py tests/test_gpu_mode_causal_conv1d.py
git commit -m "refactor(causal_conv1d): simplify type definitions to 3-tuple

Update input_t to (x, weight, config) and remove bias/residual from TestSpec.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Implement Simplified generate_input

**Files:**
- Modify: `k_search/tasks/gpu_mode/causal_conv1d/reference.py:21-48`

**Step 1: Test already written in Task 1**

The test `test_generate_input_returns_three_tuple` validates the new signature.

**Step 2: Add test for shape and dtype**

```python
def test_generate_input_shapes_and_dtype(self):
    """Verify generate_input produces correct shapes and dtype."""
    B, T, D, W = 2, 128, 64, 4
    x, weight, config = generate_input(B=B, T=T, D=D, W=W, seed=42)

    assert x.shape == (B, T, D)
    assert weight.shape == (D, W)
    assert x.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    assert isinstance(config, dict)
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestGenerateInput::test_generate_input_shapes_and_dtype -xvs`

Expected: FAIL - generate_input not yet updated

**Step 4: Update generate_input implementation**

Replace existing `generate_input` in `reference.py`:

```python
def generate_input(
    B: int,
    T: int,
    D: int,
    W: int,
    seed: int,
) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    weight = torch.randn(D, W, device="cuda", dtype=torch.bfloat16, generator=gen)

    config = {}

    return (x, weight, config)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestGenerateInput -xvs`

Expected: PASS for both tests

**Step 6: Add test for seeded reproducibility**

```python
def test_generate_input_seeded(self):
    """Verify same seed produces same outputs."""
    data1 = generate_input(B=2, T=64, D=32, W=4, seed=123)
    data2 = generate_input(B=2, T=64, D=32, W=4, seed=123)

    assert torch.allclose(data1[0], data2[0])
    assert torch.allclose(data1[1], data2[1])
```

**Step 7: Run test to verify it passes**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestGenerateInput::test_generate_input_seeded -xvs`

Expected: PASS

**Step 8: Commit generate_input changes**

```bash
git add k_search/tasks/gpu_mode/causal_conv1d/reference.py tests/test_gpu_mode_causal_conv1d.py
git commit -m "refactor(causal_conv1d): simplify generate_input to 3-tuple

Remove activation, withbias, withresidual parameters. Generate only x and
weight tensors with empty config dict.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Implement causal_conv1d_ref_torch Reference

**Files:**
- Modify: `k_search/tasks/gpu_mode/causal_conv1d/reference.py:7-18`

**Step 1: Write test for reference kernel correctness**

Add to `tests/test_gpu_mode_causal_conv1d.py`:

```python
@pytest.mark.cuda
class TestReference:
    def test_ref_kernel_output_shape(self):
        """Verify ref_kernel produces correct output shape."""
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        output = ref_kernel(data)

        assert output.shape == (2, 128, 64)
        assert output.dtype == torch.bfloat16

    def test_ref_kernel_causal_property(self):
        """Verify output at time t only depends on inputs up to time t."""
        B, T, D, W = 1, 10, 4, 4
        x, weight, config = generate_input(B=B, T=T, D=D, W=W, seed=42)

        # Modify input after time t=5
        x_modified = x.clone()
        x_modified[:, 6:, :] = 0

        output_original = ref_kernel((x, weight, config))
        output_modified = ref_kernel((x_modified, weight, config))

        # Output at t=5 and earlier should be unchanged
        assert torch.allclose(output_original[:, :6, :], output_modified[:, :6, :])
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestReference -xvs`

Expected: FAIL - ref_kernel still using old FLA implementation

**Step 3: Implement causal_conv1d_ref_torch**

Replace `ref_kernel` function in `reference.py`:

```python
import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import make_match_reference


def causal_conv1d_ref_torch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Minimal causal 1D convolution reference with hardcoded SiLU activation.

    Args:
        x: [B, T, D] bfloat16
        weight: [D, W] bfloat16

    Returns:
        [B, T, D] bfloat16
    """
    B, T, D = x.shape
    W = weight.shape[1]

    # Transpose to [B, D, T] for conv1d
    x = x.transpose(1, 2)

    # Causal convolution with padding
    out = F.conv1d(x, weight.unsqueeze(1), bias=None, padding=W - 1, groups=D)
    out = out[..., :T]  # Trim to original length

    # SiLU activation
    out = F.silu(out)

    # Back to [B, T, D]
    out = out.transpose(1, 2)

    return out


def ref_kernel(data: input_t) -> output_t:
    x, weight, config = data
    return causal_conv1d_ref_torch(x, weight)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestReference -xvs`

Expected: PASS for both tests

**Step 5: Commit reference implementation**

```bash
git add k_search/tasks/gpu_mode/causal_conv1d/reference.py tests/test_gpu_mode_causal_conv1d.py
git commit -m "feat(causal_conv1d): add minimal PyTorch reference implementation

Replace FLA Triton kernel with pure PyTorch causal_conv1d_ref_torch.
Hardcode SiLU activation, remove bias/residual/state handling.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Update Baseline Submission

**Files:**
- Modify: `k_search/tasks/gpu_mode/causal_conv1d/submission.py:1-16`

**Step 1: Write test for submission correctness**

Add to `tests/test_gpu_mode_causal_conv1d.py`:

```python
@pytest.mark.cuda
class TestIntegration:
    def test_baseline_matches_reference(self):
        """Verify custom_kernel matches ref_kernel exactly."""
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)

        expected = ref_kernel(data)
        actual = custom_kernel(data)

        assert torch.allclose(expected, actual, rtol=2e-2, atol=2e-2)

    def test_baseline_output_shape_and_dtype(self):
        """Verify custom_kernel produces correct output."""
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        output = custom_kernel(data)

        assert output.shape == (2, 128, 64)
        assert output.dtype == torch.bfloat16
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestIntegration -xvs`

Expected: FAIL - custom_kernel still using old FLA implementation

**Step 3: Update custom_kernel implementation**

Replace `custom_kernel` in `submission.py`:

```python
from task import input_t, output_t
import torch
import torch.nn.functional as F


def custom_kernel(data: input_t) -> output_t:
    """
    Baseline causal conv1d implementation using PyTorch.

    Applies causal 1D convolution with SiLU activation.
    """
    x, weight, config = data

    B, T, D = x.shape
    W = weight.shape[1]

    # Transpose to [B, D, T] for conv1d
    x = x.transpose(1, 2)

    # Causal convolution
    out = F.conv1d(x, weight.unsqueeze(1), bias=None, padding=W - 1, groups=D)
    out = out[..., :T]

    # SiLU activation
    out = F.silu(out)

    # Back to [B, T, D]
    return out.transpose(1, 2)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestIntegration -xvs`

Expected: PASS for both tests

**Step 5: Commit submission changes**

```bash
git add k_search/tasks/gpu_mode/causal_conv1d/submission.py tests/test_gpu_mode_causal_conv1d.py
git commit -m "refactor(causal_conv1d): replace FLA with pure PyTorch baseline

Remove FLA dependency from custom_kernel. Use F.conv1d + F.silu matching
reference implementation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Update Spec Text

**Files:**
- Modify: `k_search/tasks/gpu_mode/causal_conv1d/spec.py:1-85`

**Step 1: Write test for spec text content**

Add to `tests/test_gpu_mode_causal_conv1d.py`:

```python
class TestSpec:
    def test_spec_text_loads(self):
        """Verify spec text loads and is non-empty."""
        assert isinstance(CAUSAL_CONV1D_SPEC_TEXT_TRITON, str)
        assert len(CAUSAL_CONV1D_SPEC_TEXT_TRITON) > 500

    def test_spec_contains_interface(self):
        """Verify spec documents the 3-tuple interface."""
        assert "custom_kernel(data)" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "(x, weight, config)" in CAUSAL_CONV1D_SPEC_TEXT_TRITON

    def test_spec_mentions_silu(self):
        """Verify spec documents SiLU activation."""
        assert "silu" in CAUSAL_CONV1D_SPEC_TEXT_TRITON.lower()

    def test_spec_has_test_case(self):
        """Verify spec contains test case constraints."""
        assert "B = 2" in CAUSAL_CONV1D_SPEC_TEXT_TRITON or "B=2" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "4096" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "2048" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
```

**Step 2: Run tests to verify some fail**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestSpec -xvs`

Expected: Some FAIL - spec still has old interface and FLA reference

**Step 3: Update spec.py**

Replace content in `spec.py`:

```python
"""Causal Conv1d spec text for prompting."""

from __future__ import annotations


CAUSAL_CONV1D_SPEC_TEXT_TRITON = """Causal Conv1d Forward (Triton submission)

Task:
- Optimize the *forward* pass of causal 1D convolution with SiLU activation.
- Primary goal: Analyze the shapes of the test cases and find the best hybrid Triton + PyTorch approach to achieve high performance.
  - **Fuse** operators to minimize large intermediate tensors, global-memory traffic, and kernel launch overhead.
  - Use a **mixed precision** strategy where appropriate. Only the returned tensor **must be bfloat16**.
  - You do **not** need to re-implement everything in Triton. You may choose to have some of the operations done in pytorch.
    - You need to experiment with **many different** hybrid Triton + PyTorch approaches to find the best one.
    - Analyze the shapes of the test cases to find the best hybrid approach.
- Include a short comment at the top summarizing your new implementation.
- For each round, you can see your current best solution and the previous round's summary, therefore you can implement the kernel step by step.

Data interface:
- Python/Triton: custom_kernel(data) where:
  data = (x, weight, config)
- Include the code inside '```' and '```' blocks.

Inputs:
- x: torch.Tensor, shape [B, T, D], dtype bfloat16
- weight: torch.Tensor, shape [D, W], dtype bfloat16
- config: Dict (currently empty, reserved for future use)

Output:
- torch.Tensor, shape [B, T, D], dtype bfloat16

Correctness:
- Must match reference within typical tolerances: rtol=2e-2, atol=2e-2.

**Problem Constraints:**
- B = 2, T = 4096, D = 2048, W = 4
- The input distribution will be sampled from a standard Normal distribution.

Test Cases for runtime (optimize runtime for these):
- {"B": 2, "T": 4096, "D": 2048, "W": 4}

Remarks:
- The convolution width W=4 is small; consider loop unrolling or direct computation.
- The sequence dimension T=4096 is large; consider tiling strategies.
- Memory access patterns matter: coalesce reads/writes along the D dimension.
- Causality constraint: output at time t depends only on inputs at times [max(0, t-W+1), ..., t].
- Fuse the SiLU activation with the convolution to avoid memory round-trips.
"""
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestSpec -xvs`

Expected: PASS for all spec tests

**Step 5: Commit spec changes**

```bash
git add k_search/tasks/gpu_mode/causal_conv1d/spec.py tests/test_gpu_mode_causal_conv1d.py
git commit -m "refactor(causal_conv1d): update spec to match trimul structure

Remove FLA kernel reference, simplify to 3-tuple interface, document
B=2, T=4096, D=2048, W=4 test case. Match trimul spec wording.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Add check_implementation Test

**Files:**
- Modify: `tests/test_gpu_mode_causal_conv1d.py`

**Step 1: Write test for check_implementation**

Add to TestIntegration class:

```python
def test_check_implementation_passes_baseline(self):
    """Verify check_implementation accepts correct output."""
    data = generate_input(B=2, T=128, D=64, W=4, seed=42)
    output = custom_kernel(data)

    passed, message = check_implementation(data, output)
    assert passed, f"Baseline should pass: {message}"

def test_check_implementation_fails_wrong_output(self):
    """Verify check_implementation rejects incorrect output."""
    data = generate_input(B=2, T=128, D=64, W=4, seed=42)
    wrong_output = torch.zeros_like(data[0])

    passed, message = check_implementation(data, wrong_output)
    assert not passed, "Should fail for wrong output"

def test_check_implementation_fails_wrong_shape(self):
    """Verify check_implementation rejects wrong shape."""
    data = generate_input(B=2, T=128, D=64, W=4, seed=42)
    wrong_output = torch.zeros(2, 64, 64, device="cuda", dtype=torch.bfloat16)

    passed, message = check_implementation(data, wrong_output)
    assert not passed, "Should fail for wrong shape"
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestIntegration::test_check_implementation_passes_baseline -xvs`
Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestIntegration::test_check_implementation_fails_wrong_output -xvs`
Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestIntegration::test_check_implementation_fails_wrong_shape -xvs`

Expected: PASS for all three (check_implementation already uses make_match_reference)

**Step 3: Commit test additions**

```bash
git add tests/test_gpu_mode_causal_conv1d.py
git commit -m "test(causal_conv1d): add check_implementation validation tests

Verify check_implementation correctly accepts valid outputs and rejects
invalid outputs and wrong shapes.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Run Full Test Suite and Verify E2E

**Files:**
- No file changes

**Step 1: Run complete test suite**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py -xvs`

Expected: All tests PASS

**Step 2: Run with larger test case**

Add temporary test to verify performance test case works:

```python
@pytest.mark.cuda
@pytest.mark.slow
class TestPerformance:
    def test_full_scale_test_case(self):
        """Verify reference and baseline work on full-scale test case."""
        # B=2, T=4096, D=2048, W=4 from spec
        data = generate_input(B=2, T=4096, D=2048, W=4, seed=42)

        ref_output = ref_kernel(data)
        baseline_output = custom_kernel(data)

        assert ref_output.shape == (2, 4096, 2048)
        assert baseline_output.shape == (2, 4096, 2048)
        assert torch.allclose(ref_output, baseline_output, rtol=2e-2, atol=2e-2)
```

Run: `pytest tests/test_gpu_mode_causal_conv1d.py::TestPerformance::test_full_scale_test_case -xvs`

Expected: PASS (may be slow, verifies GPU memory handling)

**Step 3: Clean up test file**

Remove any temporary debugging tests, ensure final test structure is clean.

**Step 4: Final commit**

```bash
git add tests/test_gpu_mode_causal_conv1d.py
git commit -m "test(causal_conv1d): add full-scale performance test case

Verify reference and baseline work on B=2, T=4096, D=2048, W=4 test case
from spec.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Step 5: Run complete test suite one final time**

Run: `pytest tests/test_gpu_mode_causal_conv1d.py -v`

Expected: All tests PASS

**Step 6: Verify types with mypy**

Run: `ty check k_search/tasks/gpu_mode/causal_conv1d/`

Expected: No type errors

---

## Task 8: Verify Task Integration

**Files:**
- No file changes (verification only)

**Step 1: Test task instantiation**

Create temporary test script `test_task_integration.py`:

```python
import sys
from pathlib import Path

# Add task to path
task_dir = Path("k_search/tasks/gpu_mode/causal_conv1d")
sys.path.insert(0, str(task_dir))

from reference import generate_input, ref_kernel, check_implementation
from submission import custom_kernel
from spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON

# Test basic workflow
data = generate_input(B=2, T=128, D=64, W=4, seed=42)
ref_output = ref_kernel(data)
baseline_output = custom_kernel(data)

passed, msg = check_implementation(data, baseline_output)
print(f"Check implementation: {passed}")
print(f"Message: {msg}")

print(f"\nSpec text length: {len(CAUSAL_CONV1D_SPEC_TEXT_TRITON)} chars")
print(f"Contains interface: {'(x, weight, config)' in CAUSAL_CONV1D_SPEC_TEXT_TRITON}")
```

Run: `python test_task_integration.py`

Expected: "Check implementation: True"

**Step 2: Clean up test script**

```bash
rm test_task_integration.py
```

**Step 3: Document completion**

Update plan status:

All tasks complete. Causal conv1d task successfully simplified to match trimul structure:
- ✅ Type definitions updated to 3-tuple
- ✅ generate_input simplified (no bias/residual/activation params)
- ✅ Reference implementation using pure PyTorch
- ✅ Baseline submission using pure PyTorch
- ✅ Spec text updated to match trimul format
- ✅ All tests passing
- ✅ E2E workflow verified

---

## Post-Implementation Checklist

- [ ] All tests pass: `pytest tests/test_gpu_mode_causal_conv1d.py -v`
- [ ] Type checking passes: `ty check k_search/tasks/gpu_mode/causal_conv1d/`
- [ ] No FLA dependencies remain in task files
- [ ] Spec text is clear and matches trimul format
- [ ] Baseline matches reference perfectly
- [ ] Full-scale test case (B=2, T=4096, D=2048, W=4) works
- [ ] All commits have descriptive messages

## Notes for Future Work

Once this minimal version is validated:
- Consider adding optional bias parameter
- Consider adding optional residual connection
- Consider supporting cu_seqlens for variable-length sequences
- Consider supporting different activations (not just SiLU)
- Benchmark against FLA implementation for performance comparison
