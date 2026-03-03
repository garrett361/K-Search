# Causal Conv1d Task Simplification Design

**Date:** 2026-03-03
**Status:** Approved

## Goal

Simplify the causal_conv1d GPU Mode task to align with the trimul example structure. Replace the FLA Triton kernel reference with a minimal PyTorch reference based on `causal_conv1d_ref_torch` from test_conv.py.

## Motivation

The current causal_conv1d task uses FLA's Triton kernel as reference, making it dependent on external code and more complex than necessary for initial development. The trimul task demonstrates a cleaner pattern using pure PyTorch references. Simplifying to this pattern will:

- Remove FLA dependency
- Provide clearer reference implementation for LLM prompt context
- Align task structure across GPU Mode tasks
- Enable faster iteration with minimal configuration

## Approach

Adopt a minimal-first strategy (Approach 1 with bfloat16):
- Strip down to bare essentials
- Remove all optional features (bias, residual, cu_seqlens, variable activation)
- Hardcode SiLU activation
- Use bfloat16 throughout (no dtype conversions)
- Match trimul's structural patterns

## Design

### 1. Reference Implementation

File: `k_search/tasks/gpu_mode/causal_conv1d/reference.py`

Create `causal_conv1d_ref_torch` that implements minimal causal convolution:

```python
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

**Key simplifications:**
- No dtype conversions (stay in bfloat16)
- No bias, initial_state, final_state, residual
- Hardcoded SiLU activation
- Direct implementation using PyTorch F.conv1d

### 2. Data Interface

File: `k_search/tasks/gpu_mode/causal_conv1d/task.py`

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

**Changes:**
- 3-tuple instead of 5-tuple: `(x, weight, config)`
- Remove bias, residual from tuple
- Remove activation, withbias, withresidual from TestSpec
- Minimal config dict (empty, reserved for future use)

### 3. Input Generation

File: `k_search/tasks/gpu_mode/causal_conv1d/reference.py`

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

**Changes:**
- Remove activation, withbias, withresidual parameters
- Generate only x and weight tensors
- Empty config dict
- Keep seeded generation for reproducibility

### 4. Baseline Submission

File: `k_search/tasks/gpu_mode/causal_conv1d/submission.py`

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

**Changes:**
- Remove FLA dependency
- Pure PyTorch implementation
- Same logic as reference (perfect baseline accuracy)

### 5. Spec Text

File: `k_search/tasks/gpu_mode/causal_conv1d/spec.py`

Structure modeled directly on trimul spec:

```python
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

**Changes:**
- Remove FLA Triton kernel reference
- Simplify to 3-tuple interface
- Remove bias, residual, variable activation mentions
- Single fixed test case: B=2, T=4096, D=2048, W=4
- Match trimul's task description structure and wording
- Keep optimization hints relevant to the problem

### 6. Test Updates

File: `tests/test_gpu_mode_causal_conv1d.py`

Update tests to match simplified interface:

```python
@pytest.mark.cuda
class TestGenerateInput:
    def test_generate_input_shapes(self):
        B, T, D, W = 2, 128, 64, 4
        x, weight, config = generate_input(B=B, T=T, D=D, W=W, seed=42)

        assert x.shape == (B, T, D)
        assert weight.shape == (D, W)
        assert x.dtype == torch.bfloat16
        assert weight.dtype == torch.bfloat16
        assert isinstance(config, dict)

    def test_generate_input_seeded(self):
        data1 = generate_input(B=2, T=64, D=32, W=4, seed=123)
        data2 = generate_input(B=2, T=64, D=32, W=4, seed=123)

        assert torch.allclose(data1[0], data2[0])
        assert torch.allclose(data1[1], data2[1])


@pytest.mark.cuda
class TestIntegration:
    def test_baseline_correctness(self):
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)

        expected = ref_kernel(data)
        actual = custom_kernel(data)

        assert torch.allclose(expected, actual, rtol=2e-2, atol=2e-2)

    def test_check_implementation_passes_baseline(self):
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        output = custom_kernel(data)

        passed, message = check_implementation(data, output)
        assert passed, f"Baseline should pass: {message}"

    def test_check_implementation_fails_wrong_output(self):
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        wrong_output = torch.zeros_like(data[0])

        passed, message = check_implementation(data, wrong_output)
        assert not passed, "Should fail for wrong output"
```

**Changes:**
- Remove FLA-dependent spec tests
- Remove bias/residual parameter tests
- Update to 3-tuple unpacking
- Keep test structure and coverage

## Files Modified

1. `k_search/tasks/gpu_mode/causal_conv1d/task.py` - Type definitions
2. `k_search/tasks/gpu_mode/causal_conv1d/reference.py` - Reference implementation
3. `k_search/tasks/gpu_mode/causal_conv1d/submission.py` - Baseline submission
4. `k_search/tasks/gpu_mode/causal_conv1d/spec.py` - Spec text
5. `tests/test_gpu_mode_causal_conv1d.py` - Test updates

## Validation Plan

E2E testing:
1. Unit tests pass for generate_input (shape, dtype, seeding)
2. Baseline submission matches reference perfectly
3. check_implementation correctly validates/rejects outputs
4. Spec text loads and contains required interface details
5. Full task can be instantiated and run through eval pipeline

## Future Extensions

Once minimal version is working:
- Add back optional bias parameter
- Add back optional residual connection
- Support variable-length sequences (cu_seqlens)
- Support different activations
- Add larger test cases

But for now: minimal, working, aligned with trimul structure.
