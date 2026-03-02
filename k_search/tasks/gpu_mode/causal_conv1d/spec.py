"""Causal Conv1d spec text for prompting."""

from __future__ import annotations


def _read_reference_kernel() -> str:
    """Load FLA causal_conv1d Triton kernel source for inclusion in prompts."""
    import inspect
    import re

    import fla.modules.convolution as conv_module

    source_file = inspect.getfile(conv_module)
    with open(source_file) as f:
        content = f.read()

    pattern = r"(@triton\.heuristics.*?@triton\.jit\ndef causal_conv1d_fwd_kernel\(.*?\n)(\s{4}.*?)(?=\n@|\ndef [a-z]|\nclass |\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1) + match.group(2)
    else: 
        raise ValueError("FLA ref kernel not found")


_REFERENCE_KERNEL = _read_reference_kernel().strip()
_REFERENCE_BLOCK = (
    "\nReference code (FLA baseline kernel):\n\n```python\n" + _REFERENCE_KERNEL + "\n```\n"
    if _REFERENCE_KERNEL
    else ""
)


CAUSAL_CONV1D_SPEC_TEXT_TRITON = """Causal Conv1d Forward (Triton submission)

Task:
- Optimize the *forward* pass of causal 1D convolution used in state space models (Mamba, etc).
- You may use *mixed precision* computations, but the final output must match the input dtype.
- You do not need to implement everything in Triton; you may use PyTorch for some operations.
- Include a short comment at the top summarizing your implementation.
- For each round, you can see your current best solution and the previous round's summary.

Data interface:
- Python/Triton: `custom_kernel(data)` where:
  `data = (x, weight, bias, residual, config)`
- Include the code inside '```' and '```' blocks.

Inputs:
- x: torch.Tensor, shape [B, T, D], dtype bfloat16
- weight: torch.Tensor, shape [D, W], dtype bfloat16
- bias: torch.Tensor or None, shape [D], dtype bfloat16
- residual: torch.Tensor or None, shape [B, T, D], dtype bfloat16
- config: Dict with keys:
  - activation: str ('silu' or None)

Output:
- torch.Tensor, shape [B, T, D], dtype bfloat16

Mathematical Definition:
```
conv_out[b, t, d] = sum_{i=0}^{min(t, W-1)} x[b, t-i, d] * weight[d, i]
activated[b, t, d] = activation(conv_out[b, t, d] + bias[d])
y[b, t, d] = activated[b, t, d] + residual[b, t, d]  # if residual provided
```

Correctness:
- Must match reference within tolerances: rtol=2e-2, atol=2e-2

Problem Constraints:
- B in {2, 4, 8}
- T in {4096, 8192}
- D in {2048, 4096}
- W = 4 (fixed)
- activation = 'silu' (fixed)
- Input distribution: standard Normal

Remarks:
- The convolution width W=4 is small; consider loop unrolling.
- Memory access patterns matter: coalesce reads/writes along the D dimension.
- The D dimension is large; tile/block over it for occupancy.
- Causality constraint: output at time t depends only on inputs at times [max(0, t-W+1), ..., t].
- Fuse the activation (SiLU) to avoid memory round-trips.
- Handle boundary conditions (t < W) efficiently.

""" + _REFERENCE_BLOCK
