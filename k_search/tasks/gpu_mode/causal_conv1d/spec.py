"""Causal Conv1d spec text for prompting."""

from __future__ import annotations

from pathlib import Path


def _read_causal_conv1d_reference_submission_py() -> str:
    """
    Load the reference `submission.py` so prompts can include a concrete baseline.
    This is best-effort; tasks should still function if the file is missing.
    """
    try:
        p = Path(__file__).resolve().parent / "submission.py"
        return p.read_text()
    except Exception:
        return ""


_CAUSAL_CONV1D_REFERENCE_SUBMISSION_PY = _read_causal_conv1d_reference_submission_py().strip()
_CAUSAL_CONV1D_REFERENCE_BLOCK = (
    ("\nReference code (baseline `submission.py`):\n\n" + _CAUSAL_CONV1D_REFERENCE_SUBMISSION_PY + "\n")
    if _CAUSAL_CONV1D_REFERENCE_SUBMISSION_PY
    else ""
)


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

""" + _CAUSAL_CONV1D_REFERENCE_BLOCK
