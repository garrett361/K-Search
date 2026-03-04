from pathlib import Path


def _read_reference_submission_py() -> str:
    try:
        return (Path(__file__).resolve().parent / "submission.py").read_text()
    except Exception:
        return ""


_CODE = _read_reference_submission_py()
_REFERENCE_BLOCK = (
    f"\nReference code (baseline `submission.py`):\n\n```python\n{_CODE}\n```\n"
    if _CODE
    else ""
)

MOE_SPEC_TEXT_TRITON = (
    """
GPUMode Mixture of Experts (Triton submission)

Task:
- Optimize the forward pass of SwiGLU-based Mixture of Experts.
- Tokens are pre-sorted by expert assignment.
- Primary goal: eliminate Python for-loop over experts.
  - Consider custom **Triton kernels** for fused operations.
- Include short comment at top summarizing implementation.

Data interface:
- Python/Triton: custom_kernel(data) where:
  data = (x, w1, w2, w3, num_tokens_per_expert, config)
- Include code inside '```' and '```' blocks.

Inputs:
- x: torch.Tensor [total_tokens, dim] bfloat16 - tokens sorted by expert
- w1: torch.Tensor [num_experts, hidden_dim, dim] bfloat16 - gate projection
- w2: torch.Tensor [num_experts, dim, hidden_dim] bfloat16 - down projection
- w3: torch.Tensor [num_experts, hidden_dim, dim] bfloat16 - up projection
- num_tokens_per_expert: torch.Tensor [num_experts] int32
- config: Dict (currently empty)

Output:
- torch.Tensor [total_tokens, dim] bfloat16

Algorithm (SwiGLU per expert):
```python
h = silu(x @ w1[i].T) * (x @ w3[i].T)  # gate * up
out = h @ w2[i].T                       # down projection
```

Correctness:
- Must match reference within: rtol=2e-2, atol=2e-2.

Test Cases (optimize for these):
- Arcee Trinity Large: seq_tokens=4096, top_k=4, dim=3072, hidden_dim=3072, num_experts=256
  (total_tokens=16384, sparsity=1.56%)

Expert token boundaries:
- offsets = cumsum(num_tokens_per_expert)  # [num_experts], same device as inputs
- Expert i processes tokens from offsets[i-1] (or 0 if i==0) to offsets[i]

Triton implementation notes:
- tl.dot(a, b) requires a.dtype == b.dtype
- For fp32 accumulation: cast BOTH operands with .to(tl.float32) before tl.dot()
- Use tl.trans(tensor) for transposition - tl.dot() has no trans_a/trans_b args
- Cast output to bf16 with .to(tl.bfloat16) before tl.store()

Triton block size constraints:
- BLOCK sizes for tl.arange, tl.zeros, tl.dot must be powers of 2 (32, 64, 128, etc.)
- dim=3072 is NOT a power of 2 - use BLOCK_K=128 and loop with masking
- Pattern: for k in range(0, dim, BLOCK_K): mask = (k + tl.arange(0, BLOCK_K)) < dim

Triton control flow limitations:
- No `continue` or `break` inside tl.static_range loops
- Use tl.where(cond, val, 0.0) instead of if/continue patterns
- Triton functions (@triton.jit) cannot be called from Python - only as kernels

Optimization hints:
- Fuse the two up-projections (w1, w3) if possible
"""
    + _REFERENCE_BLOCK
)
