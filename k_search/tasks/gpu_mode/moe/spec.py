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
- Tokens are pre-sorted by expert assignment (grouped_mm compatible).
- Primary goal: eliminate Python for-loop over experts.
  - Use **torch._grouped_mm** for parallel expert computation.
  - Or implement custom **Triton kernels** for fused operations.
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

Optimization hints:
- torch._grouped_mm(x, w.transpose(-2,-1), offs=cumsum(num_tokens_per_expert))
  replaces the for-loop with a single fused operation
- Offsets must be int32 cumsum of token counts
- Fuse the two up-projections (w1, w3) if possible
"""
    + _REFERENCE_BLOCK
)
