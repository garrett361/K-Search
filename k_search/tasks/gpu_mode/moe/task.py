from typing import TypeAlias, TypedDict

import torch

input_t: TypeAlias = tuple[
    torch.Tensor,  # x: [total_tokens, dim] bfloat16 - tokens sorted by expert
    torch.Tensor,  # w1: [num_experts, hidden_dim, dim] bfloat16 - gate projection
    torch.Tensor,  # w2: [num_experts, dim, hidden_dim] bfloat16 - down projection
    torch.Tensor,  # w3: [num_experts, hidden_dim, dim] bfloat16 - up projection
    torch.Tensor,  # num_tokens_per_expert: [num_experts] int32
    dict[str, str],  # config (reserved)
]
output_t: TypeAlias = torch.Tensor  # [total_tokens, dim] bfloat16


class TestSpec(TypedDict):
    seq_tokens: int  # Original sequence tokens before routing
    top_k: int  # Experts activated per token (sparsity control)
    dim: int  # Model dimension
    hidden_dim: int  # Expert FFN intermediate dimension
    num_experts: int  # Total routed experts
    seed: int
