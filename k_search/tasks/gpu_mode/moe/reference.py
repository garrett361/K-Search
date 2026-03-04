import torch
import torch.nn.functional as F

from task import input_t, output_t
from utils import make_match_reference


def ref_kernel(data: input_t) -> output_t:
    """For-loop MoE expert computation (from torchtitan).

    SwiGLU: h = silu(x @ w1.T) * (x @ w3.T); out = h @ w2.T
    """
    x, w1, w2, w3, num_tokens_per_expert, config = data

    num_tokens_per_expert_list = num_tokens_per_expert.tolist()
    x_splits = torch.split(x, num_tokens_per_expert_list, dim=0)

    out_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        if x_expert.numel() == 0:
            continue
        h = F.silu(x_expert @ w1[expert_idx].T) * (x_expert @ w3[expert_idx].T)
        out = h @ w2[expert_idx].T
        out_splits.append(out)

    return torch.cat(out_splits, dim=0) if out_splits else x.new_empty((0, x.shape[1]))


def generate_input(
    seq_tokens: int,
    top_k: int,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    seed: int,
) -> input_t:
    """Generate test data with balanced token distribution across experts.

    Args:
        seq_tokens: Original sequence tokens before routing
        top_k: Experts activated per token (controls sparsity)
        dim: Model dimension
        hidden_dim: Expert FFN intermediate dimension
        num_experts: Total routed experts
        seed: Random seed

    Sparsity ratio = top_k / num_experts
    Total expert activations = seq_tokens * top_k
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    total_tokens = seq_tokens * top_k

    x = torch.randn(
        total_tokens, dim, device="cuda", dtype=torch.bfloat16, generator=gen
    )

    w1 = (
        torch.randn(
            num_experts,
            hidden_dim,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            num_experts,
            dim,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.02
    )
    w3 = (
        torch.randn(
            num_experts,
            hidden_dim,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.02
    )

    base_tokens = total_tokens // num_experts
    remainder = total_tokens % num_experts
    counts = [base_tokens + (1 if i < remainder else 0) for i in range(num_experts)]
    num_tokens_per_expert = torch.tensor(counts, device="cuda", dtype=torch.int32)

    return (x, w1, w2, w3, num_tokens_per_expert, {})


check_implementation = make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)
