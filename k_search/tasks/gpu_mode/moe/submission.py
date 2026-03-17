import torch
import torch.nn.functional as F

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """For-loop MoE expert computation baseline.

    Optimization target: replace for-loop with Triton kernels or better PyTorch.
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
