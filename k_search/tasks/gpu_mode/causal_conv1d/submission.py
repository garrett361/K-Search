from task import input_t, output_t
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
