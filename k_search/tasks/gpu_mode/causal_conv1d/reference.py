from task import input_t, output_t
from utils import make_match_reference

import torch
import torch.nn.functional as F


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
    """Task protocol wrapper for reference implementation.

    Args:
        data: Input tuple (x, weight, config) where:
            - x: Input tensor [B, T, D] bfloat16
            - weight: Convolution weights [D, W] bfloat16
            - config: Configuration dict (currently unused)

    Returns:
        Output tensor [B, T, D] bfloat16 after causal conv + SiLU.
    """
    x, weight, config = data
    return causal_conv1d_ref_torch(x, weight)


def generate_input(
    B: int,
    T: int,
    D: int,
    W: int,
    seed: int,
) -> input_t:
    """Generate random inputs for causal conv1d task.

    Args:
        B: Batch size.
        T: Sequence length (time steps).
        D: Number of channels (feature dimension).
        W: Convolution kernel width.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (x, weight, config) where:
            - x: Random input tensor [B, T, D] bfloat16
            - weight: Random convolution weights [D, W] bfloat16
            - config: Empty configuration dict
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    weight = torch.randn(D, W, device="cuda", dtype=torch.bfloat16, generator=gen)

    config = {}

    return (x, weight, config)


check_implementation = make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)
