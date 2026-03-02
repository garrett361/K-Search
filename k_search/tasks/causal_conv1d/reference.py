"""Reference implementation wrapper for Flash Linear Attention causal_conv1d."""

from __future__ import annotations

import logging
from typing import Any

import torch


logger = logging.getLogger(__name__)


def fla_causal_conv1d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    activation: str | None = "silu",
    **kwargs: Any,
) -> torch.Tensor:
    """
    Wrapper around Flash Linear Attention causal_conv1d_fwd implementation.

    Args:
        x: Input tensor [B, T, D]
        weight: Convolution weights [D, W]
        bias: Optional bias term [D]
        residual: Optional residual connection [B, T, D]
        activation: Activation function ('silu' or None)
        **kwargs: Additional arguments (ignored)

    Returns:
        Output tensor [B, T, D]
    """
    try:
        from fla.modules.convolution import causal_conv1d_fwd
    except ImportError as e:
        logger.error("Failed to import FLA causal_conv1d. Install with: pip install flash-linear-attention")
        raise RuntimeError(
            "flash-linear-attention package not found. "
            "Install with: pip install flash-linear-attention"
        ) from e

    y, _ = causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        activation=activation,
    )

    return y


def create_reference_inputs(
    B: int,
    T: int,
    D: int,
    W: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
    with_bias: bool = True,
    with_residual: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Generate random test inputs for causal_conv1d.

    Args:
        B: Batch size
        T: Sequence length
        D: Feature dimension
        W: Convolution width
        dtype: Tensor dtype
        device: Device to place tensors on
        with_bias: Whether to include bias tensor
        with_residual: Whether to include residual tensor

    Returns:
        Dictionary with keys: x, weight, bias (optional), residual (optional)
    """
    torch.manual_seed(42)

    inputs = {
        "x": torch.randn(B, T, D, dtype=dtype, device=device),
        "weight": torch.randn(D, W, dtype=dtype, device=device),
    }

    if with_bias:
        inputs["bias"] = torch.randn(D, dtype=dtype, device=device)
    else:
        inputs["bias"] = None

    if with_residual:
        inputs["residual"] = torch.randn(B, T, D, dtype=dtype, device=device)
    else:
        inputs["residual"] = None

    return inputs
