"""Triton execution harness for causal_conv1d kernels."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch
import triton

logger = logging.getLogger(__name__)


class CausalConv1dHarness:
    """
    Triton execution and benchmarking harness for causal_conv1d kernels.

    Handles:
    - Triton code compilation
    - Kernel execution
    - Benchmarking with proper warmup
    """

    def __init__(self):
        self._compiled_kernel: Callable | None = None
        self._kernel_code: str | None = None

    def compile_kernel(self, kernel_code: str) -> Callable:
        """
        Compile Triton kernel code.

        Args:
            kernel_code: Triton kernel source code

        Returns:
            Compiled kernel function

        Raises:
            RuntimeError: If compilation fails
        """
        if kernel_code == self._kernel_code and self._compiled_kernel is not None:
            return self._compiled_kernel

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(kernel_code)
                temp_path = Path(f.name)

            namespace: dict[str, Any] = {"triton": triton, "torch": torch, "tl": triton.language}

            exec(compile(kernel_code, str(temp_path), "exec"), namespace)

            temp_path.unlink(missing_ok=True)

            kernel_fn = None
            for name, obj in namespace.items():
                if hasattr(obj, "__module__") and hasattr(obj, "__name__"):
                    if "triton" in str(getattr(obj, "__module__", "")):
                        kernel_fn = obj
                        break

            if kernel_fn is None:
                raise RuntimeError("No Triton kernel function found in compiled code")

            self._compiled_kernel = kernel_fn
            self._kernel_code = kernel_code

            return kernel_fn

        except Exception as e:
            logger.error(f"Kernel compilation failed: {e}")
            raise RuntimeError(f"Triton kernel compilation failed: {e}") from e

    def execute_once(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        activation: str | None = "silu",
    ) -> torch.Tensor:
        """
        Execute kernel once (for correctness checking).

        Args:
            x: Input tensor [B, T, D]
            weight: Weight tensor [D, W]
            bias: Optional bias [D]
            residual: Optional residual [B, T, D]
            activation: Activation function

        Returns:
            Output tensor [B, T, D]
        """
        if self._compiled_kernel is None:
            raise RuntimeError("Kernel not compiled. Call compile_kernel first.")

        B, T, D = x.shape
        W = weight.shape[1]

        output = torch.empty_like(x)

        self._compiled_kernel(
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias if bias is not None else torch.empty(0, device=x.device),
            residual_ptr=residual if residual is not None else torch.empty(0, device=x.device),
            out_ptr=output,
            B=B,
            T=T,
            D=D,
            W=W,
            activation=activation,
        )

        return output
