"""Generic benchmarking harness for custom Triton kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
from triton.testing import do_bench


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmarking parameters."""

    warmup: int = 25
    rep: int = 100


class BenchmarkHarness(Protocol):
    """
    Backend-specific benchmark runner protocol.

    Implementations handle execution for specific kernel types
    (Triton, CUDA, libkernelbot, etc.).
    """

    def run(
        self,
        kernel_fn: Callable[..., Any],
        inputs: dict[str, Any],
        config: BenchmarkConfig,
    ) -> dict[str, float]:
        """
        Execute and benchmark a kernel.

        Args:
            kernel_fn: Compiled kernel function to benchmark
            inputs: Dictionary of input tensors/arguments
            config: Benchmark configuration

        Returns:
            Dictionary containing at minimum:
            - latency_ms: Mean latency in milliseconds
        """
        ...


def benchmark_triton_kernel(
    kernel_fn: Callable[..., Any],
    inputs: dict[str, Any],
    config: BenchmarkConfig,
) -> dict[str, float]:
    """
    Benchmark a Triton kernel using triton.testing.do_bench.

    Args:
        kernel_fn: Compiled Triton kernel function
        inputs: Dictionary of input tensors/arguments
        config: Benchmark configuration

    Returns:
        Dictionary with latency_ms (mean) and latency quantiles
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for benchmarking")

    device = torch.device("cuda")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.device != device:
            inputs[k] = v.to(device)

    fn = lambda: kernel_fn(**inputs)  # noqa: E731

    latency_ms = do_bench(fn, warmup=config.warmup, rep=config.rep)

    return {"latency_ms": latency_ms}


def clear_l2_cache() -> None:
    """Clear L2 cache to ensure fair benchmarking between runs."""
    if not torch.cuda.is_available():
        return

    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    dummy.fill_(42)
    del dummy
    torch.cuda.synchronize()
