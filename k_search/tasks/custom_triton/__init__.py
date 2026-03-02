"""Generic evaluation framework for custom Triton kernel tasks."""

from .aggregation import AggregationStrategy, GeometricMeanAggregation, MeanAggregation
from .benchmarking import BenchmarkConfig, BenchmarkHarness, benchmark_triton_kernel, clear_l2_cache
from .correctness import CorrectnessConfig, check_correctness
from .evaluator import GenericKernelEvaluator

__all__ = [
    "AggregationStrategy",
    "BenchmarkConfig",
    "BenchmarkHarness",
    "CorrectnessConfig",
    "GenericKernelEvaluator",
    "GeometricMeanAggregation",
    "MeanAggregation",
    "benchmark_triton_kernel",
    "check_correctness",
    "clear_l2_cache",
]
