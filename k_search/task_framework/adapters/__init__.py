"""Adapters wrapping existing task implementations."""

from k_search.task_framework.adapters.wrappers import (
    GpuModeEvaluationResult,
    GpuModeImplementation,
)
from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter
from k_search.task_framework.adapters.gpu_mode_evaluator import GpuModeEvaluator

__all__ = [
    "GpuModeAdapter",
    "GpuModeEvaluationResult",
    "GpuModeEvaluator",
    "GpuModeImplementation",
]
