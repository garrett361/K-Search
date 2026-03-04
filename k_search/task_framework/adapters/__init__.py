"""Adapters wrapping existing task implementations."""

from k_search.task_framework.adapters.wrappers import (
    GpuModeEvaluationResult,
    GpuModeSolutionArtifact,
)
from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter

__all__ = [
    "GpuModeAdapter",
    "GpuModeEvaluationResult",
    "GpuModeSolutionArtifact",
]
