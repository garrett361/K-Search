"""GPU Mode Evaluator implementation."""

from typing import Any

from k_search.task_framework.adapters.gpu_mode.types import GpuModeEvaluationResult
from k_search.task_framework.protocols.results import EvaluationResult, Implementation
from k_search.tasks.gpu_mode_task import GpuModeTask


class GpuModeEvaluator:
    """Evaluator that delegates to GpuModeTask.run_benchmark()."""

    def __init__(self, task: GpuModeTask) -> None:
        self._task = task

    def evaluate(
        self,
        impl: Implementation,
        *,
        timeout_secs: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate implementation by delegating to run_benchmark."""
        solution = impl.content
        eval_result = self._task.run_benchmark(solution=solution)
        return GpuModeEvaluationResult(eval_result)
