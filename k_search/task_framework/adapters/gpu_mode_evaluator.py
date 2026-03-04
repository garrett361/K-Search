"""GpuModeEvaluator: Evaluator implementation for GpuModeTask."""

from typing import Any

from k_search.tasks.gpu_mode_task import GpuModeTask
from k_search.task_framework.protocols.results import EvaluationResult, Implementation
from k_search.task_framework.adapters.wrappers import GpuModeEvaluationResult


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
        """Evaluate implementation by delegating to V1 run_benchmark."""
        # Extract Solution from Implementation wrapper
        # GpuModeImplementation wraps Solution as .content
        solution = impl.content

        # Delegate to V1 evaluation
        eval_result = self._task.run_benchmark(solution=solution)

        # Wrap in protocol-compliant result
        return GpuModeEvaluationResult(eval_result)
