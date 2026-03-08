"""GPU Mode Evaluator implementation."""

from typing import Any

from k_search.modular.adapters.gpu_mode.wrappers import GpuModeEvaluationResult
from k_search.modular.protocols import EvaluationResult, Implementation
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask


class GpuModeEvaluator:
    """Evaluator that delegates to GpuModeTriMulTask.run_benchmark()."""

    def __init__(self, task: GpuModeTriMulTask) -> None:
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
        round_num = (context or {}).get("round_idx")
        eval_result = self._task.run_benchmark(solution=solution, round_num=round_num)
        return GpuModeEvaluationResult(eval_result)
