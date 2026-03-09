"""GPU Mode Evaluator implementation."""

import logging
from typing import Any

from k_search.modular.adapters.gpu_mode.wrappers import GpuModeEvaluationResult
from k_search.modular.protocols import EvaluationResult, Implementation
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

logger = logging.getLogger(__name__)


class GpuModeEvaluator:
    """Evaluator that delegates to GpuModeTriMulTask.run_benchmark()."""

    def __init__(self, task: GpuModeTriMulTask) -> None:
        self._task = task
        self._reference_latency_ms: float | None = None

    def _get_ref_latency(self) -> float | None:
        """Run baseline submission.py once and cache reference latency."""
        if self._reference_latency_ms is not None:
            return self._reference_latency_ms

        submission_path = self._task._cfg.task_dir / "submission.py"
        if not submission_path.exists():
            return None

        from k_search.tasks.gpu_mode.evaluator import evaluate_trimul_submission

        baseline_code = submission_path.read_text()
        summary = evaluate_trimul_submission(
            submission_code=baseline_code,
            mode=self._task._cfg.mode,
            language="triton",
            task_dir=self._task._cfg.task_dir,
        )
        if summary.status != "passed" or not summary.latency_ms:
            raise RuntimeError(f"Reference benchmark failed: {summary.status}")
        self._reference_latency_ms = summary.latency_ms
        logger.info("Reference latency: %.3f ms", self._reference_latency_ms)
        return self._reference_latency_ms

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
        return GpuModeEvaluationResult(
            eval_result, reference_latency_ms=self._get_ref_latency()
        )
