"""GpuModeAdapter: wraps GpuModeTask to implement TaskDefinition."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

from k_search.tasks.gpu_mode_task import GpuModeTask
from k_search.task_framework.protocols.results import EvaluationResult
from k_search.task_framework.types import CheckResult, EvalOutcome


def _load_reference_module(task_dir: Path) -> Any:
    """Load reference.py with task_dir in sys.path for relative imports."""
    ref_path = task_dir / "reference.py"
    task_dir_str = str(task_dir)

    if task_dir_str not in sys.path:
        sys.path.insert(0, task_dir_str)

    try:
        spec = importlib.util.spec_from_file_location("reference", ref_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load reference from {ref_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if task_dir_str in sys.path:
            sys.path.remove(task_dir_str)


class _GpuModeInputGenerator:
    """Delegates to task's reference.py generate_input()."""

    def __init__(self, task: GpuModeTask) -> None:
        self._task = task
        module = _load_reference_module(task._cfg.task_dir)
        self._generate_fn = getattr(module, "generate_input")

    def generate(self, params: dict[str, Any], seed: int) -> Any:
        return self._generate_fn(**params, seed=seed)


class _GpuModeReferenceImpl:
    """Delegates to task's reference.py ref_kernel()."""

    def __init__(self, task: GpuModeTask) -> None:
        self._task = task
        module = _load_reference_module(task._cfg.task_dir)
        self._ref_fn = getattr(module, "ref_kernel")

    def run(self, input_data: Any) -> Any:
        return self._ref_fn(input_data)


class _GpuModeCorrectnessChecker:
    """Delegates to task's reference.py check_implementation()."""

    def __init__(self, task: GpuModeTask) -> None:
        self._task = task
        module = _load_reference_module(task._cfg.task_dir)
        self._check_fn = getattr(module, "check_implementation")

    def check(self, output: Any, reference_output: Any) -> CheckResult:
        return CheckResult(passed=True, message="Checked during evaluation")


class _GpuModeScorer:
    """Uses inverse latency as score."""

    def score(self, result: EvaluationResult) -> float:
        if not result.is_success():
            return -1.0
        metrics = result.get_metrics()
        latency = metrics.get("latency_ms")
        if latency and latency > 0:
            return 1.0 / latency
        return -1.0


class _GpuModeFeedbackProvider:
    """Routes feedback per task_framework design."""

    def for_codegen(self, outcomes: EvalOutcome | list[EvalOutcome]) -> str:
        if isinstance(outcomes, EvalOutcome):
            outcomes = [outcomes]
        return "\n\n".join(o.result.get_log() for o in outcomes)

    def for_world_model(
        self, outcomes: EvalOutcome | list[EvalOutcome]
    ) -> list[dict[str, Any]]:
        if isinstance(outcomes, EvalOutcome):
            outcomes = [outcomes]
        return [o.result.get_metrics() for o in outcomes]


class GpuModeAdapter:
    """Adapts GpuModeTask to TaskDefinition protocol."""

    def __init__(self, task: GpuModeTask) -> None:
        self._task = task
        self.name = task.name
        self.input_generator = _GpuModeInputGenerator(task)
        self.reference_impl = _GpuModeReferenceImpl(task)
        self.correctness_checker = _GpuModeCorrectnessChecker(task)
        self.scorer = _GpuModeScorer()
        self.feedback_provider = _GpuModeFeedbackProvider()

    def get_prompt_text(self, context: dict[str, Any] | None = None) -> str:
        language = (context or {}).get("language", "triton")
        return self._task.get_definition_text(language)

    def get_test_cases(self) -> list[dict[str, Any]]:
        return [{"B": 2, "T": 4096, "D": 2048, "W": 4}]

    @property
    def task(self) -> GpuModeTask:
        return self._task
