"""GPU Mode TaskDefinition implementation."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

from k_search.modular.adapters.gpu_mode.wrappers import GpuModeImplementation
from k_search.modular.results import CheckResult
from k_search.modular.llm_utils import strip_markdown_fences
from k_search.modular.protocols.eval_result import EvaluationResult
from k_search.modular.round import Round
from k_search.tasks.gpu_mode_task import GpuModeTask
from k_search.tasks.task_base import BuildSpec, Solution, SourceFile, SupportedLanguages


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


class _InputGenerator:
    def __init__(self, task: GpuModeTask) -> None:
        self._task = task
        module = _load_reference_module(task._cfg.task_dir)
        self._generate_fn = getattr(module, "generate_input")

    def generate(self, params: dict[str, Any], seed: int) -> Any:
        return self._generate_fn(**params, seed=seed)


class _ReferenceImpl:
    def __init__(self, task: GpuModeTask) -> None:
        self._task = task
        module = _load_reference_module(task._cfg.task_dir)
        self._ref_fn = getattr(module, "ref_kernel")

    def run(self, input_data: Any) -> Any:
        return self._ref_fn(input_data)


class _CorrectnessChecker:
    def __init__(self, task: GpuModeTask) -> None:
        self._task = task
        module = _load_reference_module(task._cfg.task_dir)
        self._check_fn = getattr(module, "check_implementation")

    def check(self, output: Any, reference_output: Any) -> CheckResult:
        return CheckResult(passed=True, message="Checked during evaluation")


class _Scorer:
    def score(self, result: EvaluationResult) -> float:
        if not result.is_success():
            return 0.0
        metrics = result.get_metrics()
        latency = metrics.get("latency_ms")
        if latency and latency > 0:
            return 1.0 / latency
        return 0.0


class _FeedbackProvider:
    def for_codegen(self, rounds: Round | list[Round]) -> str:
        if isinstance(rounds, Round):
            rounds = [rounds]
        return "\n\n".join(r.result.get_log() for r in rounds)

    def for_world_model(
        self, rounds: Round | list[Round]
    ) -> list[dict[str, Any]]:
        if isinstance(rounds, Round):
            rounds = [rounds]
        return [r.result.get_metrics() for r in rounds]


class GpuModeTaskDefinition:
    """TaskDefinition implementation for GpuModeTask."""

    def __init__(self, task: GpuModeTask, language: str = "triton") -> None:
        self._task = task
        self._language = language
        self._impl_counter = 0
        self.name = task.name
        self.input_generator = _InputGenerator(task)
        self.reference_impl = _ReferenceImpl(task)
        self.correctness_checker = _CorrectnessChecker(task)
        self.scorer = _Scorer()
        self.feedback_provider = _FeedbackProvider()

    def get_prompt_text(self, context: dict[str, Any] | None = None) -> str:
        language = (context or {}).get("language", self._language)
        return self._task.get_definition_text(language)

    def get_test_cases(self) -> list[dict[str, Any]]:
        return [{"B": 2, "T": 4096, "D": 2048, "W": 4}]

    def create_implementation(self, llm_output: str) -> GpuModeImplementation:
        """Create Implementation from raw LLM output."""
        cleaned_code = strip_markdown_fences(llm_output)
        impl_name = f"{self.name}_r{self._impl_counter}"
        self._impl_counter += 1

        solution = Solution(
            name=impl_name,
            definition=self.name,
            author="search_v2",
            spec=BuildSpec(
                language=SupportedLanguages(self._language),
                target_hardware=[],
                entry_point="kernel.py::custom_kernel",
            ),
            sources=[SourceFile(path="kernel.py", content=cleaned_code)],
        )
        return GpuModeImplementation(solution)

    @property
    def task(self) -> GpuModeTask:
        return self._task
