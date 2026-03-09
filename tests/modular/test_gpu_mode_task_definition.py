"""Tests for GpuModeTriMulTaskDefinition."""

import pytest
from pathlib import Path

from k_search.tasks.gpu_mode_task import GpuModeTriMulTask


CAUSAL_CONV1D_DIR = (
    Path(__file__).parent.parent.parent
    / "k_search"
    / "tasks"
    / "gpu_mode"
    / "causal_conv1d"
)


@pytest.mark.cuda
class TestGpuModeTriMulTaskDefinitionConstruction:
    def test_wraps_gpu_mode_task(self):
        from k_search.modular.adapters.gpu_mode import GpuModeTriMulTaskDefinition

        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        assert task_def.name == task.name

    def test_get_prompt_text_returns_spec(self):
        from k_search.modular.adapters.gpu_mode import GpuModeTriMulTaskDefinition

        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        prompt = task_def.get_prompt_text()
        assert "custom_kernel" in prompt
        assert len(prompt) > 100

    def test_get_prompt_text_respects_language(self):
        from k_search.modular.adapters.gpu_mode import GpuModeTriMulTaskDefinition

        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        triton_prompt = task_def.get_prompt_text(context={"language": "triton"})
        assert "triton" in triton_prompt.lower() or "custom_kernel" in triton_prompt


@pytest.mark.cuda
class TestGpuModeTriMulTaskDefinitionScorer:
    def test_scorer_returns_positive_for_passed(self):
        from k_search.modular.adapters.gpu_mode import (
            GpuModeEvaluationResult,
            GpuModeTriMulTaskDefinition,
        )
        from k_search.tasks.task_base import EvalResult

        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        result = GpuModeEvaluationResult(EvalResult(status="passed", latency_ms=1.0))
        score = task_def.scorer.score(result)

        assert score > 0

    def test_scorer_returns_zero_for_failed(self):
        from k_search.modular.adapters.gpu_mode import (
            GpuModeEvaluationResult,
            GpuModeTriMulTaskDefinition,
        )
        from k_search.tasks.task_base import EvalResult

        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        result = GpuModeEvaluationResult(EvalResult(status="failed"))
        score = task_def.scorer.score(result)

        assert score == 0.0


@pytest.mark.cuda
class TestGpuModeTriMulTaskDefinitionFeedbackProvider:
    def test_for_codegen_returns_log(self):
        from k_search.modular.adapters.gpu_mode import (
            GpuModeEvaluationResult,
            GpuModeImplementation,
            GpuModeTriMulTaskDefinition,
        )
        from k_search.modular import Round
        from k_search.tasks.task_base import (
            BuildSpec,
            EvalResult,
            Solution,
            SourceFile,
            SupportedLanguages,
        )

        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        sol = Solution(
            name="test",
            definition="def",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["H100"],
                entry_point="submission.py::custom_kernel",
            ),
            sources=[SourceFile(path="submission.py", content="code")],
        )
        result = EvalResult(status="failed", log_excerpt="Error: index out of bounds")
        round_ = Round(
            impl=GpuModeImplementation(sol),
            result=GpuModeEvaluationResult(result),
            prompt="test",
            llm_response="test",
            prompt_tokens=0,
            completion_tokens=0,
            duration_secs=0.0,
            score=0.0,
        )

        feedback = task_def.feedback_provider.for_codegen(round_)
        assert "index out of bounds" in feedback

    def test_for_world_model_returns_metrics(self):
        from k_search.modular.adapters.gpu_mode import (
            GpuModeEvaluationResult,
            GpuModeImplementation,
            GpuModeTriMulTaskDefinition,
        )
        from k_search.modular import Round
        from k_search.tasks.task_base import (
            BuildSpec,
            EvalResult,
            Solution,
            SourceFile,
            SupportedLanguages,
        )

        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        sol = Solution(
            name="test",
            definition="def",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["H100"],
                entry_point="submission.py::custom_kernel",
            ),
            sources=[SourceFile(path="submission.py", content="code")],
        )
        result = EvalResult(status="passed", latency_ms=1.5)
        round_ = Round(
            impl=GpuModeImplementation(sol),
            result=GpuModeEvaluationResult(result),
            prompt="test",
            llm_response="test",
            prompt_tokens=0,
            completion_tokens=0,
            duration_secs=0.0,
            score=1.5,
        )

        metrics_list = task_def.feedback_provider.for_world_model(round_)
        assert len(metrics_list) == 1
        assert "latency_ms" in metrics_list[0]
