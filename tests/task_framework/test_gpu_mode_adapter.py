"""Tests for GpuModeAdapter."""

import pytest
from pathlib import Path

from k_search.tasks.gpu_mode_task import GpuModeTask


CAUSAL_CONV1D_DIR = Path(__file__).parent.parent.parent / "k_search" / "tasks" / "gpu_mode" / "causal_conv1d"


@pytest.mark.cuda
class TestGpuModeAdapterConstruction:
    def test_adapter_wraps_gpu_mode_task(self):
        from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter

        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        assert adapter.name == task.name

    def test_adapter_has_required_components(self):
        from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter

        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        assert adapter.input_generator is not None
        assert adapter.correctness_checker is not None
        assert adapter.scorer is not None
        assert adapter.feedback_provider is not None
        assert adapter.reference_impl is not None

    def test_get_prompt_text_returns_spec(self):
        from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter

        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        prompt = adapter.get_prompt_text()
        assert "custom_kernel" in prompt
        assert len(prompt) > 100

    def test_get_prompt_text_respects_language(self):
        from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter

        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        triton_prompt = adapter.get_prompt_text(context={"language": "triton"})
        assert "triton" in triton_prompt.lower() or "custom_kernel" in triton_prompt


@pytest.mark.cuda
class TestGpuModeAdapterScorer:
    def test_scorer_returns_positive_for_passed(self):
        from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter
        from k_search.task_framework.adapters.wrappers import GpuModeEvaluationResult
        from k_search.tasks.task_base import EvalResult

        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        result = GpuModeEvaluationResult(EvalResult(status="passed", latency_ms=1.0))
        score = adapter.scorer.score(result)

        assert score > 0

    def test_scorer_returns_negative_for_failed(self):
        from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter
        from k_search.task_framework.adapters.wrappers import GpuModeEvaluationResult
        from k_search.tasks.task_base import EvalResult

        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        result = GpuModeEvaluationResult(EvalResult(status="failed"))
        score = adapter.scorer.score(result)

        assert score < 0


@pytest.mark.cuda
class TestGpuModeAdapterFeedbackProvider:
    def test_for_codegen_returns_log(self):
        from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter
        from k_search.task_framework.adapters.wrappers import (
            GpuModeEvaluationResult,
            GpuModeSolutionArtifact,
        )
        from k_search.task_framework.types import EvalOutcome
        from k_search.tasks.task_base import (
            EvalResult,
            Solution,
            BuildSpec,
            SourceFile,
            SupportedLanguages,
        )

        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

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
        outcome = EvalOutcome(
            solution=GpuModeSolutionArtifact(sol),
            result=GpuModeEvaluationResult(result),
        )

        feedback = adapter.feedback_provider.for_codegen(outcome)
        assert "index out of bounds" in feedback

    def test_for_world_model_returns_metrics(self):
        from k_search.task_framework.adapters.gpu_mode import GpuModeAdapter
        from k_search.task_framework.adapters.wrappers import (
            GpuModeEvaluationResult,
            GpuModeSolutionArtifact,
        )
        from k_search.task_framework.types import EvalOutcome
        from k_search.tasks.task_base import (
            EvalResult,
            Solution,
            BuildSpec,
            SourceFile,
            SupportedLanguages,
        )

        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

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
        outcome = EvalOutcome(
            solution=GpuModeSolutionArtifact(sol),
            result=GpuModeEvaluationResult(result),
        )

        metrics_list = adapter.feedback_provider.for_world_model(outcome)
        assert len(metrics_list) == 1
        assert "latency_ms" in metrics_list[0]
