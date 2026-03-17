"""E2E integration test: modular framework with causal_conv1d task."""

import pytest
from pathlib import Path

from k_search.modular import (
    Round,
    GpuModeEvaluationResult,
    GpuModeImplementation,
    GpuModeTriMulTaskDefinition,
)
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask


CAUSAL_CONV1D_DIR = (
    Path(__file__).parent.parent.parent
    / "k_search"
    / "tasks"
    / "gpu_mode"
    / "causal_conv1d"
)


class TestModularE2E:
    """E2E tests validating modular framework works with real causal_conv1d task."""

    def test_task_definition_loads_causal_conv1d(self):
        """Verify task definition can wrap causal_conv1d task."""
        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        assert "causal_conv1d" in task_def.name
        assert task_def.input_generator is not None
        assert task_def.reference_impl is not None

    @pytest.mark.cuda
    @pytest.mark.cuda_subprocess
    def test_input_generator_produces_valid_data(self):
        """Verify input generator produces valid tensors."""
        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        data = task_def.input_generator.generate(
            params={"B": 2, "T": 64, "D": 32, "W": 4},
            seed=42,
        )

        assert isinstance(data, tuple)
        assert len(data) == 3
        x, weight, config = data
        assert x.shape == (2, 64, 32)

    @pytest.mark.cuda
    @pytest.mark.cuda_subprocess
    def test_reference_impl_runs(self):
        """Verify reference implementation runs."""
        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        data = task_def.input_generator.generate(
            params={"B": 2, "T": 64, "D": 32, "W": 4},
            seed=42,
        )
        output = task_def.reference_impl.run(data)

        assert output.shape == (2, 64, 32)

    def test_prompt_text_contains_spec(self):
        """Verify prompt text includes task specification."""
        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        prompt = task_def.get_prompt_text()

        assert "custom_kernel" in prompt
        assert "causal" in prompt.lower() or "conv" in prompt.lower()

    def test_scorer_works_with_wrapped_result(self):
        """Verify scorer works with GpuModeEvaluationResult."""
        from k_search.tasks.task_base import EvalResult

        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        result = GpuModeEvaluationResult(EvalResult(status="passed", latency_ms=2.0))
        score = task_def.scorer.score(result)

        assert score == 0.5  # 1/2.0

    def test_feedback_provider_formats_round(self):
        """Verify feedback provider formats Round correctly."""
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
            definition="causal_conv1d",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["H100"],
                entry_point="submission.py::custom_kernel",
            ),
            sources=[
                SourceFile(
                    path="submission.py", content="def custom_kernel(data): pass"
                )
            ],
        )
        result = EvalResult(
            status="failed",
            log_excerpt="RuntimeError: CUDA error",
        )
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

        codegen_feedback = task_def.feedback_provider.for_codegen(round_)
        assert "CUDA error" in codegen_feedback

        wm_metrics = task_def.feedback_provider.for_world_model(round_)
        assert len(wm_metrics) == 1
        assert wm_metrics[0]["status"] == "failed"

    @pytest.mark.cuda
    @pytest.mark.cuda_subprocess
    def test_full_eval_workflow(self):
        """Full workflow: generate input, run reference, check baseline."""
        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        data = task_def.input_generator.generate(
            params={"B": 2, "T": 64, "D": 32, "W": 4},
            seed=42,
        )

        ref_output = task_def.reference_impl.run(data)
        assert ref_output.shape == (2, 64, 32)

        prompt = task_def.get_prompt_text(context={"language": "triton"})
        assert len(prompt) > 500
