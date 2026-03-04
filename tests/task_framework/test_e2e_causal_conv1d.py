"""E2E integration test: task_framework with causal_conv1d task."""

import pytest
from pathlib import Path

from k_search.tasks.gpu_mode_task import GpuModeTask
from k_search.task_framework import (
    GpuModeAdapter,
    GpuModeEvaluationResult,
    GpuModeSolutionArtifact,
    EvalOutcome,
)


CAUSAL_CONV1D_DIR = (
    Path(__file__).parent.parent.parent
    / "k_search"
    / "tasks"
    / "gpu_mode"
    / "causal_conv1d"
)


class TestTaskFrameworkE2E:
    """E2E tests validating task_framework works with real causal_conv1d task."""

    def test_adapter_loads_causal_conv1d(self):
        """Verify adapter can wrap causal_conv1d task."""
        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        assert "causal_conv1d" in adapter.name
        assert adapter.input_generator is not None
        assert adapter.reference_impl is not None

    @pytest.mark.cuda
    def test_input_generator_produces_valid_data(self):
        """Verify input generator produces valid tensors."""
        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        data = adapter.input_generator.generate(
            params={"B": 2, "T": 64, "D": 32, "W": 4},
            seed=42,
        )

        assert isinstance(data, tuple)
        assert len(data) == 3
        x, weight, config = data
        assert x.shape == (2, 64, 32)

    @pytest.mark.cuda
    def test_reference_impl_runs(self):
        """Verify reference implementation runs."""
        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        data = adapter.input_generator.generate(
            params={"B": 2, "T": 64, "D": 32, "W": 4},
            seed=42,
        )
        output = adapter.reference_impl.run(data)

        assert output.shape == (2, 64, 32)

    def test_prompt_text_contains_spec(self):
        """Verify prompt text includes task specification."""
        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        prompt = adapter.get_prompt_text()

        assert "custom_kernel" in prompt
        assert "causal" in prompt.lower() or "conv" in prompt.lower()

    def test_scorer_works_with_wrapped_result(self):
        """Verify scorer works with GpuModeEvaluationResult."""
        from k_search.tasks.task_base import EvalResult

        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        result = GpuModeEvaluationResult(EvalResult(status="passed", latency_ms=2.0))
        score = adapter.scorer.score(result)

        assert score == 0.5  # 1/2.0

    def test_feedback_provider_formats_outcome(self):
        """Verify feedback provider formats EvalOutcome correctly."""
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
        outcome = EvalOutcome(
            solution=GpuModeSolutionArtifact(sol),
            result=GpuModeEvaluationResult(result),
        )

        codegen_feedback = adapter.feedback_provider.for_codegen(outcome)
        assert "CUDA error" in codegen_feedback

        wm_metrics = adapter.feedback_provider.for_world_model(outcome)
        assert len(wm_metrics) == 1
        assert wm_metrics[0]["status"] == "failed"

    @pytest.mark.cuda
    @pytest.mark.cuda_subprocess
    def test_full_eval_workflow(self):
        """Full workflow: generate input, run reference, check baseline."""
        task = GpuModeTask(task_dir=CAUSAL_CONV1D_DIR)
        adapter = GpuModeAdapter(task)

        # Generate input
        data = adapter.input_generator.generate(
            params={"B": 2, "T": 64, "D": 32, "W": 4},
            seed=42,
        )

        # Run reference
        ref_output = adapter.reference_impl.run(data)
        assert ref_output.shape == (2, 64, 32)

        # Verify prompt is usable
        prompt = adapter.get_prompt_text(context={"language": "triton"})
        assert len(prompt) > 500
