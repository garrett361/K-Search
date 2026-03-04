"""Tests for GPU mode reference benchmarking and speedup computation."""

import sys
from pathlib import Path

import pytest

from k_search.tasks.gpu_mode.evaluator import benchmark_reference
from k_search.tasks.gpu_mode_task import GpuModeTask

_TASK_DIR = (
    Path(__file__).parent.parent / "k_search" / "tasks" / "gpu_mode" / "causal_conv1d"
)
sys.path.insert(0, str(_TASK_DIR))


@pytest.mark.cuda_subprocess
@pytest.mark.cuda
class TestBenchmarkReference:
    def test_benchmark_reference_returns_latency(self):
        """Verify benchmark_reference() returns a valid GpuModeEvalSummary with latency."""
        summary = benchmark_reference(
            task_dir=_TASK_DIR, mode="benchmark", verbose=False
        )

        assert summary.status == "passed", (
            f"Reference benchmark failed: {summary.log_excerpt}"
        )
        assert summary.latency_ms is not None, "Reference benchmark returned no latency"
        assert summary.latency_ms > 0, f"Invalid latency: {summary.latency_ms}"


@pytest.mark.cuda_subprocess
@pytest.mark.cuda
class TestSpeedupFactor:
    def test_speedup_factor_populated(self):
        """Verify EvalResult.speedup_factor is numeric after run_benchmark()."""
        task = GpuModeTask(task_dir=_TASK_DIR, mode="benchmark")

        # Load baseline submission
        submission_path = _TASK_DIR / "submission.py"
        submission_code = submission_path.read_text()

        solution = task.make_solution_from_generated_code(
            cleaned_code=submission_code,
            raw_code=submission_code,
            round_num=0,
            model_name="test",
            target_gpu="H100",
            language="triton",
        )

        result = task.run_benchmark(solution=solution, round_num=0)

        assert result.status == "passed", (
            f"Baseline benchmark failed: {result.log_excerpt}"
        )
        assert result.reference_latency_ms is not None, (
            "reference_latency_ms not populated"
        )
        assert result.reference_latency_ms > 0
        assert result.speedup_factor is not None, "speedup_factor not populated"
        assert isinstance(result.speedup_factor, float)
        # Baseline should have speedup ~1.0 since it's the same as reference
        assert 0.5 < result.speedup_factor < 2.0, (
            f"Unexpected speedup: {result.speedup_factor}"
        )

    def test_reference_latency_cached(self):
        """Verify subsequent run_benchmark() calls use cached reference latency."""
        task = GpuModeTask(task_dir=_TASK_DIR, mode="benchmark")

        submission_path = _TASK_DIR / "submission.py"
        submission_code = submission_path.read_text()

        solution = task.make_solution_from_generated_code(
            cleaned_code=submission_code,
            raw_code=submission_code,
            round_num=0,
            model_name="test",
            target_gpu="H100",
            language="triton",
        )

        # First run benchmarks reference
        result1 = task.run_benchmark(solution=solution, round_num=0)
        ref_latency_1 = task._reference_latency_ms

        # Second run should use cached value
        result2 = task.run_benchmark(solution=solution, round_num=1)
        ref_latency_2 = task._reference_latency_ms

        assert ref_latency_1 is not None
        assert ref_latency_1 == ref_latency_2, "Reference latency should be cached"
        assert result1.reference_latency_ms == result2.reference_latency_ms
