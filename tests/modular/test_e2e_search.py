"""E2E integration test: modular framework with real GPU evaluation."""

import pytest
from pathlib import Path

from k_search.modular import run_search, SearchConfig
from k_search.modular.adapters import GpuModeEvaluator, GpuModeTaskDefinition
from k_search.tasks.gpu_mode_task import GpuModeTask


CAUSAL_CONV1D_DIR = (
    Path(__file__).parent.parent.parent
    / "k_search"
    / "tasks"
    / "gpu_mode"
    / "causal_conv1d"
)


@pytest.mark.cuda
@pytest.mark.cuda_subprocess
class TestE2ESearch:
    """E2E tests validating modular loop with real GPU evaluation."""

    @pytest.fixture
    def task_dir(self) -> Path:
        return CAUSAL_CONV1D_DIR

    @pytest.fixture
    def valid_triton_code(self, task_dir: Path) -> str:
        submission_path = task_dir / "submission.py"
        return submission_path.read_text()

    def test_single_round_with_valid_code(self, task_dir: Path, valid_triton_code: str):
        """Run single search round with valid Triton code, verify score and metrics."""
        gpu_task = GpuModeTask(task_dir=task_dir)
        task_def = GpuModeTaskDefinition(gpu_task)
        evaluator = GpuModeEvaluator(gpu_task)

        def mock_llm(prompt: str) -> str:
            return valid_triton_code

        config = SearchConfig(max_rounds=1)
        result = run_search(task_def, evaluator, mock_llm, config)

        assert result.rounds_completed == 1
        assert result.impl is not None
        assert result.result is not None
        assert result.score > 0, "Valid code should produce positive score"

        metrics = result.result.get_metrics()
        assert "speedup_factor" in metrics or "latency_ms" in metrics

    def test_two_rounds_best_tracked(self, task_dir: Path, valid_triton_code: str):
        """Run two rounds, verify best result is tracked correctly."""
        gpu_task = GpuModeTask(task_dir=task_dir)
        task_def = GpuModeTaskDefinition(gpu_task)
        evaluator = GpuModeEvaluator(gpu_task)

        call_count = 0

        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return valid_triton_code

        config = SearchConfig(max_rounds=2)
        result = run_search(task_def, evaluator, mock_llm, config)

        assert result.rounds_completed == 2
        assert call_count == 2
        assert result.score > 0
        assert result.result is not None
