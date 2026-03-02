"""Integration tests for CausalConv1dTask and generator interface."""

from __future__ import annotations

import pytest

from k_search.tasks.causal_conv1d import CausalConv1dTask
from k_search.tasks.task_base import BuildSpec, Solution, SupportedLanguages


class TestGeneratorInterface:
    """Verify methods generators call on the task exist and work."""

    @pytest.fixture
    def task(self):
        return CausalConv1dTask()

    def test_task_has_name(self, task):
        assert task.name == "causal_conv1d"

    def test_task_has_definition_text(self, task):
        spec = task.get_definition_text()
        assert isinstance(spec, str)
        assert len(spec) > 100
        assert "causal_conv1d" in spec.lower() or "conv" in spec.lower()

    def test_task_has_workloads(self, task):
        assert hasattr(task, "WORKLOADS")
        assert len(task.WORKLOADS) > 0
        for wl in task.WORKLOADS:
            assert "B" in wl
            assert "T" in wl
            assert "D" in wl

    def test_get_solution_baseline_returns_none(self, task):
        assert task.get_solution("baseline") is None
        assert task.get_solution("fla") is None

    def test_make_solution_from_generated_code(self, task):
        dummy_code = "# dummy kernel code"
        sol = task.make_solution_from_generated_code(
            cleaned_code=dummy_code,
            raw_code=dummy_code,
            round_num=1,
            model_name="test_model",
            target_gpu="h100",
            language="triton",
        )
        assert isinstance(sol, Solution)
        assert sol.name == "test_model_causal_conv1d_triton_r1"

    def test_solution_entry_point_matches_spec(self, task):
        dummy_code = "# kernel"
        sol = task.make_solution_from_generated_code(
            cleaned_code=dummy_code,
            raw_code=dummy_code,
            round_num=1,
            model_name="test",
            target_gpu="h100",
            language="triton",
        )
        assert sol.spec.entry_point == "submission.py::custom_kernel"

    def test_code_for_world_model_from_raw_string(self, task):
        code = task.code_for_world_model_from_raw(raw="some code", language="triton")
        assert code == "some code"

    def test_code_for_world_model_from_raw_solution(self, task):
        dummy_code = "# kernel code"
        sol = task.make_solution_from_generated_code(
            cleaned_code=dummy_code,
            raw_code=dummy_code,
            round_num=1,
            model_name="test",
            target_gpu="h100",
            language="triton",
        )
        code = task.code_for_world_model_from_raw(raw=sol, language="triton")
        assert code == dummy_code

    def test_get_config_for_logging(self, task):
        config = task.get_config_for_logging()
        assert "task" in config
        assert "dtype" in config
        assert config["task"] == "causal_conv1d"


class TestRunBenchmarkInterface:
    """Test run_benchmark returns expected structure without GPU."""

    @pytest.fixture
    def task(self):
        return CausalConv1dTask()

    def test_run_benchmark_fails_gracefully_with_bad_code(self, task):
        sol = task.make_solution_from_generated_code(
            cleaned_code="invalid python {{{",
            raw_code="invalid",
            round_num=1,
            model_name="test",
            target_gpu="h100",
            language="triton",
        )
        result = task.run_benchmark(solution=sol)
        assert result.status == "failed"
        assert result.log_excerpt is not None

    def test_run_benchmark_fails_with_empty_solution(self, task):
        sol = Solution(
            name="empty",
            definition="causal_conv1d",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["h100"],
                entry_point="submission.py::custom_kernel",
            ),
            sources=[],
            description="",
        )
        result = task.run_benchmark(solution=sol)
        assert result.status == "failed"
        assert "entry source" in result.log_excerpt.lower()


class TestEvalResultStructure:
    """Verify EvalResult has fields generators expect."""

    @pytest.fixture
    def task(self):
        return CausalConv1dTask()

    def test_failed_result_has_metrics(self, task):
        sol = task.make_solution_from_generated_code(
            cleaned_code="invalid",
            raw_code="invalid",
            round_num=1,
            model_name="test",
            target_gpu="h100",
            language="triton",
        )
        result = task.run_benchmark(solution=sol)
        assert hasattr(result, "status")
        assert hasattr(result, "log_excerpt")
        assert hasattr(result, "metrics")
