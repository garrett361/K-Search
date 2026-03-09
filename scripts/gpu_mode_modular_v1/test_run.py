"""Tests for GPU mode modular V1 entry point."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from k_search.modular import SearchConfig, SearchResult, Round
from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

from scripts.gpu_mode_modular_v1.run import (
    build_prompt,
    run_search,
    strip_markdown_fences,
    _build_round_metrics,
)


def make_eval_result_mock(succeeded: bool = True, metrics: dict | None = None) -> Mock:
    """Create a mock EvaluationResult."""
    result = Mock()
    result.succeeded.return_value = succeeded
    result.get_metrics.return_value = metrics or {"speedup_factor": 1.0}
    result.get_log.return_value = "test log"
    return result


def make_impl_mock(name: str = "test_impl") -> Mock:
    """Create a mock Implementation."""
    impl = Mock()
    impl.name = name
    return impl


def make_task_mock(name: str = "test_task") -> Mock:
    """Create a mock TaskDefinition."""
    task = Mock()
    task.name = name
    task.get_prompt_text.return_value = "Generate optimized kernel code"
    task.scorer.score.return_value = 0.5
    task.feedback_provider.for_codegen.return_value = "Previous attempt feedback"

    impl_counter = [0]

    def create_impl(llm_output: str) -> Mock:
        impl = make_impl_mock(f"{name}_r{impl_counter[0]}")
        impl_counter[0] += 1
        return impl

    task.create_impl.side_effect = create_impl
    return task


def make_evaluator_mock(result: Mock | None = None) -> Mock:
    """Create a mock Evaluator."""
    evaluator = Mock()
    evaluator.evaluate.return_value = result or make_eval_result_mock()
    return evaluator


def stub_llm(prompt: str) -> str:
    return "def custom_kernel(): pass"


class TestRunSearch:
    def test_completes_max_rounds(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=3)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.rounds_completed == 3
        assert evaluator.evaluate.call_count == 3
        assert task.scorer.score.call_count == 3

    def test_single_round(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=1)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.rounds_completed == 1
        assert evaluator.evaluate.call_count == 1

    def test_tracks_best_score_improvement(self):
        task = make_task_mock()
        scores = [0.1, 0.5, 0.3]
        task.scorer.score.side_effect = scores

        results = [
            make_eval_result_mock(metrics={"speedup_factor": 1.1}),
            make_eval_result_mock(metrics={"speedup_factor": 1.5}),
            make_eval_result_mock(metrics={"speedup_factor": 1.3}),
        ]
        evaluator = Mock()
        evaluator.evaluate.side_effect = results

        config = SearchConfig(max_rounds=3)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.score == 0.5
        assert result.result is not None
        assert result.result.get_metrics()["speedup_factor"] == 1.5

    def test_tracks_best_score_first_best(self):
        task = make_task_mock()
        scores = [0.9, 0.5, 0.3]
        task.scorer.score.side_effect = scores

        results = [
            make_eval_result_mock(metrics={"speedup_factor": 2.0}),
            make_eval_result_mock(metrics={"speedup_factor": 1.5}),
            make_eval_result_mock(metrics={"speedup_factor": 1.3}),
        ]
        evaluator = Mock()
        evaluator.evaluate.side_effect = results

        config = SearchConfig(max_rounds=3)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.score == 0.9
        assert result.result is not None
        assert result.result.get_metrics()["speedup_factor"] == 2.0

    def test_first_round_no_feedback(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        prompts_received = []

        def capture_llm(prompt: str) -> str:
            prompts_received.append(prompt)
            return "def custom_kernel(): pass"

        config = SearchConfig(max_rounds=1)

        run_search(task, evaluator, capture_llm, config)

        assert len(prompts_received) == 1
        assert "Previous attempt feedback" not in prompts_received[0]
        task.feedback_provider.for_codegen.assert_not_called()

    def test_subsequent_rounds_include_feedback(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        prompts_received = []

        def capture_llm(prompt: str) -> str:
            prompts_received.append(prompt)
            return "def custom_kernel(): pass"

        config = SearchConfig(max_rounds=3)

        run_search(task, evaluator, capture_llm, config)

        assert len(prompts_received) == 3
        assert "Previous attempt feedback" not in prompts_received[0]
        assert "Previous attempt feedback" in prompts_received[1]
        assert "Previous attempt feedback" in prompts_received[2]

    def test_returns_search_result_type(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=1)

        result = run_search(task, evaluator, stub_llm, config)

        assert isinstance(result, SearchResult)
        assert result.impl is not None
        assert result.result is not None
        assert result.score == 0.5

    def test_impl_naming_uses_task_name(self):
        task = make_task_mock(name="custom_task")
        captured_impls = []

        def capture_evaluate(impl, **kwargs):
            captured_impls.append(impl)
            return make_eval_result_mock()

        evaluator = Mock()
        evaluator.evaluate.side_effect = capture_evaluate
        config = SearchConfig(max_rounds=2)

        run_search(task, evaluator, stub_llm, config)

        assert len(captured_impls) == 2
        assert "custom_task_r0" in captured_impls[0].name
        assert "custom_task_r1" in captured_impls[1].name

    def test_zero_scores_handled(self):
        task = make_task_mock()
        task.scorer.score.side_effect = [0.0, 0.5, 0.0]

        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=3)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.score == 0.5
        assert result.rounds_completed == 3

    def test_llm_receives_task_prompt(self):
        task = make_task_mock()
        task.get_prompt_text.return_value = "Custom kernel specification"
        evaluator = make_evaluator_mock()
        prompts_received = []

        def capture_llm(prompt: str) -> str:
            prompts_received.append(prompt)
            return "def custom_kernel(): pass"

        config = SearchConfig(max_rounds=1)

        run_search(task, evaluator, capture_llm, config)

        assert "Custom kernel specification" in prompts_received[0]


class TestSearchConfig:
    def test_default_values(self):
        config = SearchConfig()
        assert config.max_rounds == 10
        assert config.timeout_secs is None

    def test_custom_max_rounds(self):
        config = SearchConfig(max_rounds=5)
        assert config.max_rounds == 5

    def test_custom_timeout(self):
        config = SearchConfig(timeout_secs=300)
        assert config.timeout_secs == 300


class TestSearchResult:
    def test_default_rounds_completed(self):
        result = SearchResult(impl=None, score=0.0, result=None)
        assert result.rounds_completed == 0

    def test_all_fields(self):
        mock_impl = Mock()
        mock_eval_result = Mock()
        result = SearchResult(
            impl=mock_impl,
            score=0.75,
            result=mock_eval_result,
            rounds_completed=5,
        )
        assert result.impl is mock_impl
        assert result.score == 0.75
        assert result.result is mock_eval_result
        assert result.rounds_completed == 5


class TestBuildPrompt:
    def test_first_round_returns_task_prompt(self):
        task = make_task_mock()
        task.get_prompt_text.return_value = "Task specification here"

        result = build_prompt(task, None)

        assert result == "Task specification here"
        task.get_prompt_text.assert_called_once()

    def test_with_round_includes_feedback(self):
        task = make_task_mock()
        task.get_prompt_text.return_value = "Task spec"
        task.feedback_provider.for_codegen.return_value = "Error on line 42"

        mock_impl = Mock()
        mock_result = make_eval_result_mock()
        round_ = Round(
            impl=mock_impl,
            result=mock_result,
            prompt="test",
            llm_response="test",
            prompt_tokens=0,
            completion_tokens=0,
            duration_secs=0.0,
            score=0.0,
        )

        result = build_prompt(task, round_)

        assert "Task spec" in result
        assert "Error on line 42" in result
        task.feedback_provider.for_codegen.assert_called_once_with(round_)


class TestCreateImpl:
    def test_increments_counter(self):
        mock_task = MagicMock()
        mock_task.name = "test_task"

        with patch(
            "k_search.modular.adapters.gpu_mode.task_definition._load_reference_module"
        ):
            task_def = GpuModeTriMulTaskDefinition(mock_task)

        assert task_def.create_impl("code1").name == "test_task_r0"
        assert task_def.create_impl("code2").name == "test_task_r1"
        assert task_def._impl_counter == 2


class TestStripMarkdownFences:
    def test_no_fences_unchanged(self):
        code = "def custom_kernel():\n    pass"
        assert strip_markdown_fences(code) == code

    def test_strips_python_fences(self):
        code = "```python\ndef custom_kernel():\n    pass\n```"
        expected = "def custom_kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_strips_triton_fences(self):
        code = "```triton\n@triton.jit\ndef kernel():\n    pass\n```"
        expected = "@triton.jit\ndef kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_strips_bare_fences(self):
        code = "```\ndef custom_kernel():\n    pass\n```"
        expected = "def custom_kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_extracts_first_fenced_block(self):
        code = (
            "Here's the code:\n```python\ndef kernel():\n    pass\n```\nAnd more text."
        )
        expected = "def kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_handles_empty_string(self):
        assert strip_markdown_fences("") == ""
        assert strip_markdown_fences(None) is None

    def test_handles_unclosed_fence(self):
        code = "```python\ndef kernel():\n    pass"
        result = strip_markdown_fences(code)
        assert result is not None
        assert "```" not in result
        assert "def kernel():" in result


class TestBuildRoundMetrics:
    @pytest.fixture
    def mock_eval_result(self):
        result = Mock()
        result.succeeded.return_value = True
        result.get_metrics.return_value = {"speedup_factor": 1.5}
        return result

    def test_builds_correct_structure(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {"speedup_factor": 1.2}
        metrics = _build_round_metrics(
            round_time_secs=3.5,
            score=0.6,
            result=mock_eval_result,
            best_score=0.7,
            prompt_toks=200,
            completion_toks=100,
            cumulative_prompt_toks=800,
            cumulative_completion_toks=400,
        )

        assert metrics == {
            "round_time_secs": 3.5,
            "score": 0.6,
            "succeeded": 1,
            "best_score": 0.7,
            "toks/prompt": 200,
            "toks/completion": 100,
            "toks/total": 300,
            "toks/cumulative_prompt": 800,
            "toks/cumulative_completion": 400,
            "toks/cumulative_total": 1200,
            "speedup_factor": 1.2,
        }

    def test_is_success_converts_to_int(self, mock_eval_result):
        mock_eval_result.succeeded.return_value = False
        metrics = _build_round_metrics(
            round_time_secs=5.0,
            score=0.0,
            result=mock_eval_result,
            best_score=0.0,
            prompt_toks=100,
            completion_toks=50,
            cumulative_prompt_toks=100,
            cumulative_completion_toks=50,
        )

        assert metrics["succeeded"] == 0

    def test_includes_numeric_eval_metrics(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {
            "speedup_factor": 2.5,
            "latency_ms": 10.3,
            "memory_mb": 512,
        }
        metrics = _build_round_metrics(
            round_time_secs=5.0,
            score=0.5,
            result=mock_eval_result,
            best_score=0.5,
            prompt_toks=100,
            completion_toks=50,
            cumulative_prompt_toks=100,
            cumulative_completion_toks=50,
        )

        assert metrics["speedup_factor"] == 2.5
        assert metrics["latency_ms"] == 10.3
        assert metrics["memory_mb"] == 512

    def test_excludes_non_numeric_and_bool_metrics(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {
            "speedup_factor": 1.5,
            "status": "success",
            "error_msg": None,
            "passed": True,
            "has_errors": False,
        }
        metrics = _build_round_metrics(
            round_time_secs=5.0,
            score=0.5,
            result=mock_eval_result,
            best_score=0.5,
            prompt_toks=100,
            completion_toks=50,
            cumulative_prompt_toks=100,
            cumulative_completion_toks=50,
        )

        assert metrics["speedup_factor"] == 1.5
        assert "status" not in metrics
        assert "error_msg" not in metrics
        assert "passed" not in metrics
        assert "has_errors" not in metrics

    def test_returns_expected_keys(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {"speedup_factor": 1.2}
        metrics = _build_round_metrics(
            round_time_secs=3.5,
            score=0.6,
            result=mock_eval_result,
            best_score=0.7,
            prompt_toks=200,
            completion_toks=100,
            cumulative_prompt_toks=800,
            cumulative_completion_toks=400,
        )

        expected_keys = {
            "round_time_secs",
            "score",
            "succeeded",
            "best_score",
            "toks/prompt",
            "toks/completion",
            "toks/total",
            "toks/cumulative_prompt",
            "toks/cumulative_completion",
            "toks/cumulative_total",
            "speedup_factor",
        }
        assert set(metrics.keys()) == expected_keys


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
        gpu_task = GpuModeTriMulTask(task_dir=task_dir)
        task_def = GpuModeTriMulTaskDefinition(gpu_task)
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
        gpu_task = GpuModeTriMulTask(task_dir=task_dir)
        task_def = GpuModeTriMulTaskDefinition(gpu_task)
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
