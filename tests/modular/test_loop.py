"""Tests for modular loop module."""

from unittest.mock import Mock

from k_search.modular import SearchConfig, SearchResult, run_search


def make_eval_result_mock(is_success: bool = True, metrics: dict | None = None) -> Mock:
    """Create a mock EvaluationResult."""
    result = Mock()
    result.is_success.return_value = is_success
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

    task.create_implementation.side_effect = create_impl
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
    """Tests for build_prompt function."""

    def test_first_round_returns_task_prompt(self):
        from k_search.modular.prompts import build_prompt

        task = make_task_mock()
        task.get_prompt_text.return_value = "Task specification here"

        result = build_prompt(task, None)

        assert result == "Task specification here"
        task.get_prompt_text.assert_called_once()

    def test_with_round_includes_feedback(self):
        from k_search.modular import Round
        from k_search.modular.prompts import build_prompt

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


class TestCreateImplementation:
    """Tests for task.create_implementation via GpuModeTriMulTaskDefinition."""

    def test_creates_valid_implementation(self):
        from unittest.mock import MagicMock

        from k_search.modular.adapters.gpu_mode import GpuModeTriMulTaskDefinition

        mock_task = MagicMock()
        mock_task.name = "my_task"
        mock_task._cfg.task_dir.exists.return_value = False

        task_def = GpuModeTriMulTaskDefinition.__new__(GpuModeTriMulTaskDefinition)
        task_def._task = mock_task
        task_def._language = "triton"
        task_def._impl_counter = 5
        task_def.name = "my_task"

        impl = task_def.create_implementation("def kernel(): pass")

        assert impl.name == "my_task_r5"
        assert impl.content.definition == "my_task"
        assert impl.content.author == "search_v2"
        assert "def kernel(): pass" in impl.content.sources[0].content

    def test_increments_counter(self):
        from unittest.mock import MagicMock

        from k_search.modular.adapters.gpu_mode import GpuModeTriMulTaskDefinition

        mock_task = MagicMock()
        mock_task.name = "test_task"

        task_def = GpuModeTriMulTaskDefinition.__new__(GpuModeTriMulTaskDefinition)
        task_def._task = mock_task
        task_def._language = "triton"
        task_def._impl_counter = 0
        task_def.name = "test_task"

        impl1 = task_def.create_implementation("code1")
        impl2 = task_def.create_implementation("code2")

        assert impl1.name == "test_task_r0"
        assert impl2.name == "test_task_r1"
        assert task_def._impl_counter == 2


class TestStripMarkdownFences:
    """Tests for strip_markdown_fences function."""

    def test_no_fences_unchanged(self):
        from k_search.modular.prompts import strip_markdown_fences

        code = "def custom_kernel():\n    pass"
        assert strip_markdown_fences(code) == code

    def test_strips_python_fences(self):
        from k_search.modular.prompts import strip_markdown_fences

        code = "```python\ndef custom_kernel():\n    pass\n```"
        expected = "def custom_kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_strips_triton_fences(self):
        from k_search.modular.prompts import strip_markdown_fences

        code = "```triton\n@triton.jit\ndef kernel():\n    pass\n```"
        expected = "@triton.jit\ndef kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_strips_bare_fences(self):
        from k_search.modular.prompts import strip_markdown_fences

        code = "```\ndef custom_kernel():\n    pass\n```"
        expected = "def custom_kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_extracts_first_fenced_block(self):
        from k_search.modular.prompts import strip_markdown_fences

        code = (
            "Here's the code:\n```python\ndef kernel():\n    pass\n```\nAnd more text."
        )
        expected = "def kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_handles_empty_string(self):
        from k_search.modular.prompts import strip_markdown_fences

        assert strip_markdown_fences("") == ""
        assert strip_markdown_fences(None) is None

    def test_handles_unclosed_fence(self):
        from k_search.modular.prompts import strip_markdown_fences

        code = "```python\ndef kernel():\n    pass"
        # Should strip the leading fence at minimum
        result = strip_markdown_fences(code)
        assert "```" not in result
        assert "def kernel():" in result
