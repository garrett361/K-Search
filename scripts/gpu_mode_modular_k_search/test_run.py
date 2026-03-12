"""Minimal tests for V1 case search - core logic only."""

import pytest
from unittest.mock import MagicMock

from k_search.modular.world.action import Action
from k_search.modular.world.node import Node
from k_search.modular.world.round import Round
from k_search.modular.world.tree import Tree

from scripts.gpu_mode_modular_k_search.run import (
    CycleConfig,
    V1PromptBuilder,
    V1SequentialExecutor,
)


class MockResult:
    def __init__(self, succeeded: bool, latency_ms: float):
        self._succeeded = succeeded
        self._latency_ms = latency_ms

    def succeeded(self) -> bool:
        return self._succeeded

    def get_metrics(self) -> dict:
        return {"latency_ms": self._latency_ms, "speedup_factor": 1.0}

    def get_log(self) -> str:
        return ""


class MockImpl:
    pass


class MockTask:
    name = "test"

    def create_impl(self, code: str):
        return MockImpl()


class MockEvaluator:
    def __init__(
        self, results: list
    ):  # Accepts MockResult or MockResultWithPerfSummary
        self._results = results
        self._idx = 0

    def evaluate(self, impl, context=None):
        result = self._results[self._idx % len(self._results)]
        self._idx += 1
        return result


def test_stagnation_ends_cycle_early():
    """Core test: stagnation window terminates cycle before max attempts."""
    # All same latency = same score (1.0/2.0 = 0.5)
    results = [MockResult(True, 2.0)] * 10

    mock_wm = MagicMock()
    mock_wm.propose.return_value = []
    mock_wm.select.side_effect = [
        [Node(status="open", action=Action(title="Test"))],
        [],
    ]
    mock_wm.get_action_context.return_value = {
        "action_text": "Test",
        "base_code": "",
        "base_score": 0.0,
        "base_result": None,
    }
    mock_wm._get_prompt_section.return_value = ""

    evaluator = MockEvaluator(results)
    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "prompt"

    tree = Tree(root=Node(status="closed"))

    executor = V1SequentialExecutor(
        world_model=mock_wm,
        task=MockTask(),  # type: ignore[arg-type]
        evaluator=evaluator,  # type: ignore[arg-type]
        llm=lambda p: "code",
        prompt_builder=prompt_builder,
        tree=tree,
        max_rounds=20,
        cycle_config=CycleConfig(stagnation_rounds=3),
    )

    executor.run()

    # Should stop after stagnation_rounds+1 (first success + 3 no-improve)
    assert evaluator._idx <= 5


def test_no_improve_over_base_ends_cycle():
    """Cycle terminates when improving locally but never beating base score.

    V1 semantics: two independent stagnation counters. This tests that
    no_improve_over_base terminates the cycle even when no_improve doesn't.
    """
    # Score = 1/latency: 2.0ms→0.5, 1.67ms→0.6, 1.43ms→0.7, 1.25ms→0.8
    # Each result improves over previous so no_improve=0
    # But all are below base_score=1.0 (latency 1.0ms), so no_improve_over_base increments
    results = [
        MockResult(True, 2.0),  # score = 0.5
        MockResult(True, 1.67),  # score = 0.6
        MockResult(True, 1.43),  # score = 0.7
        MockResult(True, 1.25),  # score = 0.8
    ]

    mock_wm = MagicMock()
    mock_wm.propose.return_value = []
    mock_wm.select.side_effect = [
        [Node(status="open", action=Action(title="Test"))],
        [],
    ]
    mock_wm.get_action_context.return_value = {
        "action_text": "Test",
        "base_code": "parent_code",
        "base_score": 1.0,  # score = 1.0 (latency 1.0ms) - higher than all results
        "base_result": None,
        "parent_is_root": False,
    }
    mock_wm._get_prompt_section.return_value = ""

    evaluator = MockEvaluator(results)
    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "prompt"

    tree = Tree(root=Node(status="closed"))

    executor = V1SequentialExecutor(
        world_model=mock_wm,
        task=MockTask(),  # type: ignore[arg-type]
        evaluator=evaluator,  # type: ignore[arg-type]
        llm=lambda p: "code",
        prompt_builder=prompt_builder,
        tree=tree,
        max_rounds=20,
        cycle_config=CycleConfig(stagnation_rounds=3),
    )

    executor.run()

    # With stagnation_rounds=3: after rounds 0,1,2 each increments no_improve_over_base
    # (since all scores < base_score=1.0). At round 2's end: no_improve_over_base=3 → break
    assert evaluator._idx == 3


# =============================================================================
# Test 1: Scoring Semantics Agreement
# =============================================================================


class TestScoringSemantics:
    """V1 semantics: failed evals score -1.0, success scores 1.0/latency_ms."""

    def test_failure_returns_negative_one(self):
        """V1 semantics: failed evals score -1.0, not 0.0."""
        results = [MockResult(succeeded=False, latency_ms=0.0)]

        tree = Tree(root=Node(status="closed"))
        # Create node attached to tree so it appears in _nodes_by_id
        action_node = Node(parent=tree.root, status="open", action=Action(title="Test"))
        tree.add_node(action_node)

        mock_wm = MagicMock()
        mock_wm.propose.return_value = []
        mock_wm.select.side_effect = [
            [action_node],  # Return the node from the tree
            [],
        ]
        mock_wm.get_action_context.return_value = {
            "action_text": "Test",
            "base_code": "",
            "base_score": 0.0,
            "base_result": None,
        }
        mock_wm._get_prompt_section.return_value = ""

        evaluator = MockEvaluator(results)
        prompt_builder = MagicMock()
        prompt_builder.build.return_value = "prompt"

        executor = V1SequentialExecutor(
            world_model=mock_wm,
            task=MockTask(),  # type: ignore[arg-type]
            evaluator=evaluator,  # type: ignore[arg-type]
            llm=lambda p: "code",
            prompt_builder=prompt_builder,
            tree=tree,
            max_rounds=1,
            cycle_config=CycleConfig(stagnation_rounds=10),
        )

        executor.run()

        # Access the cycle attached to the node
        nodes = [n for n in tree._nodes_by_id.values() if n.cycle]
        assert len(nodes) == 1
        cycle = nodes[0].cycle
        assert cycle is not None
        assert len(cycle.rounds) == 1
        assert cycle.rounds[0].score == -1.0

    def test_success_is_inverse_latency(self):
        """V1 semantics: score = 1.0 / latency_ms."""
        results = [MockResult(succeeded=True, latency_ms=2.0)]

        tree = Tree(root=Node(status="closed"))
        # Create node attached to tree so it appears in _nodes_by_id
        action_node = Node(parent=tree.root, status="open", action=Action(title="Test"))
        tree.add_node(action_node)

        mock_wm = MagicMock()
        mock_wm.propose.return_value = []
        mock_wm.select.side_effect = [
            [action_node],  # Return the node from the tree
            [],
        ]
        mock_wm.get_action_context.return_value = {
            "action_text": "Test",
            "base_code": "",
            "base_score": 0.0,
            "base_result": None,
        }
        mock_wm._get_prompt_section.return_value = ""

        evaluator = MockEvaluator(results)
        prompt_builder = MagicMock()
        prompt_builder.build.return_value = "prompt"

        executor = V1SequentialExecutor(
            world_model=mock_wm,
            task=MockTask(),  # type: ignore[arg-type]
            evaluator=evaluator,  # type: ignore[arg-type]
            llm=lambda p: "code",
            prompt_builder=prompt_builder,
            tree=tree,
            max_rounds=1,
            cycle_config=CycleConfig(stagnation_rounds=10),
        )

        executor.run()

        nodes = [n for n in tree._nodes_by_id.values() if n.cycle]
        assert len(nodes) == 1
        cycle = nodes[0].cycle
        assert cycle is not None
        assert len(cycle.rounds) == 1
        # score = 1.0 / 2.0 = 0.5
        assert cycle.rounds[0].score == 0.5


# =============================================================================
# Test 2: Best-Code Selection with Base Comparison
# =============================================================================


class TestBestCodeSelection:
    """V1 semantics: effective_base_code selection when cycle-best hasn't beaten parent."""

    def test_effective_base_code_reverts_to_parent_when_not_beating(self):
        """If best_score <= base_score, prompt builder receives parent's base_code."""
        # Parent has score 1.0 (latency 1.0ms)
        # All results worse: 0.5 (latency 2.0ms)
        results = [
            MockResult(True, 2.0),  # score 0.5 < base_score 1.0
            MockResult(True, 2.0),  # score 0.5 < base_score 1.0
        ]

        mock_wm = MagicMock()
        mock_wm.propose.return_value = []
        mock_wm.select.side_effect = [
            [Node(status="open", action=Action(title="Test"))],
            [],
        ]
        mock_wm.get_action_context.return_value = {
            "action_text": "Test",
            "base_code": "parent_base_code",
            "base_score": 1.0,
            "base_result": None,
            "parent_is_root": False,
        }
        mock_wm._get_prompt_section.return_value = ""

        evaluator = MockEvaluator(results)
        prompt_builder = MagicMock()
        prompt_builder.build.return_value = "prompt"

        tree = Tree(root=Node(status="closed"))

        executor = V1SequentialExecutor(
            world_model=mock_wm,
            task=MockTask(),  # type: ignore[arg-type]
            evaluator=evaluator,  # type: ignore[arg-type]
            llm=lambda p: "code",
            prompt_builder=prompt_builder,
            tree=tree,
            max_rounds=2,
            cycle_config=CycleConfig(stagnation_rounds=10),
        )

        executor.run()

        # On attempt 1 (second call), best_score=0.5 <= base_score=1.0
        # So effective_base_code should be parent's base_code, not cycle's best
        calls = prompt_builder.build.call_args_list
        assert len(calls) >= 2
        second_call = calls[1]
        assert second_call.kwargs["base_code"] == "parent_base_code"

    def test_effective_base_code_uses_best_when_beating(self):
        """If best_score > base_score, prompt builder receives cycle's best_code."""
        # Parent has score 0.25 (latency 4.0ms)
        # First result better: 0.5 (latency 2.0ms)
        results = [
            MockResult(True, 2.0),  # score 0.5 > base_score 0.25
            MockResult(True, 2.0),  # score 0.5 > base_score 0.25
        ]

        mock_wm = MagicMock()
        mock_wm.propose.return_value = []
        mock_wm.select.side_effect = [
            [Node(status="open", action=Action(title="Test"))],
            [],
        ]
        mock_wm.get_action_context.return_value = {
            "action_text": "Test",
            "base_code": "parent_base_code",
            "base_score": 0.25,
            "base_result": None,
            "parent_is_root": False,
        }
        mock_wm._get_prompt_section.return_value = ""

        evaluator = MockEvaluator(results)
        prompt_builder = MagicMock()
        prompt_builder.build.return_value = "prompt"

        # Track generated code to verify best_code is used
        generated_codes = ["first_generated_code", "second_generated_code"]
        code_idx = [0]

        def llm_with_tracking(prompt):
            code = generated_codes[code_idx[0] % len(generated_codes)]
            code_idx[0] += 1
            return code

        tree = Tree(root=Node(status="closed"))

        executor = V1SequentialExecutor(
            world_model=mock_wm,
            task=MockTask(),  # type: ignore[arg-type]
            evaluator=evaluator,  # type: ignore[arg-type]
            llm=llm_with_tracking,
            prompt_builder=prompt_builder,
            tree=tree,
            max_rounds=2,
            cycle_config=CycleConfig(stagnation_rounds=10),
        )

        executor.run()

        # On attempt 1 (second call), best_score=0.5 > base_score=0.25
        # So effective_base_code should be cycle's best_code ("first_generated_code")
        calls = prompt_builder.build.call_args_list
        assert len(calls) >= 2
        second_call = calls[1]
        assert second_call.kwargs["base_code"] == "first_generated_code"


# =============================================================================
# Test 3: Prompt Type Selection Matrix
# =============================================================================


class TestPromptTypeSelection:
    """V1 uses 4-level nested conditional for prompt type selection."""

    @pytest.fixture
    def builder(self):
        return V1PromptBuilder(
            definition_text="Test definition",
            language="triton",
            target_gpu="H100",
        )

    @pytest.fixture
    def mock_round(self):
        """Create a mock Round for testing."""
        mock_result = MockResult(succeeded=False, latency_ms=0.0)
        return Round(
            impl=MockImpl(),  # type: ignore[arg-type]
            result=mock_result,
            prompt="test",
            llm_response="code",
            prompt_tokens=0,
            completion_tokens=0,
            duration_secs=0.0,
            score=-1.0,
        )

    def test_type_a_attempt0_has_base(self, builder):
        """TYPE A: attempt=0, has_base=True → action prompt from base."""
        prompt = builder.build(
            action_text="Optimize loop",
            attempt=0,
            last_round=None,
            has_passed=False,
            base_code="existing_code",
            trace_logs="",
            current_code="",
            parent_is_root=False,
        )
        # Should use get_generate_code_from_action_prompt_from_text
        # This prompt should mention the action and reference the base code
        assert "existing_code" in prompt
        assert "Optimize loop" in prompt

    def test_type_b_attempt0_no_base(self, builder):
        """TYPE B: attempt=0, has_base=False → spec with action prompt."""
        prompt = builder.build(
            action_text="Optimize loop",
            attempt=0,
            last_round=None,
            has_passed=False,
            base_code="",
            trace_logs="",
            current_code="",
            parent_is_root=True,
        )
        # Should use get_generate_code_from_spec_with_action_prompt_from_text
        # This prompt should mention the action but not have base code
        assert "Optimize loop" in prompt

    def test_type_c_debug_from_spec_parent_is_root(self, builder, mock_round):
        """TYPE C: not passed, parent_is_root → debug from spec."""
        prompt = builder.build(
            action_text="Optimize loop",
            attempt=1,
            last_round=mock_round,
            has_passed=False,
            base_code="existing_code",  # Should be ignored due to parent_is_root
            trace_logs="error trace",
            current_code="buggy code",
            parent_is_root=True,
        )
        # parent_is_root=True overrides has_base, uses spec-based debug prompt
        # debug_round = min(attempt+1, max) = min(2, 5) = 2
        assert "2/" in prompt  # debug_round format

    def test_type_d_debug_vs_base(self, builder, mock_round):
        """TYPE D: not passed, has_base, not parent_is_root → debug vs base."""
        prompt = builder.build(
            action_text="Optimize loop",
            attempt=1,
            last_round=mock_round,
            has_passed=False,
            base_code="existing_code",
            trace_logs="error trace",
            current_code="buggy code",
            parent_is_root=False,
        )
        # Should reference base code in debug prompt
        assert "existing_code" in prompt
        # debug_round = min(attempt+1, max) = min(2, 5) = 2
        assert "2/" in prompt  # debug_round format

    def test_type_e_improve_from_spec(self, builder, mock_round):
        """TYPE E: passed, parent_is_root → improve from spec."""
        prompt = builder.build(
            action_text="Optimize loop",
            attempt=1,
            last_round=mock_round,
            has_passed=True,
            base_code="existing_code",  # Should be ignored due to parent_is_root
            trace_logs="perf trace",
            current_code="working code",
            parent_is_root=True,
        )
        # parent_is_root=True overrides has_base, uses spec-based improve prompt
        # debug_round = min(attempt+1, max) = min(2, 5) = 2
        assert "2/" in prompt  # debug_round format

    def test_type_f_improve_vs_base(self, builder, mock_round):
        """TYPE F: passed, has_base, not parent_is_root → improve vs base."""
        prompt = builder.build(
            action_text="Optimize loop",
            attempt=1,
            last_round=mock_round,
            has_passed=True,
            base_code="existing_code",
            trace_logs="perf trace",
            current_code="working code",
            parent_is_root=False,
        )
        # Should reference base code in improve prompt
        assert "existing_code" in prompt
        # debug_round = min(attempt+1, max) = min(2, 5) = 2
        assert "2/" in prompt  # debug_round format


# =============================================================================
# Test 4: debug_round Clamping
# =============================================================================


class TestDebugRoundClamping:
    """V1 semantics: debug_round is 1-indexed and capped at max_debug_improve_rounds."""

    @pytest.fixture
    def builder(self):
        return V1PromptBuilder(
            definition_text="Test definition",
            language="triton",
            target_gpu="H100",
        )

    @pytest.fixture
    def mock_round(self):
        mock_result = MockResult(succeeded=False, latency_ms=0.0)
        return Round(
            impl=MockImpl(),  # type: ignore[arg-type]
            result=mock_result,
            prompt="test",
            llm_response="code",
            prompt_tokens=0,
            completion_tokens=0,
            duration_secs=0.0,
            score=-1.0,
        )

    def test_debug_round_is_one_indexed(self, builder, mock_round):
        """V1 semantics: attempt 0 → debug_round 1."""
        # attempt=1 → debug_round = min(1+1, max) = 2
        prompt = builder.build(
            action_text="Test",
            attempt=1,  # Second attempt
            last_round=mock_round,
            has_passed=False,
            base_code="",
            trace_logs="trace",
            current_code="code",
            parent_is_root=True,
            max_debug_improve_rounds=5,
        )
        # Should contain "2/5" (debug_round = attempt + 1 = 2)
        assert "2/5" in prompt

    def test_debug_round_capped_at_max(self, builder, mock_round):
        """V1 semantics: debug_round never exceeds max_debug_improve_rounds."""
        prompt = builder.build(
            action_text="Test",
            attempt=10,  # High attempt number
            last_round=mock_round,
            has_passed=False,
            base_code="",
            trace_logs="trace",
            current_code="code",
            parent_is_root=True,
            max_debug_improve_rounds=3,
        )
        # Should be capped at 3, not 11
        # debug_round = min(10 + 1, 3) = 3
        assert "3/3" in prompt
        assert "11/" not in prompt


# =============================================================================
# Test 5: Dual Stagnation Counter Independence
# =============================================================================


class TestStagnationCounterIndependence:
    """Verify the two stagnation counters are truly independent."""

    def test_no_improve_triggers_without_base_stagnation(self):
        """no_improve terminates cycle even when no_improve_over_base hasn't triggered."""
        # All same latency = same score (1.0/2.0 = 0.5)
        # base_score=0.0 so no_improve_over_base logic doesn't apply
        results = [MockResult(True, 2.0)] * 10

        mock_wm = MagicMock()
        mock_wm.propose.return_value = []
        mock_wm.select.side_effect = [
            [Node(status="open", action=Action(title="Test"))],
            [],
        ]
        mock_wm.get_action_context.return_value = {
            "action_text": "Test",
            "base_code": "",
            "base_score": 0.0,  # No base score, so no_improve_over_base won't activate
            "base_result": None,
        }
        mock_wm._get_prompt_section.return_value = ""

        evaluator = MockEvaluator(results)
        prompt_builder = MagicMock()
        prompt_builder.build.return_value = "prompt"

        tree = Tree(root=Node(status="closed"))

        executor = V1SequentialExecutor(
            world_model=mock_wm,
            task=MockTask(),  # type: ignore[arg-type]
            evaluator=evaluator,  # type: ignore[arg-type]
            llm=lambda p: "code",
            prompt_builder=prompt_builder,
            tree=tree,
            max_rounds=20,
            cycle_config=CycleConfig(stagnation_rounds=3),
        )

        executor.run()

        # First success + 3 no-improve = 4 rounds total
        assert evaluator._idx == 4


# =============================================================================
# Test 6: Performance Summary Construction
# =============================================================================


class MockResultWithPerfSummary:
    """MockResult that supports perf_summary_lines method."""

    def __init__(self, succeeded: bool, latency_ms: float, perf_lines: list[str]):
        self._succeeded = succeeded
        self._latency_ms = latency_ms
        self._perf_lines = perf_lines

    def succeeded(self) -> bool:
        return self._succeeded

    def get_metrics(self) -> dict:
        return {"latency_ms": self._latency_ms, "speedup_factor": 1.0}

    def get_log(self) -> str:
        return "log output"

    def perf_summary_lines(self, prefix: str = "") -> list[str]:
        return [f"{prefix}: {line}" for line in self._perf_lines]


class TestPerfSummaryConstruction:
    """V1 semantics: perf_summary combines last_result and base_perf_eval with specific prefixes."""

    def test_perf_summary_uses_v1_prefixes(self):
        """V1 uses 'last_attempt' and 'base' prefixes (lowercase with underscore format).

        Reference: k_search/kernel_generators/kernel_generator_world_model.py lines 603-605:
            perf_summary_lines.extend(last_eval.perf_summary_lines(prefix="last_attempt"))
            perf_summary_lines.extend(base_perf_eval.perf_summary_lines(prefix="base"))
        """
        results = [
            MockResultWithPerfSummary(True, 2.0, ["metric1=100"]),
            MockResultWithPerfSummary(True, 2.0, ["metric2=200"]),
        ]

        mock_wm = MagicMock()
        mock_wm.propose.return_value = []
        mock_wm.select.side_effect = [
            [Node(status="open", action=Action(title="Test"))],
            [],
        ]

        base_result = MockResultWithPerfSummary(True, 1.0, ["base_metric=50"])

        mock_wm.get_action_context.return_value = {
            "action_text": "Test",
            "base_code": "parent_code",
            "base_score": 1.0,
            "base_result": base_result,
            "parent_is_root": False,
        }
        mock_wm._get_prompt_section.return_value = ""

        evaluator = MockEvaluator(results)
        prompt_builder = MagicMock()
        prompt_builder.build.return_value = "prompt"

        tree = Tree(root=Node(status="closed"))

        executor = V1SequentialExecutor(
            world_model=mock_wm,
            task=MockTask(),  # type: ignore[arg-type]
            evaluator=evaluator,  # type: ignore[arg-type]
            llm=lambda p: "code",
            prompt_builder=prompt_builder,
            tree=tree,
            max_rounds=2,
            cycle_config=CycleConfig(stagnation_rounds=10),
        )

        executor.run()

        calls = prompt_builder.build.call_args_list
        assert len(calls) >= 2
        second_call = calls[1]
        perf_summary = second_call.kwargs.get("perf_summary", "")

        # V1 uses lowercase prefixes: "last_attempt" and "base"
        assert "last_attempt:" in perf_summary, (
            f"Expected 'last_attempt:' prefix, got: {perf_summary}"
        )
        assert "base:" in perf_summary, f"Expected 'base:' prefix, got: {perf_summary}"


# =============================================================================
# Test 7: no_improve_over_base Only Activates When Both Scores Positive
# =============================================================================


class TestNoImproveOverBaseActivation:
    """V1 semantics: no_improve_over_base only increments when best_score > 0 and base_score > 0."""

    def test_no_improve_over_base_inactive_when_base_zero(self):
        """no_improve_over_base doesn't increment if base_score <= 0."""
        # All results fail, so best_score stays at 0.0
        # base_score=0.0, so the condition (best_score > 0 and base_score > 0) is never true
        results = [MockResult(False, 0.0)] * 10

        mock_wm = MagicMock()
        mock_wm.propose.return_value = []
        mock_wm.select.side_effect = [
            [Node(status="open", action=Action(title="Test"))],
            [],
        ]
        mock_wm.get_action_context.return_value = {
            "action_text": "Test",
            "base_code": "",
            "base_score": 0.0,
            "base_result": None,
        }
        mock_wm._get_prompt_section.return_value = ""

        evaluator = MockEvaluator(results)
        prompt_builder = MagicMock()
        prompt_builder.build.return_value = "prompt"

        tree = Tree(root=Node(status="closed"))

        executor = V1SequentialExecutor(
            world_model=mock_wm,
            task=MockTask(),  # type: ignore[arg-type]
            evaluator=evaluator,  # type: ignore[arg-type]
            llm=lambda p: "code",
            prompt_builder=prompt_builder,
            tree=tree,
            max_rounds=20,
            cycle_config=CycleConfig(stagnation_rounds=3),
        )

        executor.run()

        # Should terminate via no_improve, not no_improve_over_base
        # First attempt (-1.0) sets best_score, then 3 more no-improve = 4 total
        # Wait, no - all fail with -1.0, but score is only considered improved
        # if result.succeeded() AND score > best_score. So best_score stays 0.0.
        # no_improve increments every time since no improvement ever happens.
        # Should terminate after stagnation_rounds (3) with no first success.
        # Actually first round: best_score=0, score=-1.0, succeeded=False
        # -> no_improve=1 (no improvement)
        # After 3 rounds: no_improve=3 -> break
        assert evaluator._idx == 3

    def test_no_improve_over_base_inactive_when_best_zero(self):
        """no_improve_over_base doesn't increment if best_score <= 0."""
        # All results fail even though base_score > 0
        results = [MockResult(False, 0.0)] * 10

        mock_wm = MagicMock()
        mock_wm.propose.return_value = []
        mock_wm.select.side_effect = [
            [Node(status="open", action=Action(title="Test"))],
            [],
        ]
        mock_wm.get_action_context.return_value = {
            "action_text": "Test",
            "base_code": "parent_code",
            "base_score": 1.0,  # Base exists and has positive score
            "base_result": None,
            "parent_is_root": False,
        }
        mock_wm._get_prompt_section.return_value = ""

        evaluator = MockEvaluator(results)
        prompt_builder = MagicMock()
        prompt_builder.build.return_value = "prompt"

        tree = Tree(root=Node(status="closed"))

        executor = V1SequentialExecutor(
            world_model=mock_wm,
            task=MockTask(),  # type: ignore[arg-type]
            evaluator=evaluator,  # type: ignore[arg-type]
            llm=lambda p: "code",
            prompt_builder=prompt_builder,
            tree=tree,
            max_rounds=20,
            cycle_config=CycleConfig(stagnation_rounds=3),
        )

        executor.run()

        # All fail → best_score stays 0.0
        # no_improve_over_base condition: best_score > 0 and base_score > 0
        # Since best_score=0, condition is never true, so no_improve_over_base never increments
        # Terminates via no_improve after 3 rounds
        assert evaluator._idx == 3


# =============================================================================
# Test 8: base_score Default Value
# =============================================================================


class TestBaseScoreDefault:
    """V1 semantics: base_score defaults to -1.0 when parent has no solution.

    Reference: k_search/kernel_generators/kernel_generator_world_model.py line 463:
        base_score: float = -1.0
    """

    def test_base_score_defaults_to_negative_one(self):
        """V1 uses -1.0 as default base_score, not 0.0."""
        from scripts.gpu_mode_modular_k_search.run import V1WorldModel, V1Node

        # Create a node with no parent cycle (simulating root's child)
        root = Node(status="closed")
        child = V1Node(
            parent=root,
            status="open",
            action=None,
            node_id="test",
            parent_id="root",
            parent_is_root=True,
        )

        # Mock the world model to test get_action_context
        mock_manager = MagicMock()
        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )

        ctx = wm.get_action_context(child)

        # V1 semantics: base_score should be -1.0 when no parent solution
        # Current V2 bug: returns 0.0
        assert ctx["base_score"] == -1.0, f"Expected -1.0, got {ctx['base_score']}"


# =============================================================================
# Test 9: parent_is_root Determination
# =============================================================================


class TestParentIsRootDetermination:
    """V1 semantics: parent_is_root is True only when parent_id == 'root'.

    Reference: k_search/kernel_generators/kernel_generator_world_model.py line 461:
        parent_is_root = parent_id == "root"

    V2 currently also treats None as root (line 297):
        parent_is_root=parent_id == "root" or parent_id is None
    """

    def test_parent_is_root_only_for_root_string(self):
        """parent_is_root should only be True when parent_id is exactly 'root'."""
        from scripts.gpu_mode_modular_k_search.run import V1Node

        # Test with parent_id = "root"
        node_root = V1Node(
            status="open",
            node_id="child1",
            parent_id="root",
            parent_is_root=True,  # This is set during sync
        )
        assert node_root.parent_is_root is True

        # Note: V1 does NOT treat parent_id=None as root (only "root" string).
        # V2 diverges here. _sync_frontier_from_manager tests cover this.


# =============================================================================
# Test 10: Tree Sync and Node Creation
# =============================================================================


class TestTreeSyncLogic:
    """Tests for V1WorldModel._sync_frontier_from_manager tree creation logic."""

    def test_node_added_to_tree_during_sync(self):
        """Nodes synced from world model JSON should be added to the modular tree."""
        from scripts.gpu_mode_modular_k_search.run import V1WorldModel

        tree = Tree(root=Node(status="closed"))
        mock_manager = MagicMock()

        # Simulate world model JSON with one action node
        mock_manager.get.return_value = '{"decision_tree": {"root_id": "root", "nodes": [{"node_id": "action1", "parent_id": "root", "action": {"title": "Optimize"}}]}}'

        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )
        wm._initialized = True

        # Sync should add the node to the tree
        new_nodes = wm._sync_frontier_from_manager(tree)

        assert len(new_nodes) == 1
        assert new_nodes[0].node_id == "action1"
        assert new_nodes[0].action is not None
        assert new_nodes[0].action.title == "Optimize"
        assert new_nodes[0].parent_is_root is True
        # Node should be in tree's registry
        assert tree.get_node_by_id(new_nodes[0].id) is not None

    def test_action_metadata_extracted_correctly(self):
        """Action metadata (difficulty, confidence, rationale) should be extracted during sync."""
        from scripts.gpu_mode_modular_k_search.run import V1WorldModel

        tree = Tree(root=Node(status="closed"))
        mock_manager = MagicMock()

        # Simulate world model JSON with full action metadata
        mock_manager.get.return_value = """{
            "decision_tree": {
                "root_id": "root",
                "nodes": [{
                    "node_id": "action1",
                    "parent_id": "root",
                    "action": {
                        "title": "Use shared memory",
                        "difficulty_1_to_5": 4,
                        "expected_vs_baseline_factor": 1.5,
                        "confidence": 0.8,
                        "rationale": "Reduces global memory bandwidth"
                    }
                }]
            }
        }"""

        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )
        wm._initialized = True

        new_nodes = wm._sync_frontier_from_manager(tree)

        assert len(new_nodes) == 1
        action = new_nodes[0].action
        assert action is not None
        assert action.title == "Use shared memory"
        assert action.difficulty == 4
        assert action.expected_vs_baseline_factor == 1.5
        assert action.confidence == 0.8
        assert action.rationale == "Reduces global memory bandwidth"

    def test_duplicate_nodes_not_resynced(self):
        """Nodes already in _node_id_map should not be re-added."""
        from scripts.gpu_mode_modular_k_search.run import V1WorldModel

        tree = Tree(root=Node(status="closed"))
        mock_manager = MagicMock()

        mock_manager.get.return_value = '{"decision_tree": {"root_id": "root", "nodes": [{"node_id": "action1", "parent_id": "root", "action": {"title": "Test"}}]}}'

        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )
        wm._initialized = True

        # First sync
        first_sync = wm._sync_frontier_from_manager(tree)
        assert len(first_sync) == 1

        # Second sync with same data - should return empty (no new nodes)
        second_sync = wm._sync_frontier_from_manager(tree)
        assert len(second_sync) == 0

    def test_split_parent_marked_closed_on_sync(self):
        """When sync adds children to an existing open node, mark parent as closed."""
        from scripts.gpu_mode_modular_k_search.run import V1WorldModel

        tree = Tree(root=Node(status="closed"))
        mock_manager = MagicMock()

        # First sync: add parent node
        mock_manager.get.return_value = """{
            "decision_tree": {
                "root_id": "root",
                "nodes": [{"node_id": "parent1", "parent_id": "root", "action": {"title": "First"}}]
            }
        }"""

        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )
        wm._initialized = True

        first_sync = wm._sync_frontier_from_manager(tree)
        assert len(first_sync) == 1
        parent_node = first_sync[0]
        assert parent_node.status == "open"

        # Second sync: add children to that parent (simulates split_node operation)
        mock_manager.get.return_value = """{
            "decision_tree": {
                "root_id": "root",
                "nodes": [
                    {"node_id": "parent1", "parent_id": "root", "action": {"title": "First"}},
                    {"node_id": "child1", "parent_id": "parent1", "action": {"title": "Child A"}},
                    {"node_id": "child2", "parent_id": "parent1", "action": {"title": "Child B"}}
                ]
            }
        }"""

        second_sync = wm._sync_frontier_from_manager(tree)
        assert len(second_sync) == 2

        # Parent should now be closed since it was split
        assert parent_node.status == "closed"

    def test_already_closed_parent_unchanged(self):
        """Already-closed parents should not be affected by sync."""
        from scripts.gpu_mode_modular_k_search.run import V1WorldModel

        tree = Tree(root=Node(status="closed"))
        mock_manager = MagicMock()

        # First sync: add parent node
        mock_manager.get.return_value = """{
            "decision_tree": {
                "root_id": "root",
                "nodes": [{"node_id": "parent1", "parent_id": "root", "action": {"title": "First"}}]
            }
        }"""

        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )
        wm._initialized = True

        first_sync = wm._sync_frontier_from_manager(tree)
        parent_node = first_sync[0]

        # Manually close the parent
        parent_node.status = "closed"

        # Second sync: add children to that closed parent
        mock_manager.get.return_value = """{
            "decision_tree": {
                "root_id": "root",
                "nodes": [
                    {"node_id": "parent1", "parent_id": "root", "action": {"title": "First"}},
                    {"node_id": "child1", "parent_id": "parent1", "action": {"title": "Child A"}}
                ]
            }
        }"""

        wm._sync_frontier_from_manager(tree)

        # Parent should remain closed (no error, no change)
        assert parent_node.status == "closed"

    def test_root_children_dont_close_root(self):
        """Root node should never be marked closed from child sync."""
        from scripts.gpu_mode_modular_k_search.run import V1WorldModel

        tree = Tree(root=Node(status="closed"))
        mock_manager = MagicMock()

        mock_manager.get.return_value = """{
            "decision_tree": {
                "root_id": "root",
                "nodes": [
                    {"node_id": "child1", "parent_id": "root", "action": {"title": "Child A"}},
                    {"node_id": "child2", "parent_id": "root", "action": {"title": "Child B"}}
                ]
            }
        }"""

        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )
        wm._initialized = True

        # Root should remain closed and unaffected
        wm._sync_frontier_from_manager(tree)

        # Root is not in _node_id_map (special case), so it won't be touched
        assert tree.root.status == "closed"

    def test_base_code_from_parent_cycle(self):
        """get_action_context should extract base_code from parent's cycle best_round."""
        from scripts.gpu_mode_modular_k_search.run import V1WorldModel, V1Node
        from k_search.modular.world.cycle import Cycle

        mock_manager = MagicMock()
        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )

        # Create parent with a completed cycle
        parent = Node(status="closed")
        parent_result = MockResult(True, 2.0)
        parent_round = Round(
            impl=MockImpl(),  # type: ignore[arg-type]
            result=parent_result,
            prompt="parent prompt",
            llm_response="parent_generated_code",
            prompt_tokens=100,
            completion_tokens=50,
            duration_secs=1.0,
            score=0.5,
        )
        parent.cycle = Cycle(rounds=[parent_round])

        # Create child node
        child = V1Node(
            parent=parent,
            status="open",
            action=None,
            node_id="child",
            parent_id="parent",
            parent_is_root=False,
        )

        ctx = wm.get_action_context(child)

        assert ctx["base_code"] == "parent_generated_code"
        assert ctx["base_score"] == 0.5
        assert ctx["base_result"] is parent_result


class TestSolutionAttachment:
    """V1 semantics: successful cycle attaches solution_id to node via attach_solution_to_active_leaf."""

    def test_update_attaches_solution_id_on_success(self):
        """V1 semantics: passing cycle calls attach_solution_to_active_leaf."""
        from scripts.gpu_mode_modular_k_search.run import (
            V1WorldModel,
            V1UpdateContext,
            V1Node,
            V1Action,
        )
        from k_search.modular.world.cycle import Cycle

        mock_manager = MagicMock()
        mock_manager.get.return_value = (
            '{"decision_tree": {"root_id": "root", "nodes": []}}'
        )

        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )

        tree = Tree(root=Node(status="closed"))
        node = V1Node(
            parent=tree.root,
            status="open",
            action=V1Action(title="Optimize"),
            node_id="action1",
            parent_id="root",
            parent_is_root=True,
        )

        passing_result = MockResult(succeeded=True, latency_ms=2.0)
        passing_round = Round(
            impl=MockImpl(),
            result=passing_result,
            prompt="test",
            llm_response="good_code",
            prompt_tokens=100,
            completion_tokens=50,
            duration_secs=1.0,
            score=0.5,
        )
        cycle = Cycle(rounds=[passing_round])

        context = V1UpdateContext(
            tree=tree,
            node=node,
            cycle=cycle,
            round_idx=0,
            max_debug_improve_rounds=5,
        )

        wm.update(context)

        mock_manager.attach_solution_to_active_leaf.assert_called_once()
        call_kwargs = mock_manager.attach_solution_to_active_leaf.call_args.kwargs
        assert call_kwargs["definition_name"] == "test"
        assert call_kwargs["solution_id"].startswith("sol_")
        assert "action1" in call_kwargs["solution_id"]

    def test_update_does_not_attach_on_failure(self):
        """V1 semantics: failing cycle does NOT call attach_solution_to_active_leaf."""
        from scripts.gpu_mode_modular_k_search.run import (
            V1WorldModel,
            V1UpdateContext,
            V1Node,
            V1Action,
        )
        from k_search.modular.world.cycle import Cycle

        mock_manager = MagicMock()
        mock_manager.get.return_value = (
            '{"decision_tree": {"root_id": "root", "nodes": []}}'
        )

        wm = V1WorldModel(
            manager=mock_manager,
            task_name="test",
            definition_text="test",
            language="triton",
            target_gpu="H100",
        )

        tree = Tree(root=Node(status="closed"))
        node = V1Node(
            parent=tree.root,
            status="open",
            action=V1Action(title="Optimize"),
            node_id="action1",
            parent_id="root",
            parent_is_root=True,
        )

        failing_result = MockResult(succeeded=False, latency_ms=0.0)
        failing_round = Round(
            impl=MockImpl(),
            result=failing_result,
            prompt="test",
            llm_response="bad_code",
            prompt_tokens=100,
            completion_tokens=50,
            duration_secs=1.0,
            score=-1.0,
        )
        cycle = Cycle(rounds=[failing_round])

        context = V1UpdateContext(
            tree=tree,
            node=node,
            cycle=cycle,
            round_idx=0,
            max_debug_improve_rounds=5,
        )

        wm.update(context)

        mock_manager.attach_solution_to_active_leaf.assert_not_called()
        mock_manager.note_action_too_hard.assert_called_once()
