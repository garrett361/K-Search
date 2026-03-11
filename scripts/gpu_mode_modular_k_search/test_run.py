"""Minimal tests for V1 case search - core logic only."""

from unittest.mock import MagicMock

from k_search.modular.world.action import Action
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree

from scripts.gpu_mode_modular_k_search.run import CycleConfig, V1SequentialExecutor


class MockResult:
    def __init__(self, succeeded: bool, score: float):
        self._succeeded = succeeded
        self._score = score

    def succeeded(self) -> bool:
        return self._succeeded

    def get_metrics(self) -> dict:
        return {"score": self._score}

    def get_log(self) -> str:
        return ""


class MockImpl:
    pass


class MockTask:
    name = "test"

    def create_impl(self, code: str):
        return MockImpl()

    class scorer:
        @staticmethod
        def score(result) -> float:
            return result._score if result._succeeded else 0.0


class MockEvaluator:
    def __init__(self, results: list[MockResult]):
        self._results = results
        self._idx = 0

    def evaluate(self, impl, context=None) -> MockResult:
        result = self._results[self._idx % len(self._results)]
        self._idx += 1
        return result


def test_stagnation_ends_cycle_early():
    """Core test: stagnation window terminates cycle before max attempts."""
    results = [MockResult(True, 0.5)] * 10  # All same score

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
    }
    mock_wm._get_prompt_section.return_value = ""

    evaluator = MockEvaluator(results)
    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "prompt"

    tree = Tree(root=Node(status="closed"))

    executor = V1SequentialExecutor(
        world_model=mock_wm,
        task=MockTask(),
        evaluator=evaluator,
        llm=lambda p: "code",
        prompt_builder=prompt_builder,
        tree=tree,
        max_rounds=20,
        cycle_config=CycleConfig(max_rounds_per_cycle=10, stagnation_rounds=3),
    )

    executor.run()

    # Should stop after stagnation_rounds+1 (first success + 3 no-improve)
    assert evaluator._idx <= 5


def test_no_improve_over_base_ends_cycle():
    """Cycle terminates when improving locally but never beating base score.

    V1 semantics: two independent stagnation counters. This tests that
    no_improve_over_base terminates the cycle even when no_improve doesn't.
    """
    # Each result improves over previous (0.5 → 0.6 → 0.7...) so no_improve=0
    # But all are below base_score=1.0, so no_improve_over_base increments each round
    results = [
        MockResult(True, 0.5),
        MockResult(True, 0.6),
        MockResult(True, 0.7),
        MockResult(True, 0.8),
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
        "base_score": 1.0,  # Higher than all results
        "parent_is_root": False,
    }
    mock_wm._get_prompt_section.return_value = ""

    evaluator = MockEvaluator(results)
    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "prompt"

    tree = Tree(root=Node(status="closed"))

    executor = V1SequentialExecutor(
        world_model=mock_wm,
        task=MockTask(),
        evaluator=evaluator,
        llm=lambda p: "code",
        prompt_builder=prompt_builder,
        tree=tree,
        max_rounds=20,
        cycle_config=CycleConfig(max_rounds_per_cycle=10, stagnation_rounds=3),
    )

    executor.run()

    # With stagnation_rounds=3: after rounds 0,1,2 each increments no_improve_over_base
    # (since all scores < base_score=1.0). At round 2's end: no_improve_over_base=3 → break
    assert evaluator._idx == 3
