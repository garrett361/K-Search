"""Tests for SequentialExecutor."""

from unittest.mock import MagicMock

from k_search.modular.executors.sequential import SequentialExecutor
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree
from k_search.modular.world_models.simple import SimpleWorldModel


def _mock_task():
    """Create mock task."""
    task = MagicMock()
    task.get_prompt_text.return_value = "optimize kernel"
    task.create_impl.return_value = MagicMock(name="impl")
    task.scorer.score.return_value = 0.5
    return task


def _mock_evaluator(success=True):
    """Create mock evaluator."""
    evaluator = MagicMock()
    result = MagicMock()
    result.succeeded.return_value = success
    result.get_metrics.return_value = {}
    result.get_log.return_value = ""
    evaluator.evaluate.return_value = result
    return evaluator


def _simple_action_prompt_fn(tree, context):
    return "What to try next?"


def _simple_code_prompt_fn(node, task):
    action_title = node.action.title if node.action else "implement"
    return f"{task.get_prompt_text()}\n\nAction: {action_title}\n\nGenerate:"


def test_run_completes_rounds():
    """Executor runs for max_rounds.

    Round 0: initial action (no LLM) + code gen (1 LLM call)
    Round 1+: propose action (1 LLM) + code gen (1 LLM) = 2 calls each
    """
    root = Node(status="closed")
    tree = Tree(root=root)

    # Round 0: code only, Round 1: action + code, Round 2: action + code
    llm_responses = iter(["code 0", "action 1", "code 1", "action 2", "code 2"])
    mock_llm = MagicMock(side_effect=lambda p: next(llm_responses))

    task = _mock_task()
    world_model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)
    evaluator = _mock_evaluator()

    executor = SequentialExecutor(
        world_model,
        task,
        evaluator,
        mock_llm,
        _simple_code_prompt_fn,
        tree,
        max_rounds=3,
    )
    executor.run()

    # Round 0: 1 call (code), Round 1: 2 calls, Round 2: 2 calls = 5 total
    assert mock_llm.call_count == 5
    assert evaluator.evaluate.call_count == 3


def test_run_adds_nodes_to_tree():
    """Executor adds nodes to tree."""
    root = Node(status="closed")
    tree = Tree(root=root)

    mock_llm = MagicMock(return_value="response")
    task = _mock_task()
    world_model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)
    evaluator = _mock_evaluator()

    executor = SequentialExecutor(
        world_model,
        task,
        evaluator,
        mock_llm,
        _simple_code_prompt_fn,
        tree,
        max_rounds=1,
    )
    executor.run()

    assert len(root.children) == 1
    node = root.children[0]
    assert node.status == "closed"
    assert node.action is not None  # Round 0 has initial action
    assert "optimized" in node.action.title.lower()
    assert node.cycle is not None
    assert len(node.cycle.rounds) == 1


def test_run_stops_on_empty_select():
    """Executor stops when select returns empty."""
    root = Node(status="closed")
    tree = Tree(root=root)

    world_model = MagicMock()
    world_model.propose.return_value = []
    world_model.select.return_value = []

    mock_llm = MagicMock(return_value="code")
    task = _mock_task()
    evaluator = _mock_evaluator()

    executor = SequentialExecutor(
        world_model,
        task,
        evaluator,
        mock_llm,
        _simple_code_prompt_fn,
        tree,
        max_rounds=10,
    )
    executor.run()

    # No nodes proposed, select returns empty, no execution
    assert evaluator.evaluate.call_count == 0
