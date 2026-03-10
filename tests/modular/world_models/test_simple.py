"""Tests for SimpleWorldModel."""

from unittest.mock import MagicMock

import pytest

from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree
from k_search.modular.world_models.simple import (
    SimpleWorldModel,
    SimpleWorldModelContext,
)


def _simple_action_prompt_fn(context: SimpleWorldModelContext) -> str:
    return "What to try next?"


def test_propose_uses_initial_action_on_empty_tree():
    """propose() uses initial action on empty tree (no LLM call)."""
    mock_llm = MagicMock(return_value="try loop tiling")

    root = Node(status="closed")
    tree = Tree(root=root)
    model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)

    nodes = model.propose(SimpleWorldModelContext(tree=tree))

    assert len(nodes) == 1
    assert nodes[0].action is not None
    assert "optimized" in nodes[0].action.title.lower()
    assert nodes[0].status == "open"
    assert nodes[0].parent is root
    assert nodes[0] not in root.children
    mock_llm.assert_not_called()  # No LLM call for initial action


def test_propose_calls_llm_with_history():
    """propose() calls LLM when tree has history."""
    mock_llm = MagicMock(return_value="try loop tiling")

    root = Node(status="closed")
    tree = Tree(root=root)

    # Add a successful node so tree has history
    from k_search.modular.world.cycle import Cycle

    first = Node(parent=root, status="closed")
    mock_round = MagicMock()
    mock_round.score = 0.5
    mock_round.result.succeeded.return_value = True
    first.cycle = Cycle(rounds=[mock_round])
    tree.add_node(first)

    model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)
    nodes = model.propose(SimpleWorldModelContext(tree=tree))

    assert len(nodes) == 1
    assert nodes[0].action is not None
    assert nodes[0].action.title == "try loop tiling"
    mock_llm.assert_called_once()


def test_propose_sets_parent_to_last_in_chain():
    """propose() sets parent to last node in linear chain."""
    mock_llm = MagicMock(return_value="action 2")

    root = Node(status="closed")
    tree = Tree(root=root)
    first = Node(parent=root, status="closed")
    tree.add_node(first)

    model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)
    nodes = model.propose(SimpleWorldModelContext(tree=tree))

    assert nodes[0].parent is first


def test_propose_raises_on_branching_tree():
    """propose() raises if tree has branching (not linear)."""
    mock_llm = MagicMock(return_value="action")

    root = Node(status="closed")
    tree = Tree(root=root)
    child1 = Node(parent=root, status="closed")
    child2 = Node(parent=root, status="closed")
    tree.add_node(child1)
    tree.add_node(child2)

    model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)

    with pytest.raises(ValueError, match="linear tree"):
        model.propose(SimpleWorldModelContext(tree=tree))


def test_select_returns_latest():
    """select() returns the latest open node."""
    root = Node(status="closed")
    tree = Tree(root=root)

    node1 = Node(parent=root, status="closed")
    node2 = Node(parent=root, status="open")
    tree.add_node(node1)
    tree.add_node(node2)

    model = SimpleWorldModel(MagicMock(), _simple_action_prompt_fn)
    selected = model.select(SimpleWorldModelContext(tree=tree))

    assert selected == [node2]


def test_select_empty_frontier():
    """select() returns empty list when no open nodes."""
    root = Node(status="closed")
    tree = Tree(root=root)

    model = SimpleWorldModel(MagicMock(), _simple_action_prompt_fn)
    selected = model.select(SimpleWorldModelContext(tree=tree))

    assert selected == []


def test_update_is_noop():
    """update() does nothing."""
    root = Node(status="closed")
    tree = Tree(root=root)

    model = SimpleWorldModel(MagicMock(), _simple_action_prompt_fn)
    model.update(SimpleWorldModelContext(tree=tree))
