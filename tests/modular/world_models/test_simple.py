"""Tests for SimpleWorldModel."""

from unittest.mock import MagicMock

import pytest

from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree
from k_search.modular.world_models.simple import SimpleWorldModel


def _simple_action_prompt_fn(tree, context):
    return "What to try next?"


def test_propose_creates_node_with_action():
    """propose() calls LLM and creates node with action (doesn't add to tree)."""
    mock_llm = MagicMock(return_value="try loop tiling")

    root = Node(status="closed")
    tree = Tree(root=root)
    model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)

    nodes = model.propose(tree)

    assert len(nodes) == 1
    assert nodes[0].action.title == "try loop tiling"
    assert nodes[0].status == "open"
    assert nodes[0].parent is root
    assert nodes[0] not in root.children
    mock_llm.assert_called_once()


def test_propose_sets_parent_to_last_in_chain():
    """propose() sets parent to last node in linear chain."""
    mock_llm = MagicMock(return_value="action 2")

    root = Node(status="closed")
    tree = Tree(root=root)
    first = Node(parent=root, status="closed")
    tree.add_node(first)

    model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)
    nodes = model.propose(tree)

    assert nodes[0].parent is first


def test_propose_passes_context_to_prompt_fn():
    """propose() passes context to action_prompt_fn."""
    mock_llm = MagicMock(return_value="action")
    mock_prompt_fn = MagicMock(return_value="prompt")

    root = Node(status="closed")
    tree = Tree(root=root)
    model = SimpleWorldModel(mock_llm, mock_prompt_fn)

    context = {"round_idx": 5}
    model.propose(tree, context)

    mock_prompt_fn.assert_called_once_with(tree, context)


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
        model.propose(tree)


def test_select_returns_latest():
    """select() returns the latest open node."""
    root = Node(status="closed")
    tree = Tree(root=root)

    node1 = Node(parent=root, status="closed")
    node2 = Node(parent=root, status="open")
    tree.add_node(node1)
    tree.add_node(node2)

    model = SimpleWorldModel(MagicMock(), _simple_action_prompt_fn)
    selected = model.select(tree)

    assert selected == [node2]


def test_select_empty_frontier():
    """select() returns empty list when no open nodes."""
    root = Node(status="closed")
    tree = Tree(root=root)

    model = SimpleWorldModel(MagicMock(), _simple_action_prompt_fn)
    selected = model.select(tree)

    assert selected == []


def test_update_is_noop():
    """update() does nothing."""
    root = Node(status="closed")
    tree = Tree(root=root)

    model = SimpleWorldModel(MagicMock(), _simple_action_prompt_fn)
    model.update(tree)
