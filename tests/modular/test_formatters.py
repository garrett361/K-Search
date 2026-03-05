"""Tests for StateFormatter implementations."""

from unittest.mock import MagicMock

from k_search.modular.formatters.simple import SimpleStateFormatter
from k_search.modular.world.action import Action
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree


def test_simple_formatter_format_tree():
    formatter = SimpleStateFormatter()
    root = Node(status="closed")
    tree = Tree(root=root)
    child = Node(parent=root, action=Action(title="Child Action"))
    tree.add_node(child)

    result = formatter.format_tree(tree)

    assert "[0]" in result
    assert "[1]" in result
    assert "Child Action" in result


def test_simple_formatter_includes_score():
    formatter = SimpleStateFormatter()
    node = Node(action=Action(title="Test"), status="closed")
    node._id = "0"
    node.cycle = MagicMock()
    node.cycle.best_round.score = 0.85

    result = formatter.format_node(node)

    assert "0.85" in result
