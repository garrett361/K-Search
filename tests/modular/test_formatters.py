"""Tests for StateFormatter implementations."""

import json
from unittest.mock import MagicMock

from k_search.modular.formatters.simple import SimpleStateFormatter
from k_search.modular.formatters.legacy_json import LegacyJSONFormatter
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


def test_legacy_formatter_matches_v1_schema():
    """Verify output matches v1 WorldModelManager's expected structure."""
    formatter = LegacyJSONFormatter()
    root = Node(status="closed")
    tree = Tree(root=root)
    child = Node(
        parent=root, action=Action(title="Test Action", annotations={"difficulty": 3})
    )
    tree.add_node(child)

    result = formatter.format_tree(tree)
    data = json.loads(result)

    # V1 expects decision_tree.nodes as list with these fields per node
    dt = data["decision_tree"]
    assert dt["root_id"] == "0"
    assert "active_leaf_id" in dt

    node = dt["nodes"][1]  # child node
    assert node["node_id"] == "1"
    assert node["parent_id"] == "0"
    assert node["status"] == "open"
    assert node["action"]["title"] == "Test Action"
    assert node["action"]["annotations"] == {"difficulty": 3}
