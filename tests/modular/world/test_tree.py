"""Tests for Tree methods (add_node, get_frontier, get_best_node, get_path_to_root)."""

from unittest.mock import MagicMock

import pytest

from k_search.modular.world.tree import Tree
from k_search.modular.world.node import Node


def test_add_node_attaches_to_parent_children():
    root = Node(status="closed")
    tree = Tree(root=root)

    child = Node(parent=root)
    tree.add_node(child)

    assert child in root.children


def test_add_node_errors_on_orphan():
    tree = Tree(root=Node(status="closed"))
    orphan = Node(parent=None)

    with pytest.raises(ValueError, match="parent"):
        tree.add_node(orphan)


def test_get_frontier_returns_open_nodes():
    root = Node(status="closed")
    tree = Tree(root=root)

    open_node = Node(parent=root, status="open")
    closed_node = Node(parent=root, status="closed")
    tree.add_node(open_node)
    tree.add_node(closed_node)

    frontier = tree.get_frontier()
    assert open_node in frontier
    assert closed_node not in frontier


def test_get_best_node_by_score():
    root = Node(status="closed")
    tree = Tree(root=root)

    def _node_with_cycle(score: float, succeeded: bool) -> Node:
        n = Node(parent=root, status="closed")
        n.cycle = MagicMock()
        n.cycle.succeeded = succeeded
        n.cycle.best_round.score = score
        return n

    low = _node_with_cycle(0.5, True)
    high = _node_with_cycle(0.9, True)
    failed = _node_with_cycle(1.0, False)  # ignored

    tree.add_node(low)
    tree.add_node(high)
    tree.add_node(failed)

    assert tree.get_best_node() is high


def test_get_path_to_root():
    root = Node(status="closed")
    child = Node(parent=root, status="closed")
    grandchild = Node(parent=child, status="closed")
    tree = Tree(root=root)
    tree.add_node(child)
    tree.add_node(grandchild)

    assert tree.get_path_to_root(grandchild) == [grandchild, child, root]


def test_tree_assigns_sequential_ids():
    root = Node(status="closed")
    tree = Tree(root=root)

    child1 = Node(parent=root)
    child2 = Node(parent=root)
    tree.add_node(child1)
    tree.add_node(child2)

    assert root._id == "0"
    assert child1._id == "1"
    assert child2._id == "2"


def test_get_node_by_id():
    root = Node(status="closed")
    tree = Tree(root=root)
    child = Node(parent=root)
    tree.add_node(child)

    assert tree._get_node_by_id("0") is root
    assert tree._get_node_by_id("1") is child
    assert tree._get_node_by_id("999") is None
