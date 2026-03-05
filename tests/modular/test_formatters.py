"""Tests for StateFormatter implementations."""

from k_search.modular.formatters.simple import DefaultFormatter
from k_search.modular.world.action import Action
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree


def test_format_tree_shows_hierarchy():
    formatter = DefaultFormatter()
    root = Node(status="closed")
    tree = Tree(root=root)
    child1 = Node(parent=root, action=Action(title="Child 1"), status="open")
    child2 = Node(parent=root, action=Action(title="Child 2"), status="open")
    tree.add_node(child1)
    tree.add_node(child2)

    result = formatter.format_tree(tree)

    assert "├──" in result or "└──" in result
    assert "id=0" in result
    assert "id=1" in result
    assert "id=2" in result
    assert "Child 1" in result
    assert "Child 2" in result


def test_format_tree_nested_children():
    formatter = DefaultFormatter()
    root = Node(status="closed")
    tree = Tree(root=root)
    child = Node(parent=root, action=Action(title="Level 1"))
    tree.add_node(child)
    grandchild = Node(parent=child, action=Action(title="Level 2"))
    tree.add_node(grandchild)

    result = formatter.format_tree(tree)
    lines = result.split("\n")

    assert len(lines) == 3
    assert "Level 1" in lines[1]
    assert "Level 2" in lines[2]


def test_format_node_structure():
    formatter = DefaultFormatter()
    node = Node(action=Action(title="Test Action"), status="in_progress")
    node._id = "5"

    result = formatter.format_node(node)

    assert result.startswith("(")
    assert result.endswith(")")
    assert "id=5" in result
    assert "status=in_progress" in result
    assert 'title="Test Action"' in result


def test_format_node_with_annotations():
    formatter = DefaultFormatter()
    node = Node(
        action=Action(title="Test"),
        status="open",
        annotations={"priority": "high", "difficulty": 3},
    )
    node._id = "0"

    result = formatter.format_node(node)

    assert "annotations={" in result
    assert "priority: high" in result
    assert "difficulty: 3" in result


def test_format_node_without_annotations():
    formatter = DefaultFormatter()
    node = Node(action=Action(title="Test"), status="open")
    node._id = "0"

    result = formatter.format_node(node)

    assert "annotations" not in result


def test_format_node_root():
    formatter = DefaultFormatter()
    node = Node(status="closed")
    node._id = "0"

    result = formatter.format_node(node)

    assert 'title="root"' in result
    assert "id=0" in result


def test_format_tree_structure():
    formatter = DefaultFormatter()
    root = Node(status="closed")
    tree = Tree(root=root)

    a = Node(parent=root, action=Action(title="A"), status="open")
    b = Node(parent=root, action=Action(title="B"), status="open")
    tree.add_node(a)
    tree.add_node(b)

    a1 = Node(parent=a, action=Action(title="A1"), status="open")
    tree.add_node(a1)

    result = formatter.format_tree(tree)
    lines = result.split("\n")

    assert lines[0].startswith("(id=0")
    assert "├──" in lines[1]
    assert "└──" in lines[2] and "A1" in lines[2]
    assert "└──" in lines[3] and "B" in lines[3]
