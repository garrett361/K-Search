"""Tests for tree tools."""

from k_search.modular.world.tools import TREE_TOOLS, get_tree_tools, apply_tool_call
from k_search.modular.world.tree import Tree
from k_search.modular.world.node import Node


def test_tree_tools_openai_format():
    names = {t["function"]["name"] for t in TREE_TOOLS}
    assert names == {
        "insert_node",
        "update_node",
        "split_node",
        "delete_node",
        "select_node",
    }
    for tool in TREE_TOOLS:
        assert tool["type"] == "function"
        assert "parameters" in tool["function"]


def test_get_tree_tools_filters():
    result = get_tree_tools(enabled={"insert_node", "select_node"})
    names = {t["function"]["name"] for t in result}
    assert names == {"insert_node", "select_node"}

    assert get_tree_tools(enabled=None) == TREE_TOOLS
    assert get_tree_tools(enabled=set()) == []


def test_apply_insert_node():
    root = Node(status="closed")
    tree = Tree(root=root)

    result = apply_tool_call(
        tree, "insert_node", {"parent_id": "0", "title": "Test Action"}
    )

    assert result.success is True
    assert result.value is not None
    assert result.value.action is not None
    assert result.value.action.title == "Test Action"
    assert result.value._id == "1"


def test_apply_insert_node_invalid_parent():
    tree = Tree(root=Node())
    result = apply_tool_call(tree, "insert_node", {"parent_id": "999", "title": "Test"})
    assert result.success is False
    assert result.error is not None
    assert "not found" in result.error


def test_apply_update_node():
    root = Node()
    tree = Tree(root=root)
    result = apply_tool_call(
        tree, "update_node", {"node_id": "0", "annotations": {"key": "value"}}
    )
    assert result.success is True
    assert root.annotations == {"key": "value"}


def test_apply_split_node():
    root = Node(status="open")
    tree = Tree(root=root)
    result = apply_tool_call(
        tree,
        "split_node",
        {"node_id": "0", "children": [{"title": "A"}, {"title": "B"}]},
    )
    assert result.success is True
    assert root.status == "closed"
    assert len(root.children) == 2


def test_apply_delete_node():
    root = Node(status="closed")
    tree = Tree(root=root)
    child = Node(parent=root, status="open")
    tree.add_node(child)
    result = apply_tool_call(tree, "delete_node", {"node_id": "1"})
    assert result.success is True
    assert child.status == "deleted"


def test_apply_select_node():
    root = Node()
    tree = Tree(root=root)
    result = apply_tool_call(tree, "select_node", {"node_id": "0"})
    assert result.success is True
    assert result.value is root


def test_apply_unknown_tool():
    tree = Tree(root=Node())
    result = apply_tool_call(tree, "unknown_tool", {})
    assert result.success is False
    assert result.error is not None
    assert "unknown tool" in result.error
