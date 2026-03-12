# Parsing & Formatting Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bridge Tree data model and LLM interactions with tool schemas, formatters, and result types.

**Architecture:** OpenAI-style tool calling. Tree manages Node IDs. tools.py handles ID resolution. ParseResult[T] for fallible operations.

**Tech Stack:** Python 3.12, dataclasses, typing (Generic/TypeVar), pytest

---

## Task 1: Add _id to Node and ID Management to Tree

**Files:**
- Modify: `k_search/modular/world/node.py`
- Modify: `k_search/modular/world/tree.py`
- Test: `tests/modular/world/test_tree.py`

**Step 1: Write tests for ID assignment behavior**

Add to `tests/modular/world/test_tree.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/modular/world/test_tree.py::test_tree_assigns_sequential_ids tests/modular/world/test_tree.py::test_get_node_by_id -v`

**Step 3: Implement**

In `k_search/modular/world/node.py`, add `_id` as first field:

```python
@dataclass
class Node:
    """Search tree node."""

    _id: str = ""
    parent: Node | None = None
    # ... rest unchanged
```

In `k_search/modular/world/tree.py`:

```python
@dataclass
class Tree:
    """Search tree container."""

    root: Node
    annotations: dict[str, Any] | None = None
    _next_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._assign_id(self.root)

    def _assign_id(self, node: Node) -> None:
        node._id = str(self._next_id)
        self._next_id += 1

    def add_node(self, node: Node) -> None:
        """Add node to tree, attaching to its parent's children list."""
        if node.parent is None:
            raise ValueError("Cannot add node without parent (use root for root node)")
        self._assign_id(node)
        node.parent.children.append(node)

    def _get_node_by_id(self, id: str) -> Node | None:
        """Look up node by ID. Returns None if not found."""
        for node in self._all_nodes():
            if node._id == id:
                return node
        return None

    # ... rest unchanged
```

**Step 4: Verify and commit**

```bash
pytest tests/modular/world/test_tree.py -v
git add k_search/modular/world/node.py k_search/modular/world/tree.py tests/modular/world/test_tree.py
git commit -m "feat(world): add _id to Node and ID management to Tree"
```

---

## Task 2: Add Tree Mutation Methods

**Files:**
- Modify: `k_search/modular/world/tree.py`
- Test: `tests/modular/world/test_tree.py`

**Step 1: Write tests**

Add to `tests/modular/world/test_tree.py`:

```python
def test_update_node_merges_annotations():
    root = Node(annotations={"a": 1})
    tree = Tree(root=root)
    tree.update_node(root, {"b": 2})
    assert root.annotations == {"a": 1, "b": 2}


def test_update_node_creates_annotations_if_none():
    root = Node(annotations=None)
    tree = Tree(root=root)
    tree.update_node(root, {"x": "y"})
    assert root.annotations == {"x": "y"}


def test_split_node_creates_children():
    root = Node(status="open")
    tree = Tree(root=root)

    children = tree.split_node(root, [
        {"title": "Option A"},
        {"title": "Option B", "annotations": {"priority": "high"}},
    ])

    assert len(children) == 2
    assert root.status == "closed"
    assert children[0].parent is root
    assert children[0].action.title == "Option A"
    assert children[1].action.annotations == {"priority": "high"}
    assert children[0]._id == "1"
    assert children[1]._id == "2"


def test_delete_node_soft_deletes():
    root = Node(status="closed")
    tree = Tree(root=root)
    child = Node(parent=root, status="open")
    tree.add_node(child)

    tree.delete_node(child)

    assert child.status == "deleted"
    assert child in root.children
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/modular/world/test_tree.py -k "update_node or split_node or delete_node" -v`

**Step 3: Implement**

Add import at top of `tree.py`:

```python
from k_search.modular.world.action import Action
```

Add to `Tree` class:

```python
def update_node(self, node: Node, annotations: dict[str, Any]) -> None:
    """Merge annotations into node."""
    if node.annotations is None:
        node.annotations = {}
    node.annotations.update(annotations)

def split_node(self, node: Node, children_data: list[dict[str, Any]]) -> list[Node]:
    """Split node into children. Mark parent closed, add children as open."""
    node.status = "closed"
    new_children = []
    for data in children_data:
        child = Node(
            parent=node,
            status="open",
            action=Action(title=data.get("title", ""), annotations=data.get("annotations")),
        )
        self.add_node(child)
        new_children.append(child)
    return new_children

def delete_node(self, node: Node) -> None:
    """Soft delete - mark node deleted, preserves tree structure."""
    node.status = "deleted"
```

**Step 4: Verify and commit**

```bash
pytest tests/modular/world/test_tree.py -v
git add k_search/modular/world/tree.py tests/modular/world/test_tree.py
git commit -m "feat(world): add update_node, split_node, delete_node to Tree"
```

---

## Task 3: Create ParseResult[T]

**Files:**
- Create: `k_search/modular/world/parse_result.py`
- Test: `tests/modular/world/test_parse_result.py`

**Step 1: Write tests**

Create `tests/modular/world/test_parse_result.py`:

```python
"""Tests for ParseResult[T]."""

from k_search.modular.world.parse_result import ParseResult


def test_parse_result_ok():
    result = ParseResult.ok("value")
    assert result.success is True
    assert result.value == "value"
    assert result.error is None


def test_parse_result_fail():
    result = ParseResult.fail("something went wrong")
    assert result.success is False
    assert result.value is None
    assert result.error == "something went wrong"
```

**Step 2: Implement**

Create `k_search/modular/world/parse_result.py`:

```python
"""ParseResult for fallible operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ParseResult(Generic[T]):
    """Result of parsing/applying a tool call."""

    success: bool
    value: T | None = None
    error: str | None = None

    @classmethod
    def ok(cls, value: T) -> ParseResult[T]:
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error: str) -> ParseResult[T]:
        return cls(success=False, error=error)
```

**Step 3: Verify and commit**

```bash
pytest tests/modular/world/test_parse_result.py -v
git add k_search/modular/world/parse_result.py tests/modular/world/test_parse_result.py
git commit -m "feat(world): add ParseResult[T] for fallible operations"
```

---

## Task 4: Create Tool Schemas and apply_tool_call

**Files:**
- Create: `k_search/modular/world/tools.py`
- Test: `tests/modular/world/test_tools.py`

**Step 1: Write tests**

Create `tests/modular/world/test_tools.py`:

```python
"""Tests for tree tools."""

from k_search.modular.world.tools import TREE_TOOLS, get_tree_tools, apply_tool_call
from k_search.modular.world.tree import Tree
from k_search.modular.world.node import Node


def test_tree_tools_openai_format():
    names = {t["function"]["name"] for t in TREE_TOOLS}
    assert names == {"insert_node", "update_node", "split_node", "delete_node", "select_node"}
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

    result = apply_tool_call(tree, "insert_node", {"parent_id": "0", "title": "Test Action"})

    assert result.success is True
    assert result.value.action.title == "Test Action"
    assert result.value._id == "1"


def test_apply_insert_node_invalid_parent():
    tree = Tree(root=Node())
    result = apply_tool_call(tree, "insert_node", {"parent_id": "999", "title": "Test"})
    assert result.success is False
    assert "not found" in result.error


def test_apply_update_node():
    root = Node()
    tree = Tree(root=root)
    result = apply_tool_call(tree, "update_node", {"node_id": "0", "annotations": {"key": "value"}})
    assert result.success is True
    assert root.annotations == {"key": "value"}


def test_apply_split_node():
    root = Node(status="open")
    tree = Tree(root=root)
    result = apply_tool_call(tree, "split_node", {"node_id": "0", "children": [{"title": "A"}, {"title": "B"}]})
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
    assert "unknown tool" in result.error
```

**Step 2: Implement**

Create `k_search/modular/world/tools.py`:

```python
"""Tool schemas and application for tree operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from k_search.modular.world.action import Action
from k_search.modular.world.node import Node
from k_search.modular.world.parse_result import ParseResult

if TYPE_CHECKING:
    from k_search.modular.world.tree import Tree

TREE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "insert_node",
            "description": "Add a new action node to the tree",
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_id": {"type": "string", "description": "ID of parent node"},
                    "title": {"type": "string", "description": "Action title"},
                    "annotations": {"type": "object", "description": "Optional metadata"},
                },
                "required": ["parent_id", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_node",
            "description": "Update annotations on an existing node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of node to update"},
                    "annotations": {"type": "object", "description": "Annotations to merge"},
                },
                "required": ["node_id", "annotations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "split_node",
            "description": "Split a node into multiple child actions",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of node to split"},
                    "children": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"title": {"type": "string"}, "annotations": {"type": "object"}}, "required": ["title"]},
                        "description": "Child actions to create",
                    },
                },
                "required": ["node_id", "children"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_node",
            "description": "Mark a node as deleted",
            "parameters": {"type": "object", "properties": {"node_id": {"type": "string", "description": "ID of node to delete"}}, "required": ["node_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_node",
            "description": "Select a node to pursue next",
            "parameters": {"type": "object", "properties": {"node_id": {"type": "string", "description": "ID of node to select"}}, "required": ["node_id"]},
        },
    },
]


def get_tree_tools(enabled: set[str] | None = None) -> list[dict[str, Any]]:
    """Return tool schemas, optionally filtered to enabled set."""
    if enabled is None:
        return TREE_TOOLS
    return [t for t in TREE_TOOLS if t["function"]["name"] in enabled]


def apply_tool_call(tree: Tree, tool_name: str, args: dict[str, Any]) -> ParseResult[Node]:
    """Route tool call to Tree method. Returns ParseResult for error handling."""
    if tool_name == "insert_node":
        parent = tree._get_node_by_id(args.get("parent_id", ""))
        if parent is None:
            return ParseResult.fail(f"parent_id not found: {args.get('parent_id')}")
        node = Node(parent=parent, action=Action(title=args["title"], annotations=args.get("annotations")))
        tree.add_node(node)
        return ParseResult.ok(node)

    if tool_name == "update_node":
        node = tree._get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        tree.update_node(node, args["annotations"])
        return ParseResult.ok(node)

    if tool_name == "split_node":
        node = tree._get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        children = tree.split_node(node, args.get("children", []))
        return ParseResult.ok(children[0] if children else node)

    if tool_name == "delete_node":
        node = tree._get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        tree.delete_node(node)
        return ParseResult.ok(node)

    if tool_name == "select_node":
        node = tree._get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        return ParseResult.ok(node)

    return ParseResult.fail(f"unknown tool: {tool_name}")
```

**Step 3: Verify and commit**

```bash
pytest tests/modular/world/test_tools.py -v
git add k_search/modular/world/tools.py tests/modular/world/test_tools.py
git commit -m "feat(world): add TREE_TOOLS schema and apply_tool_call"
```

---

## Task 5: Create Formatters

**Files:**
- Create: `k_search/modular/formatters/__init__.py`
- Create: `k_search/modular/formatters/simple.py`
- Create: `k_search/modular/formatters/legacy_json.py`
- Test: `tests/modular/test_formatters.py`

**Step 1: Write tests**

Create `tests/modular/test_formatters.py`:

```python
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
    child = Node(parent=root, action=Action(title="Test Action", annotations={"difficulty": 3}))
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
```

**Step 2: Implement**

Create `k_search/modular/formatters/__init__.py`:

```python
"""Formatters for tree state serialization."""

from k_search.modular.formatters.simple import SimpleStateFormatter
from k_search.modular.formatters.legacy_json import LegacyJSONFormatter

__all__ = ["SimpleStateFormatter", "LegacyJSONFormatter"]
```

Create `k_search/modular/formatters/simple.py`:

```python
"""Simple tree formatter for LLM prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class SimpleStateFormatter:
    """Minimal tree formatting for LLM prompts."""

    def format_tree(self, tree: Tree) -> str:
        return "\n".join(self.format_node(n) for n in tree._all_nodes())

    def format_node(self, node: Node) -> str:
        title = node.action.title if node.action else "(root)"
        score = f" (score: {node.cycle.best_round.score:.2f})" if node.cycle and node.cycle.best_round else ""
        return f"[{node._id}] {node.status}: {title}{score}"
```

Create `k_search/modular/formatters/legacy_json.py`:

```python
"""V1-compatible JSON formatter for parity validation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class LegacyJSONFormatter:
    """V1-compatible JSON format for parity validation."""

    def format_tree(self, tree: Tree) -> str:
        return json.dumps(self._tree_to_dict(tree), indent=2)

    def format_node(self, node: Node) -> str:
        return json.dumps(self._node_to_dict(node))

    def _tree_to_dict(self, tree: Tree) -> dict[str, Any]:
        nodes = [self._node_to_dict(n) for n in tree._all_nodes()]
        best = tree.get_best_node()
        return {"decision_tree": {"root_id": tree.root._id, "active_leaf_id": best._id if best else tree.root._id, "nodes": nodes}}

    def _node_to_dict(self, node: Node) -> dict[str, Any]:
        action_dict = {"title": node.action.title, "description": "", "annotations": node.action.annotations or {}} if node.action else None
        score = node.cycle.best_round.score if node.cycle and node.cycle.best_round else None
        return {"node_id": node._id, "parent_id": node.parent._id if node.parent else None, "status": node.status, "action": action_dict, "score": score, "annotations": node.annotations or {}}
```

**Step 3: Verify and commit**

```bash
pytest tests/modular/test_formatters.py -v
git add k_search/modular/formatters/ tests/modular/test_formatters.py
git commit -m "feat(formatters): add SimpleStateFormatter and LegacyJSONFormatter"
```

---

## Task 6: Export and Verify

**Files:**
- Modify: `k_search/modular/world/__init__.py`

**Step 1: Update exports**

```python
"""World module - tree data model for search state."""

from k_search.modular.world.action import Action
from k_search.modular.world.cycle import Cycle
from k_search.modular.world.node import Node
from k_search.modular.world.parse_result import ParseResult
from k_search.modular.world.round import Round
from k_search.modular.world.tools import TREE_TOOLS, apply_tool_call, get_tree_tools
from k_search.modular.world.tree import Tree

__all__ = ["Action", "Cycle", "Node", "ParseResult", "Round", "TREE_TOOLS", "Tree", "apply_tool_call", "get_tree_tools"]
```

**Step 2: Full verification**

```bash
pytest tests/modular/ -v
ty check k_search/modular/world/ k_search/modular/formatters/
ruff check k_search/modular/world/ k_search/modular/formatters/
```

**Step 3: Commit**

```bash
git add k_search/modular/world/__init__.py
git commit -m "feat(world): export ParseResult and tool functions"
```

---

## Task 7 (Optional): API Compatibility Test

**Files:**
- Create: `tests/modular/test_tool_calling_api.py`

```python
"""API compatibility test for OpenAI tool calling."""

import os
import pytest
from k_search.modular.world.tools import TREE_TOOLS

pytestmark = pytest.mark.api


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_tool_calling():
    import json
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a tree editor. Use insert_node to add a node."},
            {"role": "user", "content": "Add a node called 'Optimize memory' under the root (id 0)."},
        ],
        tools=TREE_TOOLS,
        tool_choice="required",
    )

    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "insert_node"
    args = json.loads(tool_call.function.arguments)
    assert "parent_id" in args and "title" in args
```

Run: `pytest tests/modular/test_tool_calling_api.py -v -m api`

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add _id to Node, ID management to Tree, _get_node_by_id |
| 2 | Add update_node, split_node, delete_node to Tree |
| 3 | Create ParseResult[T] |
| 4 | Create TREE_TOOLS schema, get_tree_tools, apply_tool_call |
| 5 | Create SimpleStateFormatter and LegacyJSONFormatter |
| 6 | Export new types and run full verification |
| 7 | (Optional) API compatibility test |
