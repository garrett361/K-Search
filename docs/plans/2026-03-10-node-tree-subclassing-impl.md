# Node/Tree/Action Subclassing Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `annotations` dict from Node, Action, Tree, and Span in favor of plain dataclass subclassing.

**Architecture:** Simplify core data model by removing flexible `annotations: dict[str, Any]` fields. Users extend via subclassing with typed fields. Tree gains O(1) node lookup via `_nodes_by_id` dict.

**Tech Stack:** Python dataclasses, pytest

**Spec:** `docs/plans/2026-03-10-node-tree-subclassing-refactor.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `k_search/modular/world/node.py` | Modify | Remove `annotations`, rename `_id` to `id`, default `status=""` |
| `k_search/modular/world/action.py` | Modify | Remove `annotations` field |
| `k_search/modular/world/tree.py` | Modify | Remove `annotations`, add `_nodes_by_id`, simplify `split_node`, remove `update_node`/`_all_nodes`/`_assign_id` |
| `k_search/modular/span.py` | Modify | Remove `annotations` field |
| `k_search/modular/formatters/simple.py` | Modify | Remove annotation display logic |
| `tests/modular/world/test_tree.py` | Modify | Update tests for new API |
| `tests/modular/test_formatters.py` | Modify | Remove annotation test |
| `scripts/gpu_mode_modular_k_search/run.py` | Modify | Define V1Node/V1Action, migrate annotation usage |
| `docs/plans/2026-03-05-typed-annotations-design.md` | Delete | Superseded |

---

## Chunk 1: Core Data Model

### Task 1: Update Node

**Files:**
- Modify: `k_search/modular/world/node.py`

- [ ] **Step 1: Update Node dataclass**

Replace entire file content:

```python
"""Node dataclass for search tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.action import Action
    from k_search.modular.world.cycle import Cycle


@dataclass
class Node:
    """Search tree node."""

    id: str = ""
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    status: str = ""
    action: Action | None = None
    cycle: Cycle | None = None
```

- [ ] **Step 2: Verify no syntax errors**

Run: `python -c "from k_search.modular.world.node import Node; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add k_search/modular/world/node.py
git commit -m "refactor(world): remove annotations from Node, rename _id to id"
```

---

### Task 2: Update Action

**Files:**
- Modify: `k_search/modular/world/action.py`

- [ ] **Step 1: Update Action dataclass**

Replace entire file content:

```python
"""Action dataclass for search proposals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Action:
    """Proposal for what to try next."""

    title: str
```

- [ ] **Step 2: Verify no syntax errors**

Run: `python -c "from k_search.modular.world.action import Action; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add k_search/modular/world/action.py
git commit -m "refactor(world): remove annotations from Action"
```

---

### Task 3: Update Span

**Files:**
- Modify: `k_search/modular/span.py`

- [ ] **Step 1: Update Span dataclass**

Replace entire file content:

```python
"""Execution context for Node lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field

from k_search.modular.timer import Timer
from k_search.modular.world.node import Node


@dataclass
class Span:
    """Execution context for a Node's lifecycle.

    Wraps a Node, owns timing via Timer, extensible for future attributes.
    Passive container — executor manages lifecycle.
    """

    node: Node
    timer: Timer = field(default_factory=Timer)

    def get_metrics(self) -> dict[str, float]:
        """Return metrics dict suitable for MetricsTracker.log()."""
        return self.timer.get_timing_secs()
```

- [ ] **Step 2: Verify no syntax errors**

Run: `python -c "from k_search.modular.span import Span; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add k_search/modular/span.py
git commit -m "refactor(modular): remove annotations from Span"
```

---

## Chunk 2: Tree Refactor

### Task 4: Update Tree

**Files:**
- Modify: `k_search/modular/world/tree.py`

- [ ] **Step 1: Update Tree dataclass**

Replace entire file content:

```python
"""Tree dataclass for search state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.node import Node


@dataclass
class Tree:
    """Search tree container."""

    root: Node
    _next_id: int = field(default=0, init=False)
    _nodes_by_id: dict[str, Node] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root.id = str(self._next_id)
        self._next_id += 1
        self._nodes_by_id[self.root.id] = self.root

    def add_node(self, node: Node) -> None:
        """Add node to tree, attaching to its parent's children list."""
        if node.parent is None:
            raise ValueError("Cannot add node without parent (use root for root node)")
        node.id = str(self._next_id)
        self._next_id += 1
        self._nodes_by_id[node.id] = node
        node.parent.children.append(node)

    def get_node_by_id(self, id: str) -> Node | None:
        """Look up node by ID. O(1) via dict."""
        return self._nodes_by_id.get(id)

    def get_frontier(self) -> list[Node]:
        """Return all nodes with status 'open'."""
        return [n for n in self._nodes_by_id.values() if n.status == "open"]

    def get_best_node(self) -> Node | None:
        """Return best completed node by score, or None."""
        completed = [
            n
            for n in self._nodes_by_id.values()
            if n.status == "closed" and n.cycle and n.cycle.succeeded
        ]
        if not completed:
            return None
        return max(completed, key=lambda n: n.cycle.best_round.score)

    def get_path_to_root(self, node: Node) -> list[Node]:
        """Return path from node to root (inclusive)."""
        path = []
        current: Node | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def split_node(self, node: Node, children: list[Node]) -> None:
        """Mark node closed and add pre-constructed children to tree."""
        node.status = "closed"
        for child in children:
            child.parent = node
            self.add_node(child)

    def delete_node(self, node: Node) -> None:
        """Soft delete - mark node deleted, preserves tree structure."""
        node.status = "deleted"
```

- [ ] **Step 2: Verify no syntax errors**

Run: `python -c "from k_search.modular.world.tree import Tree; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add k_search/modular/world/tree.py
git commit -m "refactor(world): simplify Tree, add O(1) lookup, remove annotations"
```

---

## Chunk 3: Tests

### Task 5: Update Tree Tests

**Files:**
- Modify: `tests/modular/world/test_tree.py`

- [ ] **Step 1: Update test file**

Replace entire file content:

```python
"""Tests for Tree methods."""

from unittest.mock import MagicMock

import pytest

from k_search.modular.world.tree import Tree
from k_search.modular.world.node import Node
from k_search.modular.world.action import Action


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
    failed = _node_with_cycle(1.0, False)

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

    assert root.id == "0"
    assert child1.id == "1"
    assert child2.id == "2"


def test_get_node_by_id():
    root = Node(status="closed")
    tree = Tree(root=root)
    child = Node(parent=root)
    tree.add_node(child)

    assert tree.get_node_by_id("0") is root
    assert tree.get_node_by_id("1") is child
    assert tree.get_node_by_id("999") is None


def test_split_node_adds_children():
    root = Node(status="open")
    tree = Tree(root=root)

    child1 = Node(action=Action(title="Option A"), status="open")
    child2 = Node(action=Action(title="Option B"), status="open")

    tree.split_node(root, [child1, child2])

    assert root.status == "closed"
    assert child1.parent is root
    assert child2.parent is root
    assert child1 in root.children
    assert child2 in root.children
    assert child1.id == "1"
    assert child2.id == "2"


def test_delete_node_soft_deletes():
    root = Node(status="closed")
    tree = Tree(root=root)
    child = Node(parent=root, status="open")
    tree.add_node(child)

    tree.delete_node(child)

    assert child.status == "deleted"
    assert child in root.children
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/modular/world/test_tree.py -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/modular/world/test_tree.py
git commit -m "test(world): update Tree tests for new API"
```

---

### Task 6: Update Formatter Tests

**Files:**
- Modify: `tests/modular/test_formatters.py`

- [ ] **Step 1: Update test file**

Replace entire file content:

```python
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
    node.id = "5"

    result = formatter.format_node(node)

    assert result.startswith("(")
    assert result.endswith(")")
    assert "id=5" in result
    assert "status=in_progress" in result
    assert 'title="Test Action"' in result


def test_format_node_root():
    formatter = DefaultFormatter()
    node = Node(status="closed")
    node.id = "0"

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
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/modular/test_formatters.py -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/modular/test_formatters.py
git commit -m "test(formatters): remove annotation tests"
```

---

## Chunk 4: Formatter and Cleanup

### Task 7: Update Formatter

**Files:**
- Modify: `k_search/modular/formatters/simple.py`

- [ ] **Step 1: Update formatter**

Replace entire file content:

```python
"""Default tree formatter for LLM prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class DefaultFormatter:
    """File-tree-like formatting with full node info for LLM prompts."""

    def format_tree(self, tree: Tree) -> str:
        lines: list[str] = []
        self._format_subtree(tree.root, "", True, lines)
        return "\n".join(lines)

    def _format_subtree(
        self, node: Node, prefix: str, is_last: bool, lines: list[str]
    ) -> None:
        lines.append(self._format_node_line(node, prefix, is_last))
        children = node.children
        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            if prefix == "":
                child_prefix = ""
            else:
                child_prefix = prefix[:-4] + ("    " if is_last else "│   ")
            self._format_subtree(
                child,
                child_prefix + ("└── " if child_is_last else "├── "),
                child_is_last,
                lines,
            )

    def _format_node_line(self, node: Node, prefix: str, is_last: bool) -> str:
        node_str = self.format_node(node)
        if prefix:
            return prefix[:-4] + ("└── " if is_last else "├── ") + node_str
        return node_str

    def format_node(self, node: Node) -> str:
        parts: list[str] = [f"id={node.id}"]
        parts.append(f"status={node.status}")

        title = node.action.title if node.action else "root"
        parts.append(f'title="{title}"')

        return "(" + ", ".join(parts) + ")"
```

- [ ] **Step 2: Run all tests so far**

Run: `pytest tests/modular/test_formatters.py tests/modular/world/test_tree.py -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add k_search/modular/formatters/simple.py
git commit -m "refactor(formatters): remove annotation display"
```

---

### Task 8: Delete Superseded Design Doc

**Files:**
- Delete: `docs/plans/2026-03-05-typed-annotations-design.md`

- [ ] **Step 1: Delete file**

```bash
git rm docs/plans/2026-03-05-typed-annotations-design.md
```

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: remove superseded typed-annotations design"
```

---

## Chunk 5: V1 Migration

### Task 9: Define V1 Subclasses and Migrate run.py

**Files:**
- Modify: `scripts/gpu_mode_modular_k_search/run.py`

- [ ] **Step 1: Add V1Action and V1Node dataclasses**

After the imports section (around line 45), add:

```python
@dataclass
class V1Action(Action):
    """V1-specific action with difficulty/confidence metadata."""

    difficulty: int = 3
    expected_vs_baseline_factor: float | None = None
    confidence: float = 0.5
    rationale: str = ""
    v1_action_data: dict[str, Any] | None = None


@dataclass
class V1Node(Node):
    """V1-specific node with v1 ID mapping."""

    v1_node_id: str = ""
    v1_parent_id: str = ""
    parent_is_root: bool = False
```

- [ ] **Step 2: Update V1ActionRouter.add_actions_to_tree method**

Replace the node creation code (around lines 247-266) with:

```python
            new_node = V1Node(
                parent=parent_node,
                status="open",
                action=V1Action(
                    title=title,
                    difficulty=difficulty,
                    expected_vs_baseline_factor=expected_vs_baseline,
                    confidence=confidence,
                    rationale=rationale,
                    v1_action_data=action_data,
                ),
                v1_node_id=node_id,
                v1_parent_id=parent_id,
                parent_is_root=parent_id == "root" or parent_id is None,
            )
```

- [ ] **Step 3: Update V1ActionRouter.get_action_context method**

Replace the method (around lines 273-302) with:

```python
    def get_action_context(self, node: Node) -> dict[str, Any]:
        """Get context for the selected action node - reads from V1Node/V1Action fields."""
        if not node:
            return {}

        # Cast to V1Node for typed access (runtime type is V1Node)
        v1_node = node  # type: V1Node
        v1_action = node.action  # type: V1Action | None

        parent_is_root = getattr(v1_node, "parent_is_root", node.parent is None or node.parent.parent is None)

        # Get base code/score from parent's cycle if it exists
        base_code = ""
        base_score = 0.0
        if node.parent and node.parent.cycle and node.parent.cycle.best_round:
            best = node.parent.cycle.best_round
            base_code = best.llm_response
            base_score = best.score

        return {
            "v1_node_id": getattr(v1_node, "v1_node_id", ""),
            "action_text": node.action.title if node.action else "",
            "difficulty": getattr(v1_action, "difficulty", 3) if v1_action else 3,
            "confidence": getattr(v1_action, "confidence", 0.5) if v1_action else 0.5,
            "rationale": getattr(v1_action, "rationale", "") if v1_action else "",
            "parent_is_root": parent_is_root,
            "base_code": base_code,
            "base_score": base_score,
        }
```

- [ ] **Step 4: Run the script's tests**

Run: `pytest scripts/gpu_mode_modular_k_search/test_run.py -v --tb=short`
Expected: Tests pass (or adjust if tests need updating)

- [ ] **Step 5: Commit**

```bash
git add scripts/gpu_mode_modular_k_search/run.py
git commit -m "refactor(v1): migrate to V1Node/V1Action subclasses"
```

---

## Chunk 6: Final Verification

### Task 10: Run Full Test Suite

- [ ] **Step 1: Run all modular tests**

Run: `pytest tests/modular/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Run ruff check**

Run: `ruff check k_search/modular/ scripts/gpu_mode_modular_k_search/run.py`
Expected: No errors

- [ ] **Step 3: Run ruff format check**

Run: `ruff format --check k_search/modular/ scripts/gpu_mode_modular_k_search/run.py`
Expected: No reformatting needed (or run `ruff format` to fix)

- [ ] **Step 4: Final commit if any fixes**

```bash
git add -A
git commit -m "style: format and lint fixes"
```
