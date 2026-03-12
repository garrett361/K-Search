# Tree Data Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `world/` module with Tree, Node, Action, Cycle dataclasses and WorldModel/StateFormatter protocols.

**Architecture:** All search state types in `k_search/modular/world/`. Direct object references (no IDs). Protocols define world model and formatter interfaces.

**Tech Stack:** Python dataclasses, typing.Protocol

**Design doc:** `docs/plans/2026-03-05-tree-data-model-design.md`

---

## Task 1: Create world/ module and move Round

**Files:**
- Create: `k_search/modular/world/__init__.py`
- Move: `k_search/modular/round.py` → `k_search/modular/world/round.py`
- Modify: imports in `loop.py`, `prompts.py`, `artifacts/*.py`, `protocols/*.py`, `adapters/gpu_mode/*.py`

**Step 1: Create world/ directory and move round.py**

```bash
mkdir -p k_search/modular/world
touch k_search/modular/world/__init__.py
git mv k_search/modular/round.py k_search/modular/world/round.py
```

**Step 2: Update imports**

Change `from k_search.modular.round import Round` to `from k_search.modular.world.round import Round` in:
- `k_search/modular/loop.py`
- `k_search/modular/prompts.py`
- `k_search/modular/__init__.py`
- `k_search/modular/artifacts/noop.py`
- `k_search/modular/artifacts/local.py`
- `k_search/modular/artifacts/wandb.py`
- `k_search/modular/protocols/artifact_store.py`
- `k_search/modular/protocols/feedback_provider.py`
- `k_search/modular/adapters/gpu_mode/task_definition.py`

**Step 3: Verify and commit**

Run: `pytest tests/modular/ -v --tb=short`

```bash
git add -A && git commit -m "refactor(modular): create world/ and move Round"
```

---

## Task 2: Create Action and Node dataclasses

**Files:**
- Create: `k_search/modular/world/action.py`
- Create: `k_search/modular/world/node.py`

No tests - these are simple dataclasses with no behavior.

**Step 1: Create Action**

File: `k_search/modular/world/action.py`

```python
"""Action dataclass for search proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Action:
    """Proposal for what to try next."""

    title: str
    annotations: dict[str, Any] | None = None
```

**Step 2: Create Node**

File: `k_search/modular/world/node.py`

```python
"""Node dataclass for search tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from k_search.modular.world.action import Action
    from k_search.modular.world.cycle import Cycle


@dataclass
class Node:
    """Search tree node."""

    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    status: str = "open"  # "open" | "in_progress" | "closed"

    action: Action | None = None
    cycle: Cycle | None = None
    annotations: dict[str, Any] | None = None
```

**Step 3: Commit**

```bash
git add k_search/modular/world/action.py k_search/modular/world/node.py
git commit -m "feat(modular): add Action and Node dataclasses"
```

---

## Task 3: Create Cycle dataclass

**Files:**
- Create: `k_search/modular/world/cycle.py`
- Test: `tests/modular/world/test_cycle.py`

**Step 1: Create test directory and write test**

```bash
mkdir -p tests/modular/world
touch tests/modular/world/__init__.py
```

File: `tests/modular/world/test_cycle.py`

```python
"""Tests for Cycle behavior (best_round, succeeded)."""

from unittest.mock import MagicMock

from k_search.modular.world.cycle import Cycle


def _mock_round(success: bool, score: float) -> MagicMock:
    r = MagicMock()
    r.result.succeeded.return_value = success
    r.score = score
    return r


def test_best_round_returns_highest_scoring_success():
    r1 = _mock_round(success=True, score=0.5)
    r2 = _mock_round(success=False, score=0.9)  # failed, ignored
    r3 = _mock_round(success=True, score=0.8)
    cycle = Cycle(rounds=[r1, r2, r3])

    assert cycle.best_round is r3
    assert cycle.succeeded is True


def test_best_round_none_when_all_failed():
    cycle = Cycle(rounds=[_mock_round(False, 0.0), _mock_round(False, 0.0)])
    assert cycle.best_round is None
    assert cycle.succeeded is False
```

**Step 2: Run test (should fail)**

Run: `pytest tests/modular/world/test_cycle.py -v`

**Step 3: Implement Cycle**

File: `k_search/modular/world/cycle.py`

```python
"""Cycle dataclass for tracking attempt rounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.round import Round


@dataclass
class Cycle:
    """Result of attempting an Action - all rounds."""

    rounds: list[Round] = field(default_factory=list)

    @property
    def best_round(self) -> Round | None:
        """Return highest-scoring successful round, or None."""
        successful = [r for r in self.rounds if r.result.succeeded()]
        return max(successful, key=lambda r: r.score) if successful else None

    @property
    def succeeded(self) -> bool:
        """Return True if any round succeeded."""
        return any(r.result.succeeded() for r in self.rounds)
```

**Step 4: Run test (should pass)**

Run: `pytest tests/modular/world/test_cycle.py -v`

**Step 5: Commit**

```bash
git add k_search/modular/world/cycle.py tests/modular/world/
git commit -m "feat(modular): add Cycle dataclass"
```

---

## Task 4: Create Tree dataclass

**Files:**
- Create: `k_search/modular/world/tree.py`
- Test: `tests/modular/world/test_tree.py`

**Step 1: Write test for Tree behavior**

File: `tests/modular/world/test_tree.py`

```python
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
```

**Step 2: Run test (should fail)**

Run: `pytest tests/modular/world/test_tree.py -v`

**Step 3: Implement Tree**

File: `k_search/modular/world/tree.py`

```python
"""Tree dataclass for search state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from k_search.modular.world.node import Node


@dataclass
class Tree:
    """Search tree container."""

    root: Node
    annotations: dict[str, Any] | None = None

    def add_node(self, node: Node) -> None:
        """Add node to tree, attaching to its parent's children list."""
        if node.parent is None:
            raise ValueError("Cannot add node without parent (use root for root node)")
        node.parent.children.append(node)

    def get_frontier(self) -> list[Node]:
        """Return all nodes with status 'open'."""
        return [n for n in self._all_nodes() if n.status == "open"]

    def get_best_node(self) -> Node | None:
        """Return best completed node by score, or None."""
        completed = [
            n for n in self._all_nodes()
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

    def _all_nodes(self) -> list[Node]:
        """Return all nodes in tree via BFS."""
        result = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node)
            queue.extend(node.children)
        return result
```

**Step 4: Run test (should pass)**

Run: `pytest tests/modular/world/test_tree.py -v`

**Step 5: Commit**

```bash
git add k_search/modular/world/tree.py tests/modular/world/test_tree.py
git commit -m "feat(modular): add Tree dataclass"
```

---

## Task 5: Create WorldModel and StateFormatter protocols

**Files:**
- Create: `k_search/modular/protocols/world_model.py`
- Create: `k_search/modular/protocols/formatter.py`
- Modify: `k_search/modular/protocols/__init__.py`

No tests - protocols are just interface definitions.

**Step 1: Create WorldModel protocol**

File: `k_search/modular/protocols/world_model.py`

```python
"""WorldModel protocol for search tree operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class WorldModel(Protocol):
    """World model interface (P_world from the paper)."""

    def propose(self, tree: Tree, context: dict[str, Any] | None = None) -> Node:
        """Generate a new frontier node with action."""
        ...

    def select(self, tree: Tree, context: dict[str, Any] | None = None) -> Node:
        """Select a frontier node to pursue."""
        ...

    def update(self, tree: Tree, context: dict[str, Any] | None = None) -> None:
        """Update tree after cycle completes."""
        ...
```

**Step 2: Create StateFormatter protocol**

File: `k_search/modular/protocols/formatter.py`

```python
"""StateFormatter protocol for tree serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class StateFormatter(Protocol):
    """Tree serialization for LLM prompts."""

    def format_tree(self, tree: Tree) -> str:
        """Format tree for P_world prompt."""
        ...

    def format_node(self, node: Node) -> str:
        """Format single node for display."""
        ...
```

**Step 3: Update protocols/__init__.py**

Add to imports:
```python
from k_search.modular.protocols.world_model import WorldModel
from k_search.modular.protocols.formatter import StateFormatter
```

Add to `__all__`: `"WorldModel"`, `"StateFormatter"`

**Step 4: Commit**

```bash
git add k_search/modular/protocols/world_model.py k_search/modular/protocols/formatter.py k_search/modular/protocols/__init__.py
git commit -m "feat(modular): add WorldModel and StateFormatter protocols"
```

---

## Task 6: Update exports and verify

**Files:**
- Modify: `k_search/modular/world/__init__.py`
- Modify: `k_search/modular/__init__.py`

**Step 1: Export from world/__init__.py**

File: `k_search/modular/world/__init__.py`

```python
"""World module: search tree data structures."""

from k_search.modular.world.action import Action
from k_search.modular.world.cycle import Cycle
from k_search.modular.world.node import Node
from k_search.modular.world.round import Round
from k_search.modular.world.tree import Tree

__all__ = [
    "Action",
    "Cycle",
    "Node",
    "Round",
    "Tree",
]
```

**Step 2: Update modular/__init__.py**

Add imports:
```python
from k_search.modular.world import Action, Cycle, Node, Tree
```

Add to `__all__`: `"Action"`, `"Cycle"`, `"Node"`, `"Tree"`, `"WorldModel"`, `"StateFormatter"`

**Step 3: Verify and run tests**

```bash
python -c "from k_search.modular import Action, Cycle, Node, Tree, Round, WorldModel, StateFormatter; print('OK')"
pytest tests/modular/ -v
ty check k_search/modular/world/
```

**Step 4: Commit**

```bash
git add k_search/modular/world/__init__.py k_search/modular/__init__.py
git commit -m "feat(modular): export world types from main module"
```

---

## Files Summary

**Create:**
- `k_search/modular/world/__init__.py`
- `k_search/modular/world/action.py`
- `k_search/modular/world/cycle.py`
- `k_search/modular/world/node.py`
- `k_search/modular/world/tree.py`
- `k_search/modular/protocols/world_model.py`
- `k_search/modular/protocols/formatter.py`
- `tests/modular/world/__init__.py`
- `tests/modular/world/test_cycle.py`
- `tests/modular/world/test_tree.py`

**Move:**
- `k_search/modular/round.py` → `k_search/modular/world/round.py`

**Modify:**
- `k_search/modular/__init__.py`
- `k_search/modular/protocols/__init__.py`
- `k_search/modular/loop.py`
- `k_search/modular/prompts.py`
- `k_search/modular/artifacts/*.py`
- `k_search/modular/protocols/artifact_store.py`
- `k_search/modular/protocols/feedback_provider.py`
- `k_search/modular/adapters/gpu_mode/task_definition.py`
