# Node/Tree Simplification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify Node to a pure structural container and remove Tree class entirely.

**Architecture:** Base Node keeps only `parent` and `children`. Domain-specific fields (id, status, action, cycle) move to V1Node subclass. Tree class deleted; tree operations become standalone helper functions.

**Tech Stack:** Python dataclasses

**Spec:** `docs/plans/2026-03-12-node-tree-simplification-design.md`

---

## Chunk 1: Delete Unused Modules

Delete entire directories and files that depend on removed Node/Tree properties.

### Task 1: Delete unused modules

**Files to delete:**
- `k_search/modular/executors/` (entire directory)
- `k_search/modular/formatters/` (entire directory)
- `k_search/modular/world_models/` (entire directory)
- `k_search/modular/world/tree.py`
- `tests/modular/executors/` (entire directory)
- `tests/modular/world_models/` (entire directory)
- `tests/modular/world/test_tree.py`
- `tests/modular/test_formatters.py`

- [ ] **Step 1: Delete source directories**

```bash
rm -rf k_search/modular/executors/
rm -rf k_search/modular/formatters/
rm -rf k_search/modular/world_models/
rm k_search/modular/world/tree.py
```

- [ ] **Step 2: Delete test directories and files**

```bash
rm -rf tests/modular/executors/
rm -rf tests/modular/world_models/
rm tests/modular/world/test_tree.py
rm tests/modular/test_formatters.py
```

---

## Chunk 2: Simplify Node

### Task 2: Simplify Node class

**Files:**
- Modify: `k_search/modular/world/node.py`

- [ ] **Step 1: Replace node.py contents**

```python
"""Node dataclass for search tree."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """Base tree node with parent/children structure only."""

    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
```

---

## Chunk 3: Update world/__init__.py

### Task 3: Remove Tree export

**Files:**
- Modify: `k_search/modular/world/__init__.py`

- [ ] **Step 1: Remove Tree import and export**

```python
"""World module: search tree data structures."""

from k_search.modular.world.action import Action
from k_search.modular.world.cycle import Cycle
from k_search.modular.world.node import Node
from k_search.modular.world.parse_result import ParseResult
from k_search.modular.world.round import Round

__all__ = [
    "Action",
    "Cycle",
    "Node",
    "ParseResult",
    "Round",
]
```

---

## Chunk 4: Update run.py

### Task 4: Update V1Node and V1SequentialExecutor

**Files:**
- Modify: `scripts/gpu_mode_modular_k_search/run.py`

- [ ] **Step 1: Remove Tree import (line 55)**

Delete this line:
```python
from k_search.modular.world.tree import Tree
```

- [ ] **Step 2: Add fields to V1Node**

Change V1Node from:
```python
@dataclass
class V1Node(Node):
    """V1-specific node with v1 ID mapping."""

    action: V1Action | None = None
    node_id: str = ""
    parent_id: str = ""
    parent_is_root: bool = False
```

To:
```python
@dataclass
class V1Node(Node):
    """V1-specific node with v1 ID mapping."""

    status: str = ""
    action: V1Action | None = None
    cycle: Cycle | None = None
    node_id: str = ""
    parent_id: str = ""
    parent_is_root: bool = False
```

- [ ] **Step 3: Add helper functions after V1Node class**

Add these functions after the V1Node class definition:

```python
def get_path_to_root(node: Node) -> list[Node]:
    """Return path from node to root (inclusive)."""
    path = []
    current: Node | None = node
    while current is not None:
        path.append(current)
        current = current.parent
    return path


def get_best_node(root: V1Node) -> V1Node | None:
    """Return best completed node by score, or None."""
    def collect_nodes(node: Node) -> list[V1Node]:
        result = [node] if isinstance(node, V1Node) else []
        for child in node.children:
            result.extend(collect_nodes(child))
        return result

    completed = [
        n for n in collect_nodes(root)
        if n.status == "closed" and n.cycle and n.cycle.succeeded and n.cycle.best_round
    ]
    if not completed:
        return None
    return max(completed, key=lambda n: n.cycle.best_round.score)  # type: ignore[union-attr]
```

- [ ] **Step 4: Update V1SequentialExecutor.__init__ signature and body**

Change `tree: Tree` parameter to `root: V1Node` and update assignment:

In `__init__` parameters, change:
```python
tree: Tree,
```
To:
```python
root: V1Node,
```

In `__init__` body, change:
```python
self.tree = tree
```
To:
```python
self.root = root
```

- [ ] **Step 5: Update V1SelectContext usage**

Change:
```python
select_context = V1SelectContext(tree=self.tree)
```
To:
```python
select_context = V1SelectContext(root=self.root)
```

- [ ] **Step 6: Update V1ProposeContext usage**

Change:
```python
propose_context = V1ProposeContext(
    tree=self.tree,
```
To:
```python
propose_context = V1ProposeContext(
    root=self.root,
```

- [ ] **Step 7: Update V1UpdateContext usage**

Change:
```python
update_context = V1UpdateContext(
    tree=self.tree,
```
To:
```python
update_context = V1UpdateContext(
    root=self.root,
```

- [ ] **Step 8: Update get_best_node() call**

Change:
```python
return self.tree.get_best_node()
```
To:
```python
return get_best_node(self.root)
```

- [ ] **Step 9: Update _sync_frontier_from_manager method**

Change method signature from:
```python
def _sync_frontier_from_manager(self, tree: Tree) -> list[V1Node]:
```
To:
```python
def _sync_frontier_from_manager(self, root: V1Node) -> list[V1Node]:
```

Inside the method, change:
```python
tree.add_node(new_node)
```
To:
```python
new_node.parent.children.append(new_node)
```

And change:
```python
parent_node = self._node_id_map.get(parent_id, tree.root)
```
To:
```python
parent_node = self._node_id_map.get(parent_id, root)
```

- [ ] **Step 10: Update _sync_frontier_from_manager call site**

Change:
```python
new_nodes = self._sync_frontier_from_manager(context.tree)
```
To:
```python
new_nodes = self._sync_frontier_from_manager(context.root)
```

- [ ] **Step 11: Update context dataclasses**

Update V1ProposeContext, V1SelectContext, V1UpdateContext to use `root: V1Node` instead of `tree: Tree`:

```python
@dataclass
class V1ProposeContext:
    """Context for V1WorldModel.propose()."""

    root: V1Node
    round_idx: int
    current_code: str = ""


@dataclass
class V1SelectContext:
    """Context for V1WorldModel.select()."""

    root: V1Node


@dataclass
class V1UpdateContext:
    """Context for V1WorldModel.update()."""

    root: V1Node
    node: V1Node
    cycle: Cycle
    round_idx: int
    max_debug_improve_rounds: int
```

- [ ] **Step 12: Update main() Tree instantiation**

Change:
```python
tree = Tree(root=Node(status="closed"))
```
To:
```python
root = V1Node(status="closed")
```

And update the executor instantiation to pass `root=root` instead of `tree=tree`.

---

## Chunk 5: Verify and Commit

### Task 5: Run tests and commit

- [ ] **Step 1: Run remaining tests**

```bash
cd K-Search && python -m pytest tests/modular/ -v --tb=short
```

Expected: All remaining tests pass (tests for deleted modules are gone).

- [ ] **Step 2: Run type check**

```bash
cd K-Search && ty check k_search/modular/world/node.py scripts/gpu_mode_modular_k_search/run.py
```

- [ ] **Step 3: Commit all changes**

```bash
git add -A
git commit -m "refactor(modular): simplify Node and remove Tree class

- Simplify Node to pure structural container (parent, children only)
- Remove Tree class entirely - use root node reference directly
- Delete unused modules: executors, formatters, world_models
- Move domain fields (id, status, action, cycle) to V1Node subclass
- Add get_path_to_root() and get_best_node() helper functions

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
