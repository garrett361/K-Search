# Node/Tree Simplification Design

Simplify `Node` to a pure structural container and remove `Tree` class entirely. Domain-specific properties (id, status, action, cycle) move to subclasses. Tree operations become standalone helper functions.

## Motivation

The current `Node` class has a TODO (lines 13-27) suggesting this refactor. Benefits:
- Base Node becomes reusable for different search domains
- Original K-Search uses arbitrary string node IDs (`"root"`, `"m0"`, etc.) which conflicts with the current auto-increment integer scheme
- Cleaner separation of tree structure from domain data
- Tree class is unnecessary overhead — a tree is just a reference to its root node

## Design

### Base Node (after)

```python
@dataclass
class Node:
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
```

### Tree class

**Removed entirely.** A tree is represented by a reference to its root node.

| Before | After |
|--------|-------|
| `tree = Tree(root=V1Node())` | `root = V1Node()` |
| `tree.root` | `root` |
| `tree.add_node(node)` | `node.parent.children.append(node)` |
| `tree.get_path_to_root(node)` | `get_path_to_root(node)` (standalone function) |

### Removed from Node

| Field | Destination |
|-------|-------------|
| `id: str` | Subclass (V1Node uses `node_id` instead) |
| `status: str` | Subclass |
| `action: Action \| None` | Subclass |
| `cycle: Cycle \| None` | Subclass |

### V1Node in run.py (updated)

The existing `V1Node` subclass in `scripts/gpu_mode_modular_k_search/run.py` already has custom fields (`node_id`, `parent_id`, `parent_is_root`). It will be updated to include the removed base fields:

```python
@dataclass
class V1Node(Node):
    # Fields moved from base Node
    status: str = ""
    action: V1Action | None = None
    cycle: Cycle | None = None
    # Existing V1-specific fields (node_id used instead of base id)
    node_id: str = ""
    parent_id: str = ""
    parent_is_root: bool = False
```

### Helper functions in run.py

Add standalone functions for tree operations:

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
        if n.status == "closed" and n.cycle and n.cycle.succeeded
    ]
    if not completed:
        return None
    return max(completed, key=lambda n: n.cycle.best_round.score)
```

## Files to Delete

| File/Directory | Reason |
|----------------|--------|
| `k_search/modular/world/tree.py` | Tree class removed |
| `k_search/modular/executors/` | Unused executor module (entire directory) |
| `k_search/modular/formatters/` | Unused formatter module (entire directory) |
| `k_search/modular/world_models/` | Unused world model module (entire directory) |
| `tests/modular/executors/` | Tests for deleted code (entire directory) |
| `tests/modular/world_models/` | Tests for deleted code (entire directory) |
| `tests/modular/world/test_tree.py` | Tests for deleted Tree class |
| `tests/modular/test_formatters.py` | Tests for deleted code |

## Files to Modify

| File | Changes |
|------|---------|
| `k_search/modular/world/node.py` | Remove id, status, action, cycle; remove TODO comment |
| `k_search/modular/world/__init__.py` | Remove Tree export |
| `scripts/gpu_mode_modular_k_search/run.py` | Add fields to V1Node; replace `Tree` usage with root reference; add helper functions; update `self.tree` to `self.root` |

### run.py changes in detail

In `V1SequentialExecutor`:

1. Remove `from k_search.modular.world.tree import Tree`
2. Add `status`, `cycle` fields to `V1Node`
3. Change `self.tree = Tree(root=V1Node(...))` to `self.root = V1Node(...)`
4. Change `self.tree.root` to `self.root` (store node directly, not wrapped in Tree)
5. Change `self.tree.add_node(node)` to `node.parent.children.append(node)`
6. Change `self.tree.get_best_node()` to `get_best_node(self.root)`
7. Add `get_path_to_root()` and `get_best_node()` helper functions

## Non-goals

- Creating shared `SearchNode` subclass — defer until we know what abstractions are actually needed
- Keeping Tree class for organizational purposes — simpler to just use root reference
