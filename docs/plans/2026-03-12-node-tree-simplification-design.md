# Node/Tree Simplification Design

Simplify base `Node` and `Tree` classes to pure structural containers. Move domain-specific properties (id, status, action, cycle) and methods (get_frontier, get_best_node, etc.) to subclasses.

## Motivation

The current `Node` class has a TODO (lines 13-27) suggesting this refactor. Benefits:
- Base classes become reusable for different search domains
- Original K-Search uses arbitrary string node IDs (`"root"`, `"m0"`, etc.) which conflicts with the current auto-increment integer scheme
- Cleaner separation of tree structure from domain data

## Design

### Base Node (after)

```python
@dataclass
class Node:
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
```

### Base Tree (after)

```python
@dataclass
class Tree:
    root: Node

    def add_node(self, node: Node) -> None:
        """Add node to tree, attaching to parent's children list."""
        if node.parent is None:
            raise ValueError("Cannot add node without parent")
        node.parent.children.append(node)

    def get_path_to_root(self, node: Node) -> list[Node]:
        """Return path from node to root (inclusive)."""
        path = []
        current: Node | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path
```

### Removed from base classes

| Removed from Node | Removed from Tree |
|-------------------|-------------------|
| `id: str` | `_next_id: int` |
| `status: str` | `_nodes_by_id: dict` |
| `action: Action \| None` | `get_node_by_id()` |
| `cycle: Cycle \| None` | `get_frontier()` |
| | `get_best_node()` |
| | `split_node()` |
| | `delete_node()` |

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

Note: V1Node uses `node_id` (arbitrary string like `"root"`, `"m0"`) rather than the base `id` (auto-incremented integer). The base `id` field is not needed.

### V1Tree in run.py

Add a `V1Tree` subclass with the `get_best_node()` method:

```python
@dataclass
class V1Tree(Tree):
    """Tree with V1-specific methods."""

    def get_best_node(self) -> V1Node | None:
        """Return best completed node by score, or None."""
        def collect_nodes(node: Node) -> list[V1Node]:
            result = [node] if isinstance(node, V1Node) else []
            for child in node.children:
                result.extend(collect_nodes(child))
            return result

        completed = [
            n for n in collect_nodes(self.root)
            if n.status == "closed" and n.cycle and n.cycle.succeeded
        ]
        if not completed:
            return None
        return max(completed, key=lambda n: n.cycle.best_round.score)
```

## Files to Delete

Unused reference implementations that depend on removed Node/Tree properties:

| File/Directory | Reason |
|----------------|--------|
| `k_search/modular/executors/` | Unused executor module (entire directory) |
| `k_search/modular/formatters/` | Unused formatter module (entire directory) |
| `k_search/modular/world_models/` | Unused world model module (entire directory) |
| `tests/modular/executors/` | Tests for deleted code (entire directory) |
| `tests/modular/world_models/` | Tests for deleted code (entire directory) |
| `tests/modular/test_formatters.py` | Tests for deleted code |

## Files to Modify

| File | Changes |
|------|---------|
| `k_search/modular/world/node.py` | Remove id, status, action, cycle |
| `k_search/modular/world/tree.py` | Remove all methods except add_node, get_path_to_root |
| `tests/modular/world/test_tree.py` | Delete 6 tests, keep 3 structural tests |
| `scripts/gpu_mode_modular_k_search/run.py` | Add removed fields to V1Node, add V1Tree subclass with get_best_node() |

### Tests to keep in test_tree.py

- `test_add_node_attaches_to_parent_children`
- `test_add_node_errors_on_orphan`
- `test_get_path_to_root`

### Tests to delete in test_tree.py

- `test_get_frontier_returns_open_nodes`
- `test_get_best_node_by_score`
- `test_tree_assigns_sequential_ids`
- `test_get_node_by_id`
- `test_split_node_adds_children`
- `test_delete_node_soft_deletes`

## Non-goals

- Creating shared `SearchNode`/`SearchTree` subclasses — defer until we know what abstractions are actually needed
- Adding ID-based lookup to base Tree — can be added later if needed
