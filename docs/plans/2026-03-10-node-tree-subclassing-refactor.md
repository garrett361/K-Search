# Node/Tree/Action Subclassing Refactor

## Problem

`Node`, `Tree`, `Action`, and `Span` all have `annotations: dict[str, Any]` fields for flexible metadata extension. This design is problematic:

1. No type safety — any key/value allowed
2. No IDE autocomplete for annotation keys
3. Awkward access pattern: `node.annotations.get("key")` vs `node.key`
4. Invited over-engineering (Generic[T] + Pydantic TypeAdapter design)

## Solution

Remove `annotations` fields. Users extend via plain dataclass subclassing.

```python
# Before
node = Node(annotations={"v1_node_id": "123", "difficulty": 3})
d = node.annotations.get("difficulty", 0)

# After
@dataclass
class V1Node(Node):
    v1_node_id: str = ""
    difficulty: int = 3

node = V1Node(v1_node_id="123", difficulty=3)
d = node.difficulty
```

## Design Decisions

**Trees are homogeneous.** All nodes in a tree are the same type. This simplifies Tree — no factory pattern or generics needed.

**Callers construct nodes.** `Tree.split_node()` takes pre-constructed nodes, not dicts. Clearer ownership.

**`id` is public, `status` defaults to `""`.**
- `id` set by Tree when node is added
- `status` set explicitly by callers when needed

**O(1) node lookup.** Tree maintains `_nodes_by_id: dict[str, Node]`.

## Changes

### Node (`k_search/modular/world/node.py`)

```python
@dataclass
class Node:
    id: str = ""
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    status: str = ""
    action: Action | None = None
    cycle: Cycle | None = None
```

### Action (`k_search/modular/world/action.py`)

```python
@dataclass
class Action:
    title: str
```

### Tree (`k_search/modular/world/tree.py`)

```python
@dataclass
class Tree:
    root: Node
    _next_id: int = field(default=0, init=False)
    _nodes_by_id: dict[str, Node] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root.id = str(self._next_id)
        self._next_id += 1
        self._nodes_by_id[self.root.id] = self.root

    def add_node(self, node: Node) -> None:
        if node.parent is None:
            raise ValueError("Cannot add node without parent")
        node.id = str(self._next_id)
        self._next_id += 1
        self._nodes_by_id[node.id] = node
        node.parent.children.append(node)

    def get_node_by_id(self, id: str) -> Node | None:
        return self._nodes_by_id.get(id)

    def get_frontier(self) -> list[Node]:
        return [n for n in self._nodes_by_id.values() if n.status == "open"]

    def get_best_node(self) -> Node | None:
        completed = [
            n for n in self._nodes_by_id.values()
            if n.status == "closed" and n.cycle and n.cycle.succeeded
        ]
        if not completed:
            return None
        return max(completed, key=lambda n: n.cycle.best_round.score)

    def get_path_to_root(self, node: Node) -> list[Node]:
        path = []
        current: Node | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def split_node(self, node: Node, children: list[Node]) -> None:
        """Mark node closed and add children to tree."""
        node.status = "closed"
        for child in children:
            child.parent = node
            self.add_node(child)

    def delete_node(self, node: Node) -> None:
        node.status = "deleted"
```

**Removed:** `update_node()`, `_assign_id()`, `_all_nodes()`, `annotations` field.

### Span (`k_search/modular/span.py`)

```python
@dataclass
class Span:
    node: Node
    timer: Timer = field(default_factory=Timer)
```

**Removed:** `annotations` field.

### Formatter (`k_search/modular/formatters/simple.py`)

Remove annotation display logic (lines 51-53). Formatters can be subclassed if domain-specific display needed.

## Files to Update

| File | Changes |
|------|---------|
| `k_search/modular/world/node.py` | Remove `annotations`, rename `_id` to `id`, default `status=""` |
| `k_search/modular/world/action.py` | Remove `annotations` |
| `k_search/modular/world/tree.py` | Remove `annotations`, `update_node`, `_all_nodes`, `_assign_id`; add `_nodes_by_id`; simplify `split_node`; rename `_get_node_by_id` |
| `k_search/modular/span.py` | Remove `annotations` |
| `k_search/modular/formatters/simple.py` | Remove annotation display (lines 51-53) |
| `scripts/gpu_mode_modular_k_search/run.py` | Define V1Node/V1Action subclasses, update all annotation access |
| `tests/modular/world/test_tree.py` | Update tests for removed/changed methods |
| `tests/modular/test_formatters.py` | Remove annotation assertions |

## Docs to Delete

- `docs/plans/2026-03-05-typed-annotations-design.md` — superseded by this simpler approach

## Migration Pattern

Call sites that used annotations:

```python
# Before
node_annot = node.annotations or {}
difficulty = node_annot.get("difficulty", 3)

# After (assuming V1Node)
difficulty = node.difficulty
```

```python
# Before
tree.split_node(root, [{"title": "A", "annotations": {"priority": "high"}}])

# After
tree.split_node(root, [V1Node(action=Action(title="A"), priority="high")])
```
