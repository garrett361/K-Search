# Impl 07: Parsing & Formatting Layer

**Goal:** Bridge between Tree data model and LLM interactions - tool schemas for tree operations, formatters for prompts, result types for fallible operations.

**Architecture:** Tool calling via OpenAI API (no JSON parsing needed). Tree stays Node-based, tools.py handles ID resolution for LLM communication.

## Design Decisions

| Decision | Choice |
|----------|--------|
| LLM output format | OpenAI-style tool calling (not JSON parsing) |
| Node IDs | Zero-indexed sequential ("0", "1", ...), assigned by Tree |
| ID counter | `Tree._next_id` counter, assigns at root creation and add_node |
| Tree methods | Node-based (tools.py resolves IDs) |
| Error handling | `ParseResult[T]` returned by apply_tool_call |
| Tool configuration | `get_tree_tools(enabled)` for configurable subsets |
| Tool schema format | Dict-based (direct OpenAI JSON), no dataclass abstraction |

## Components

### 1. Node._id Addition

```python
@dataclass
class Node:
    _id: str = ""  # Assigned by Tree, for LLM communication only
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    status: str = "open"
    action: Action | None = None
    cycle: Cycle | None = None
    annotations: dict[str, Any] | None = None
```

### 2. Tree ID Management & Mutations

```python
@dataclass
class Tree:
    root: Node
    annotations: dict[str, Any] | None = None
    _next_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._assign_id(self.root)

    def _assign_id(self, node: Node) -> None:
        node._id = str(self._next_id)
        self._next_id += 1

    # Existing
    def add_node(self, node: Node) -> None:
        """Add node to tree. Assigns ID and attaches to parent."""
        if node.parent is None:
            raise ValueError("Cannot add node without parent")
        self._assign_id(node)
        node.parent.children.append(node)

    def get_frontier(self) -> list[Node]: ...
    def get_best_node(self) -> Node | None: ...
    def get_path_to_root(self, node: Node) -> list[Node]: ...

    # New - ID lookup
    def get_node_by_id(self, id: str) -> Node | None:
        """Look up node by ID. Returns None if not found."""
        for node in self._all_nodes():
            if node._id == id:
                return node
        return None

    # New - mutations
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

### 3. ParseResult[T]

```python
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass
class ParseResult(Generic[T]):
    """Result of parsing/applying a tool call."""
    success: bool
    value: T | None = None
    error: str | None = None

    @classmethod
    def ok(cls, value: T) -> "ParseResult[T]":
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error: str) -> "ParseResult[T]":
        return cls(success=False, error=error)
```

### 4. Tool Schemas & Application

```python
# tools.py

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
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "annotations": {"type": "object"},
                            },
                            "required": ["title"],
                        },
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
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of node to delete"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_node",
            "description": "Select a node to pursue next",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of node to select"},
                },
                "required": ["node_id"],
            },
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
        parent = tree.get_node_by_id(args.get("parent_id", ""))
        if parent is None:
            return ParseResult.fail(f"parent_id not found: {args.get('parent_id')}")
        node = Node(
            parent=parent,
            action=Action(title=args["title"], annotations=args.get("annotations")),
        )
        tree.add_node(node)
        return ParseResult.ok(node)

    if tool_name == "update_node":
        node = tree.get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        tree.update_node(node, args["annotations"])
        return ParseResult.ok(node)

    if tool_name == "split_node":
        node = tree.get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        children = tree.split_node(node, args.get("children", []))
        return ParseResult.ok(children[0] if children else node)

    if tool_name == "delete_node":
        node = tree.get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        tree.delete_node(node)
        return ParseResult.ok(node)

    if tool_name == "select_node":
        node = tree.get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        return ParseResult.ok(node)

    return ParseResult.fail(f"unknown tool: {tool_name}")
```

### 5. StateFormatter Implementations

```python
# formatters/simple.py
class SimpleStateFormatter:
    """Minimal tree formatting for LLM prompts."""

    def format_tree(self, tree: Tree) -> str:
        lines = []
        for node in tree._all_nodes():
            lines.append(self.format_node(node))
        return "\n".join(lines)

    def format_node(self, node: Node) -> str:
        status = node.status
        title = node.action.title if node.action else "(root)"
        score = ""
        if node.cycle and node.cycle.best_round:
            score = f" (score: {node.cycle.best_round.score:.2f})"
        return f"[{node._id}] {status}: {title}{score}"

# formatters/legacy_json.py
class LegacyJSONFormatter:
    """V1-compatible JSON format for parity validation."""

    def format_tree(self, tree: Tree) -> str:
        return json.dumps(self._tree_to_dict(tree), indent=2)

    def format_node(self, node: Node) -> str:
        return json.dumps(self._node_to_dict(node))

    def _tree_to_dict(self, tree: Tree) -> dict[str, Any]:
        """Convert Tree to v1-compatible dict structure."""
        ...

    def _node_to_dict(self, node: Node) -> dict[str, Any]:
        """Convert Node to v1-compatible dict structure."""
        ...
```

## Files

**Modify:**
- `k_search/modular/world/node.py` - add `_id` field
- `k_search/modular/world/tree.py` - add `_next_id`, `__post_init__`, `_assign_id`, `get_node_by_id`, `update_node`, `split_node`, `delete_node`

**Create:**
- `k_search/modular/world/parse_result.py` - ParseResult[T]
- `k_search/modular/world/tools.py` - TREE_TOOLS, get_tree_tools, apply_tool_call
- `k_search/modular/formatters/__init__.py`
- `k_search/modular/formatters/simple.py` - SimpleStateFormatter
- `k_search/modular/formatters/legacy_json.py` - LegacyJSONFormatter

## Verification

1. `pytest tests/modular/world/` - existing tests pass
2. New unit tests:
   - `test_tree.py` - ID assignment, get_node_by_id, update_node, split_node, delete_node
   - `test_tools.py` - apply_tool_call with valid/invalid IDs, get_tree_tools filtering
   - `test_formatters.py` - SimpleStateFormatter output, LegacyJSONFormatter matches v1 schema
3. API compatibility test (`@pytest.mark.api`, can skip in CI):
   - `test_tool_calling_api.py` - Verify OpenAI SDK tool calling works with configured model
   - Call LLM with TREE_TOOLS, verify response has tool_calls, verify args parse correctly
4. `ty check k_search/modular/world/`
5. `ruff check k_search/modular/`

## Next: Impl 08

LLMWorldModel implementation using these tools:
- Calls LLM with TREE_TOOLS (via get_tree_tools)
- Applies tool calls via apply_tool_call
- Handles retries with error feedback from ParseResult
- Supports both LLM selection (select_node tool) and deterministic selection (policy)
