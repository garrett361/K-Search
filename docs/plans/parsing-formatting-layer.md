# Impl 07: Parsing & Formatting Layer

**Goal:** Bridge between Tree data model and LLM interactions — tool schemas for tree operations, formatters for prompts, result types for fallible operations.

**Architecture:** Tool calling via OpenAI API (no JSON parsing needed). Tree stays Node-based, tools.py handles ID resolution for LLM communication.

## Design Decisions

| Decision | Choice |
|----------|--------|
| LLM output format | OpenAI-style tool calling (not JSON parsing) |
| Node IDs | Private `_id` field, assigned by Tree, for LLM communication only |
| Tree methods | Node-based (tools.py resolves IDs) |
| Error handling | `ParseResult[T]` returned by apply_tool_call, LLMWorldModel handles retry/raise |
| Formatter | Include LegacyJSONFormatter for V1 parity validation |

## Components

### 1. Node._id Addition

```python
@dataclass
class Node:
    _id: str = ""  # Private - assigned by Tree.add_node, for LLM communication
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    status: str = "open"
    action: Action | None = None
    cycle: Cycle | None = None
    annotations: dict[str, Any] | None = None
```

### 2. Tree Methods

```python
class Tree:
    # Existing
    def add_node(self, node: Node) -> None
    def get_frontier(self) -> list[Node]
    def get_best_node(self) -> Node | None
    def get_path_to_root(self, node: Node) -> list[Node]

    # New - ID management (private, for LLM)
    def _get_node_by_id(self, id: str) -> Node | None
        """Look up node by ID. For LLM tool call resolution only."""

    # New - mutation methods (Node-based)
    def update_node(self, node: Node, annotations: dict) -> None
    def split_node(self, node: Node, children: list[Node]) -> None
    def delete_node(self, node: Node) -> None
```

### 3. ParseResult[T]

```python
@dataclass
class ParseResult(Generic[T]):
    success: bool
    value: T | None = None
    error: str | None = None
```

### 4. Tool Schemas & Application

```python
# tools.py

TREE_TOOLS = [
    {
        "name": "insert_node",
        "description": "Add a new action node to the tree",
        "parameters": {
            "parent_id": {"type": "string"},
            "title": {"type": "string"},
            "annotations": {"type": "object"},  # optional
        }
    },
    {
        "name": "update_node",
        "parameters": {
            "node_id": {"type": "string"},
            "annotations": {"type": "object"},
        }
    },
    {
        "name": "split_node",
        "parameters": {
            "node_id": {"type": "string"},
            "children": {"type": "array", "items": {"type": "object"}},  # [{title, annotations}, ...]
        }
    },
    {
        "name": "delete_node",
        "parameters": {"node_id": {"type": "string"}}
    },
    {
        "name": "select_node",
        "parameters": {"node_id": {"type": "string"}}
    },
]

def apply_tool_call(tree: Tree, tool_name: str, args: dict) -> ParseResult[Node]:
    """Route tool call to Tree method. Returns ParseResult for retry handling."""
    ...
```

### 5. StateFormatter Implementations

```python
# formatters/simple.py
class SimpleStateFormatter:
    def format_tree(self, tree: Tree) -> str: ...
    def format_node(self, node: Node) -> str: ...

# formatters/legacy_json.py
class LegacyJSONFormatter:
    """V1-compatible JSON schema for parity validation."""
    def format_tree(self, tree: Tree) -> str: ...
    def format_node(self, node: Node) -> str: ...
```

## Files

**Modify:**
- `k_search/modular/world/node.py` - add `_id` field
- `k_search/modular/world/tree.py` - add `_get_node_by_id`, ID assignment in add_node, `update_node`, `split_node`, `delete_node`

**Create:**
- `k_search/modular/world/parse_result.py` - ParseResult[T]
- `k_search/modular/world/tools.py` - TREE_TOOLS, apply_tool_call
- `k_search/modular/formatters/__init__.py`
- `k_search/modular/formatters/simple.py` - SimpleStateFormatter
- `k_search/modular/formatters/legacy_json.py` - LegacyJSONFormatter

## Verification

1. `pytest tests/modular/` - existing tests pass
2. New unit tests for:
   - Tree mutation methods (update_node, split_node, delete_node)
   - apply_tool_call with valid/invalid IDs
   - SimpleStateFormatter output
   - LegacyJSONFormatter matches V1 schema structure
3. E2E test verifying native tool calling works with configured model:
   - Call LLM with TREE_TOOLS
   - Verify response contains structured tool_calls
   - Verify tool call args parse correctly
4. `ty check k_search/modular/world/`
5. `ruff check k_search/modular/`

## Next: Impl 08

LLMWorldModel implementation using these tools:
- Calls LLM with TREE_TOOLS
- Applies tool calls via apply_tool_call
- Handles retries with error feedback
- Raises after max retries
