# LLMWorldModel Design

Concrete implementation of `WorldModel` protocol using LLM + tool calling.

## Overview

LLMWorldModel manages search tree evolution via:
- `propose`: LLM generates new frontier nodes (forced tool call)
- `select`: Deterministic highest-score selection (no LLM)
- `update`: LLM applies tree modifications post-cycle (multiple tool calls)

## Class Structure

```python
class LLMWorldModel:
    def __init__(
        self,
        llm: LLMCall,
        formatter: StateFormatter,
        task: TaskDefinition,
    ):
        self._llm = llm
        self._formatter = formatter
        self._task = task

    def propose(self, tree: Tree, context: dict | None = None) -> Node:
        """Call LLM with forced insert_node tool, apply result."""

    def select(self, tree: Tree, context: dict | None = None) -> Node:
        """Return highest-scoring open node. No LLM call."""

    def update(self, tree: Tree, context: dict | None = None) -> None:
        """Call LLM, apply any returned tool calls."""
```

## Method Details

### propose()

1. Build prompt: tree state (via formatter) + task spec
2. Call LLM with `tools=[insert_node]`, `tool_choice=forced`
3. Parse single tool call from response
4. Apply via `apply_tool_call(tree, "insert_node", args)`
5. Return new node (or raise on failure)

### select()

Deterministic - no LLM:
```python
def select(self, tree: Tree, context: dict | None = None) -> Node:
    frontier = tree.get_frontier()
    if not frontier:
        raise WorldModelError("empty frontier")
    return max(frontier, key=lambda n: self._get_node_score(n))
```

Matches V1's `choose_next_action_node_id()` behavior.

### update()

1. Build prompt: tree state + completed node + cycle results (from context)
2. Call LLM with `tools=[update_node, insert_node, split_node, delete_node]`
3. No forced tool_choice - LLM decides what operations needed
4. Parse tool calls (may be 0, 1, or many)
5. Apply each via `apply_tool_call()`, raise on first failure

Typical operations:
- `update_node` on completed node (record outcome)
- `insert_node` under completed node (continuation children)

## Data Flow

```
Orchestrator                         LLMWorldModel
    │
    ├─► propose(tree, context={})
    │       ├─► formatter.format_tree(tree)
    │       ├─► build propose prompt
    │       ├─► llm(prompt, tools=[insert_node], tool_choice=forced)
    │       ├─► apply_tool_call(tree, "insert_node", args)
    │       └─► return new_node
    │
    ├─► select(tree, context={})
    │       ├─► tree.get_frontier()
    │       └─► return max(frontier, key=score)
    │
    ├─► [execute cycle on selected_node]
    │
    └─► update(tree, context={"node": node, "cycle": cycle})
            ├─► formatter.format_tree(tree)
            ├─► build update prompt (includes cycle results)
            ├─► llm(prompt, tools=[update_node, insert_node, ...])
            ├─► for each tool_call: apply_tool_call(tree, ...)
            └─► return None
```

## Error Handling

- `ParseResult.fail` from `apply_tool_call()` → raise `WorldModelError`
- No retry logic in v1 - fail fast, surface problems
- TODO: Add retry with feedback (subclasses can override)

## Tool Usage

Tools from `TREE_TOOLS` via `get_tree_tools(enabled=...)`:

| Method | Tools Enabled | Forced |
|--------|--------------|--------|
| propose | insert_node | Yes |
| select | (none - deterministic) | N/A |
| update | update_node, insert_node, split_node, delete_node | No |

## File Structure

```
k_search/modular/
├── protocols/
│   └── world_model.py        # existing protocol (unchanged)
├── world_models/             # NEW
│   ├── __init__.py
│   └── llm.py                # LLMWorldModel, WorldModelError
├── world/                    # existing
├── metrics/
├── artifacts/
└── formatters/
```

## Dependencies

**Existing (no changes):**
- `Tree`, `Node`, `Action` from `world/`
- `apply_tool_call`, `get_tree_tools`, `ParseResult` from `world/tools.py`
- `StateFormatter` protocol
- `TaskDefinition` protocol

**New:**
- `WorldModelError` exception (in `world_models/llm.py`)

## LLM Call Signature

The `llm` parameter must support tool calling:

```python
LLMCall = Callable[
    [str, list[dict], dict | None],  # prompt, tools, tool_choice
    LLMResponse  # has .tool_calls attribute
]
```

Exact signature TBD based on current llm_utils patterns. May need adapter for OpenAI client.

## Testing

```python
# tests/modular/world_models/test_llm.py

def test_propose_inserts_node():
    # Mock LLM returns insert_node tool call
    # Verify node added to tree

def test_select_returns_highest_score():
    # Tree with multiple open nodes at different scores
    # Verify highest returned

def test_select_raises_on_empty_frontier():
    # Tree with no open nodes
    # Verify WorldModelError raised

def test_update_applies_multiple_tool_calls():
    # Mock LLM returns update_node + insert_node
    # Verify both applied in order

def test_propose_raises_on_tool_failure():
    # Mock LLM returns invalid parent_id
    # Verify WorldModelError raised

def test_update_with_no_tool_calls():
    # Mock LLM returns no tool calls
    # Verify no error, tree unchanged
```

## Deferred

- Retry logic with LLM feedback on failure
- `max_new_nodes_per_edit` guard
- LLM-based select (alternative to deterministic)
- Prompt abstractions / system prompt configuration
- V1 features: difficulty gating, continuation enforcement validation

## References

- `protocols/world_model.py` - protocol definition
- `world/tools.py` - TREE_TOOLS, apply_tool_call
- `kernel_generators/world_model_manager.py` - V1 implementation
- `2026-03-05-tree-data-model-design.md` - tree data structures
