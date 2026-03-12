# Tree Data Model Design

Design for search state representation in the V2 modular framework.

## Goals

1. Clean dataclass-based tree state (replacing JSON string manipulation from V1)
2. Simple, composable types: Tree, Node, Action, Cycle, Round
3. WorldModel protocol for propose/select/update operations
4. StateFormatter protocol for LLM prompt serialization

## Non-Goals

- Implementing world model logic (separate design)
- Formatter implementations (separate design)
- Persistence/serialization format (use dataclasses.asdict for now)

## Data Model

**Module location:** `k_search/modular/world/`

```
Tree
├── root: Node
└── annotations: dict | None

Node
├── parent: Node | None
├── children: list[Node]
├── status: "open" | "in_progress" | "closed"
├── action: Action | None
├── cycle: Cycle | None
└── annotations: dict | None

Action
├── title: str
└── annotations: dict | None

Cycle
├── rounds: list[Round]
├── best_round: Round | None  (property)
└── succeeded: bool  (property)

Round (existing, moved to world/)
├── impl, result, prompt, llm_response
├── prompt_tokens, completion_tokens, duration_secs
├── score, analysis
```

### Key Behaviors

- `Tree.get_frontier()` → traverse tree, collect nodes where `status == "open"`
- `Tree.get_best_node()` → traverse tree, find best completed node by score
- `Tree.get_path_to_root(node)` → follow parent refs up to root
- `Tree.add_node(node)` → attach node to its parent's children list
- `Cycle.best_round` → highest-scoring successful round
- `Cycle.succeeded` → any round succeeded

## Protocols

### WorldModel

Interface for P_world from the paper.

```python
class WorldModel(Protocol):
    def propose(self, tree: Tree, context: dict | None = None) -> Node:
        """Generate a new frontier node with action."""
        ...

    def select(self, tree: Tree, context: dict | None = None) -> Node:
        """Select a frontier node to pursue."""
        ...

    def update(self, tree: Tree, context: dict | None = None) -> None:
        """Update tree after cycle completes."""
        ...
```

- `propose` and `select` are separate operations
- Methods return single Node; call repeatedly if needed
- Termination logic lives in orchestrator, not protocol
- Context dict carries whatever the implementation needs

### StateFormatter

Tree serialization for LLM prompts.

```python
class StateFormatter(Protocol):
    def format_tree(self, tree: Tree) -> str:
        """Format tree for P_world prompt."""
        ...

    def format_node(self, node: Node) -> str:
        """Format single node for display."""
        ...
```

Decouples tree structure from LLM-facing representation.

## Module Structure

```
k_search/modular/
├── world/
│   ├── __init__.py      # exports all types
│   ├── action.py        # Action
│   ├── cycle.py         # Cycle
│   ├── node.py          # Node
│   ├── tree.py          # Tree
│   └── round.py         # Round (moved from modular/)
├── protocols/
│   ├── world_model.py   # WorldModel (new)
│   ├── formatter.py     # StateFormatter (new)
│   └── ...              # existing protocols
└── ...
```

## Orchestrator Flow

```python
tree = Tree(root=Node(parent=None, status="closed"))

while rounds < max_rounds:
    node = world_model.select(tree)
    node.status = "in_progress"

    cycle = execute_cycle(node)
    node.cycle = cycle
    node.status = "closed"

    world_model.update(tree, context={"node": node})

    new_node = world_model.propose(tree)
    tree.add_node(new_node)  # parent already set on node
```

## Relation to Existing Designs

- Replaces `SolutionTree`/`SolutionNode`/`ActionNode` from search-v2-design with simpler `Tree`/`Node`
- `Cycle` formalizes `cycle_outcomes: list[Round]` from llm-cycle-feedback-design
- Compatible with parallel-executor (wrap sync WorldModel for async)

## Future Considerations

**Not in scope for initial implementation:**
- Persistence/checkpointing (use `dataclasses.asdict()` for now)
- Specific WorldModel implementations (LLMWorldModel is separate design)
- Specific StateFormatter implementations (legacy_json, markdown are separate)
- Async variants for pipeline executor (wrap sync with `asyncio.to_thread`)

**Open questions deferred:**
- Error handling when select finds empty frontier
- Retry/fallback behavior on world model failures

**Naming cleanup:**
- Rename `EvaluationResult.succeeded()` to `succeeded` property for consistency with `Cycle.succeeded`

## References

- `2026-03-04-search-v2-design.md` - overall V2 architecture
- `2026-03-05-parallel-executor-design.md` - async execution model
- `2026-03-05-llm-cycle-feedback-design.md` - cycle-level failure analysis
