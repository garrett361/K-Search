# Search V2 Design

Clean reimplementation of the K-Search algorithm with better abstractions, encapsulation, and logging.

## Executive Summary

Search V2 replaces the current K-Search implementation (`world_model_manager.py` + `kernel_generator_world_model.py`) with a cleaner architecture that:

1. **Separates data from serialization** — Tree state lives in pure dataclasses; formatting for LLM prompts is handled by pluggable `StateFormatter` implementations
2. **Makes the world model explicit** — `LLMWorldModel` implements `ActionSelector` protocol, making P_world from the paper a first-class concept
3. **Adds explicit parse results** — No silent failures; every LLM response parse returns `ParseResult[T]` with success/failure and error context
4. **Centralizes logging** — Every decision, retry, and fallback is logged at appropriate levels
5. **Enables experimentation** — Swap formatters (JSON vs Markdown), selectors (LLM vs UCB vs random), without touching core logic

**Migration strategy**: Build V2 alongside V1, validate parity via extensive tests, then switch over.

---

## Current Implementation Review

### File Layout (V1)

```
k_search/kernel_generators/
├── kernel_generator.py              # Base generator (no world model)
├── kernel_generator_world_model.py  # Main loop with world model
├── world_model.py                   # JSON schema, prompts, parsing (~1700 lines)
├── world_model_manager.py           # State management, selection, edits (~1400 lines)
└── world_model_prompts.py           # Prompt templates
```

### Key V1 Concepts

| Concept | V1 Location | Issue |
|---------|-------------|-------|
| Tree state | `_world_models: Dict[str, str]` (raw JSON strings) | State stored as serialized JSON |
| Node schema | `_normalize_world_model_obj()` in `world_model.py:1093-1379` | 300-line normalization function |
| Action selection | `WorldModelSelectionPolicy` + utility function | Mixed with state management |
| Parsing | `try_parse_*` functions return `Optional[T]` | Silent failures on None |
| Fallbacks | `_fallback_insert_*` methods scattered | Ad-hoc, inconsistent |
| Edit operations | `DecisionTreeEditOps` applied to JSON string | Fragile string manipulation |

### V1 Data Flow

```
┌─────────────────┐
│  Raw JSON str   │ ← stored in _world_models dict
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    ▼         ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│ dump   │ │ load +   │ │ embed in │
│ (ser.) │ │ normalize│ │ prompts  │
└────────┘ └──────────┘ └──────────┘
```

### V1 Pain Points

1. **State is serialized JSON** — Must parse/serialize on every access
2. **Normalization is defensive** — Assumes LLM output is adversarial; 300+ lines of clamping/validation
3. **No separation of concerns** — Parsing, validation, selection, persistence all in same class
4. **Silent parse failures** — `try_parse_*` returns `None`, caller must check
5. **Scattered fallbacks** — `_fallback_insert_min_child`, `_fallback_insert_best_node_child` defined inline
6. **Hardcoded prompts** — Changing JSON schema requires editing prompts throughout

---

## V2 Architecture

```
┌───────────────────────────────────────────────┐
│           SolutionTree (dataclasses)          │
│  solutions: dict    actions: dict    metadata │
└──────────────────┬────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐       ┌──────────────┐
│ StateFormat. │       │ ActionSelect.│
│ (protocol)   │       │ (protocol)   │
│              │       │              │
│ Legacy JSON  │       │ LLMWorldModel│ ← P_world
│ Markdown     │       │ SimpleRefine │
└──────┬───────┘       └──────┬───────┘
       │                      │
       ▼                      ▼
  LLM prompt            ParseResult[T]
  (string)              (explicit errors)
```

### V1 vs V2 Comparison

| Aspect | V1 (Current) | V2 (New) |
|--------|--------------|----------|
| **State** | JSON string, parse on every access | `SolutionTree` dataclass |
| **Serialization** | Schema baked into prompts | `StateFormatter` protocol |
| **Parsing** | `Optional[T]`, silent failures | `ParseResult[T]`, explicit errors |
| **Selection** | Utility function in manager class | `ActionSelector` protocol |
| **Fallbacks** | Scattered `_fallback_*` methods | Centralized in `RetryConfig` |

### Main Loop Comparison

**V1 (simplified)**:
```
for round in range(max_rounds):
    wm_json = self._wm.get(task.name)           # Get JSON string
    obj = load_world_model_obj(wm_json)          # Parse to dict
    action = self._wm.choose_next_action(obj)    # Selection logic
    prompt = build_*_prompt(..., wm_json, ...)   # Schema in prompt
    code = llm.generate(prompt)                  # P_gen
    result = task.evaluate(code)                 # Eval
    self._wm.refine(...)                         # Edit JSON string
```

**V2 (simplified)**:
```
for round in range(max_rounds):
    actions = selector.select(tree, k=1)         # ActionSelector protocol
    prompt = formatter.format_tree(tree) + ...   # Formatter protocol
    code = codegen_llm.generate(prompt)          # P_gen
    result = evaluator.evaluate(code)            # Evaluator protocol
    selector.update(tree, action, outcome)       # Update dataclass directly
```

---

## Goals

1. **Code quality** — clean separation of concerns, well-encapsulated dataclasses
2. **Serialization flexibility** — decouple internal model from LLM-facing format (JSON, Markdown, etc.)
3. **Clear logging** — every decision, parse attempt, retry, and fallback is logged
4. **Parity first** — initial implementation matches current behavior exactly
5. **Drop-in compatible** — works with `TaskDefinition` from modular

## Non-Goals (for initial implementation)

- New serialization formats (Markdown, NL) — design supports them, but ship with JSON parity first
- New selection policies — implement current LLM world model behavior first
- Performance optimizations — correctness and clarity first

## Module Structure

> **Implementation Status:** The structure below is the *target design* for the full V2 loop with tree-based search and world model. As of 2026-03-05, only the foundation exists (`protocols/`, `adapters/`, `metrics/`, `artifacts/`, `loop.py`). The `model/`, `selectors/`, `formatters/`, and `parsing/` directories are NOT YET IMPLEMENTED.

```
k_search/modular/
├── __init__.py
│
├── model/                          # 🔲 NOT YET IMPLEMENTED
│   ├── __init__.py
│   ├── node.py                     # SolutionNode, ActionNode
│   ├── tree.py                     # SolutionTree
│   ├── action.py                   # ActionProposal, ActionOutcome
│   └── config.py                   # RetryConfig (extends existing config.py)
│
├── protocols/
│   ├── __init__.py
│   ├── action_selector.py          # 🔲 ActionSelector protocol
│   ├── formatter.py                # 🔲 StateFormatter protocol
│   ├── eval_result.py              # ✅ EvaluationResult protocol
│   ├── impl.py                     # ✅ Implementation protocol
│   ├── evaluator.py                # ✅ Evaluator protocol
│   ├── task_definition.py          # ✅ TaskDefinition protocol
│   └── ...                         # ✅ Other foundation protocols
│
├── selectors/                      # 🔲 NOT YET IMPLEMENTED
│   ├── __init__.py
│   ├── world_model.py              # LLMWorldModel (P_world) — DEFAULT
│   └── simple_refine.py            # Simple iterative refinement (no world model)
│
├── formatters/                     # 🔲 NOT YET IMPLEMENTED
│   ├── __init__.py
│   ├── legacy_json.py              # Current schema — for parity
│   ├── markdown.py                 # Future: Markdown format
│   └── natural_language.py         # Future: NL summaries
│
├── parsing/                        # 🔲 NOT YET IMPLEMENTED
│   ├── __init__.py
│   ├── result.py                   # ParseResult type
│   ├── json_parser.py              # JSON extraction and validation
│   └── legacy_schema.py            # Current schema parser/validator
│
├── search.py                       # 🔲 SearchOrchestrator (NOT YET IMPLEMENTED)
├── loop.py                         # ✅ Simple run_search() function (implemented)
└── integration.py                  # Adapters to TaskDefinition
```

## Core Data Model

### SolutionNode

```python
# model/node.py
from dataclasses import dataclass, field
from typing import Any
import time

@dataclass
class SolutionNode:
    """A solution in the search tree."""
    id: str
    parent_id: str | None

    # Solution content (references Implementation by ID or inline)
    solution_id: str | None = None
    solution_content: Any = None

    # Evaluation result
    eval_result: dict[str, Any] | None = None  # Metrics from EvaluationResult.get_metrics()
    status: str = "pending"  # pending, passed, failed

    # Tree structure
    depth: int = 0
    child_action_ids: list[str] = field(default_factory=list)

    # Metadata
    created_at: float = field(default_factory=time.time)
    round_index: int | None = None

    # World model annotations (from P_world)
    decision: str | None = None      # What decision this node represents
    choice: str | None = None        # What choice was made
    impacts: dict[str, Any] | None = None  # Predicted impacts (memory, compute, etc.)
```

### ActionNode

```python
@dataclass
class ActionNode:
    """A proposed action (optimization strategy) in the tree."""
    id: str
    parent_solution_id: str

    # Action description (from P_world)
    title: str
    description: str = ""
    rationale: str = ""

    # Predictions (from P_world)
    difficulty: int = 3  # 1-5
    predicted_score: float = 0.0
    expected_improvement: float | None = None

    # Status
    status: str = "open"  # open, in_progress, completed, abandoned

    # Result (if completed)
    result_solution_id: str | None = None
    actual_improvement: float | None = None

    # Metadata
    created_at: float = field(default_factory=time.time)
    round_index: int | None = None
```

### SolutionTree

```python
@dataclass
class SolutionTree:
    solutions: dict[str, SolutionNode] = field(default_factory=dict)
    actions: dict[str, ActionNode] = field(default_factory=dict)
    root_id: str = "root"
    active_leaf_id: str = "root"
    kernel_summary: str = ""
    open_questions: list[str] = field(default_factory=list)

    def add_solution(self, node: SolutionNode) -> None: ...
    def add_action(self, action: ActionNode) -> None: ...
    def get_frontier(self) -> list[ActionNode]: ...      # Open actions
    def get_best_solution(self) -> SolutionNode | None: ...
    def get_path_to_root(self, node_id: str) -> list[SolutionNode]: ...
```

## Protocols

```python
class StateFormatter(Protocol):
    def format_tree(self, tree: SolutionTree, context: dict | None = None) -> str: ...
    def format_frontier(self, actions: list[ActionNode]) -> str: ...

class ActionSelector(Protocol):
    def propose_actions(self, tree: SolutionTree, context: dict | None = None) -> list[ActionNode]: ...
    def select(self, tree: SolutionTree, k: int = 1) -> list[ActionNode]: ...
    def update(self, tree: SolutionTree, action: ActionNode, outcome: Round) -> None: ...

# LLM calls: just use Callable[[str], str] — no need for a protocol
LLMCall = Callable[[str], str]
```

## Parsing

All parsers return `ParseResult[T]` — explicit success/failure with error context for retries:

```python
@dataclass
class ParseResult(Generic[T]):
    success: bool
    value: T | None = None
    error: str | None = None
    raw_response: str = ""

def extract_json_object(text: str) -> ParseResult[dict]:
    """Try direct parse, then code blocks, then regex. Returns error on failure."""
    ...
```

## Selector Implementations

### LLMWorldModel (Default)

P_world from the paper. Uses LLM to propose actions, rank frontier, update tree.

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    include_error_in_retry: bool = True
    fallback_strategy: str = "refine_best"  # or "random", "highest_predicted"

class LLMWorldModel:
    def __init__(self, llm: LLMClient, formatter: StateFormatter, config: RetryConfig): ...

    # Each method: try LLM → parse with ParseResult → retry on failure → fallback
    def propose_actions(self, tree, context) -> list[ActionNode]: ...
    def select(self, tree, k=1) -> list[ActionNode]: ...
    def update(self, tree, action, outcome) -> None: ...

    # Fallbacks when LLM fails
    def _fallback_propose(self, tree) -> list[ActionNode]:  # Returns "refine best"
    def _fallback_select(self, tree, frontier, k) -> list[ActionNode]:  # By strategy
```

### SimpleRefineSelector (Alternative)

No world model — always proposes "refine current best". Matches `kernel_generator.py` behavior.

## Legacy JSON Formatter

Produces exact V1 schema for parity. Maps `SolutionTree` → V1 JSON structure:

```json
{
    "kernel_summary": "...",
    "open_questions": ["..."],
    "decision_tree": {
        "root_id": "root",
        "active_leaf_id": "...",
        "nodes": [{
            "node_id": "...", "parent_id": "...",
            "decision": "...", "choice": "...",
            "impacts": { "memory_bandwidth": {...}, ... },
            "solution_ref": { "solution_id": "...", "eval": {...} },
            "action": { "title": "...", "difficulty_1_to_5": 3, ... }
        }]
    }
}
```

## Search Orchestrator

```python
@dataclass
class SearchConfig:
    max_rounds: int = 100
    max_attempts_per_action: int = 3

class SearchOrchestrator:
    def __init__(self, task: TaskDefinition, evaluator: Evaluator,
                 codegen_llm: LLMCall, selector: ActionSelector,
                 formatter: StateFormatter, config: SearchConfig): ...

    def run(self) -> SolutionNode | None:
        self._initialize_tree()
        for round_idx in range(self.config.max_rounds):
            actions = self.selector.select(self.tree, k=1)
            if not actions: break
            outcome = self._execute_action(actions[0])
            self.selector.update(self.tree, actions[0], outcome)
        return self.tree.get_best_solution()

    def _execute_action(self, action) -> Round:
        # Generate code via P_gen, evaluate, retry on failure
        ...
```

## Integration with modular

Both V1 and V2 can use modular. Key distinction:
- `FeedbackProvider.for_world_model()` → metrics from one outcome (stored in node)
- `StateFormatter.format_tree()` → entire tree for P_world prompt (no overlap)

### Compatibility Matrix

| Component | V1 (current) | V1 + modular | V2 |
|-----------|--------------|---------------------|-----|
| Task loading | `GpuModeTriMulTask` | `GpuModeAdapter(GpuModeTriMulTask)` | `TaskDefinition` |
| Evaluation | `task.evaluate()` | `evaluator.evaluate()` | `evaluator.evaluate()` |
| Metrics extraction | `result.to_dict()` | `feedback_provider.for_world_model()` | `feedback_provider.for_world_model()` |
| Codegen feedback | `result.log_excerpt` | `feedback_provider.for_codegen()` | `feedback_provider.for_codegen()` |
| World model state | JSON string | JSON string | `SolutionTree` dataclass |
| Tree formatting | `dump_world_model_obj()` | `dump_world_model_obj()` | `StateFormatter.format_tree()` |
| Action selection | `WorldModelSelectionPolicy` | `WorldModelSelectionPolicy` | `ActionSelector` protocol |

### Migration Path

1. **modular foundation** — Implement protocols + GpuModeAdapter, V1 unchanged
2. **V1 adopts modular** (optional) — Wrap tasks in adapter, use FeedbackProvider
3. **V2 implementation** — Build modular, verify parity, switch over

## State Persistence

Dataclasses enable checkpointing via `dataclasses.asdict()` → JSON. Distinct from `LegacyJSONFormatter` (which produces V1-compatible LLM prompts).

## Logging

Every decision logged. Levels: INFO (rounds, selections, results), DEBUG (prompts, responses), WARNING (retries, fallbacks).

## Parity Testing Plan

**Goal**: V2 produces identical behavior to V1 when using `LegacyJSONFormatter` and `LLMWorldModel`.

### Test Categories

| Category | What to Test | How |
|----------|--------------|-----|
| **Formatter parity** | `LegacyJSONFormatter` output matches `dump_world_model_obj()` | Load V1 snapshots, compare JSON |
| **Parser parity** | `extract_json_object()` matches V1 `_extract_json_object()` | 50+ real LLM responses |
| **Selection parity** | Utility scores, difficulty filtering, ordering | V1 selection test cases |
| **Fallback parity** | All 3 strategies produce expected actions | Unit tests per strategy |
| **Integration replay** | Full runs with recorded LLM responses | 10+ recorded V1 runs |
| **Property tests** | Tree invariants (root exists, no orphans) | Hypothesis on random trees |
| **A/B comparison** | Same task, same seed → same results | 5 runs × 3 tasks |

### Key Invariants

- Root always exists with `parent_id=None`
- Every node's parent exists in tree
- Every action references existing solution
- `format_tree → parse → format_tree` is stable

## Future Extensions

After parity is achieved:

1. **New formatters**: MarkdownFormatter, NaturalLanguageFormatter
2. **New selectors**: Pure UCB (no LLM), hybrid approaches
3. **Parallel selection**: Select k > 1 actions, evaluate in parallel
4. **Artifact persistence**: Integrate with ArtifactStore from modular extensions
5. **LLM query mechanism**: Let LLM request additional context (top solutions, failure patterns) via QueryProvider protocol — see modular extensions §7

## References

- Current implementation: `k_search/kernel_generators/world_model_manager.py`
- Current schema: `k_search/kernel_generators/world_model.py:1093-1250`
- Task framework: `docs/plans/2026-03-04-task-framework-design.md`
- Task framework extensions: `docs/plans/2026-03-04-task-framework-extensions.md`
