# V2 Modular K-Search Design

**Status:** In Progress - Analysis Phase
**Date:** 2026-03-10
**Goal:** Create new v2 modular k-search implementation with clean protocols and abstractions.

**Location:** New directory `scripts/gpu_mode_modular_k_search_v2/`.

**Note on existing code:** The `WorldModel` protocol change (Phase 1) requires updating existing
scripts that use it. However, v2 implementation goes in the new directory. Existing scripts
continue to work after the protocol update.

## V1 Code Path Analysis

### Execution Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           V1 K-SEARCH EXECUTION FLOW                                    │
│                     (gpu_mode_modular_k_search/run.py)                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                                  ┌─────────────┐
                                  │   main()    │
                                  └──────┬──────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
            ▼                            ▼                            ▼
    ┌───────────────┐          ┌─────────────────┐          ┌────────────────┐
    │ GpuModeTask   │          │ WorldModelMgr   │          │ V1PromptBuilder│
    │ (task def)    │          │ (decision tree) │          │ (code prompts) │
    └───────┬───────┘          └────────┬────────┘          └───────┬────────┘
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ V1SequentialExec  │
                              │     .run()        │
                              └─────────┬─────────┘
                                        │
                                        ▼
═══════════════════════════════════════════════════════════════════════════════════════════
                              OUTER LOOP (max_rounds)
═══════════════════════════════════════════════════════════════════════════════════════════
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   │                   │
        ┌───────────────────┐           │                   │
        │ 1. PROPOSE PHASE  │           │                   │
        │ world_model.      │           │                   │
        │   propose()       │           │                   │
        └─────────┬─────────┘           │                   │
                  │                     │                   │
                  ▼                     │                   │
   ┌──────────────────────────────┐     │                   │
   │ ┌──────────────────────────┐ │     │                   │
   │ │    PROMPT #1 (ACTION)    │ │     │                   │
   │ │ build_decision_tree_     │ │     │                   │
   │ │   edit_prompt()          │ │     │                   │
   │ ├──────────────────────────┤ │     │                   │
   │ │ DEPS:                    │ │     │                   │
   │ │ • world_model_json       │ │     │                   │
   │ │ • definition_text        │ │     │                   │
   │ │ • current_tree_path      │ │     │                   │
   │ │ • open_frontier_nodes    │ │     │                   │
   │ │ • round_idx              │ │     │                   │
   │ └──────────────────────────┘ │     │                   │
   │          │                   │     │                   │
   │          ▼ LLM CALL          │     │                   │
   │ ┌──────────────────────────┐ │     │                   │
   │ │ → JSON edit ops          │ │     │                   │
   │ │ → New frontier nodes     │ │     │                   │
   │ └──────────────────────────┘ │     │                   │
   └──────────────────────────────┘     │                   │
                  │                     │                   │
                  ▼                     │                   │
        ┌───────────────────┐           │                   │
        │ 2. SELECT PHASE   │           │                   │
        │ world_model.      │           │                   │
        │   select()        │           │                   │
        └─────────┬─────────┘           │                   │
                  │                     │                   │
                  │ policy-based        │                   │
                  │ (no LLM call)       │                   │
                  │                     │                   │
                  ▼                     │                   │
        ┌───────────────────┐           │                   │
        │ Selected Node     │◄──────────┘                   │
        │ (action, context) │                               │
        └─────────┬─────────┘                               │
                  │                                         │
                  ▼                                         │
═══════════════════════════════════════════════════════════════════════════════════════════
                 CYCLE LOOP (_run_cycle - up to max_rounds_per_cycle)
═══════════════════════════════════════════════════════════════════════════════════════════
                  │                                         │
        ┌─────────┴─────────┐                               │
        │                   │                               │
        ▼                   │                               │
  ┌──────────────────────────────────────┐                  │
  │ ┌──────────────────────────────────┐ │                  │
  │ │    PROMPT #2 (CODE GEN)          │ │                  │
  │ │ V1PromptBuilder.build()          │ │                  │
  │ ├──────────────────────────────────┤ │                  │
  │ │ ROUTING LOGIC:                   │ │                  │
  │ │                                  │ │                  │
  │ │ if attempt == 0:                 │ │                  │
  │ │   if has_base_code:              │ │                  │
  │ │     → ACTION_PROMPT              │ │                  │
  │ │   else:                          │ │                  │
  │ │     → SPEC_WITH_ACTION_PROMPT    │ │                  │
  │ │                                  │ │                  │
  │ │ elif not has_passed:             │ │                  │
  │ │   if has_base_code:              │ │                  │
  │ │     → DEBUG_PROMPT               │ │                  │
  │ │   else:                          │ │                  │
  │ │     → DEBUG_FROM_SPEC_PROMPT     │ │                  │
  │ │                                  │ │                  │
  │ │ else: (has_passed)               │ │                  │
  │ │   if has_base_code:              │ │                  │
  │ │     → IMPROVE_PROMPT             │ │                  │
  │ │   else:                          │ │                  │
  │ │     → IMPROVE_FROM_SPEC_PROMPT   │ │                  │
  │ ├──────────────────────────────────┤ │                  │
  │ │ DEPS (all prompts):              │ │                  │
  │ │ • definition_text (spec)         │ │                  │
  │ │ • action_text                    │ │                  │
  │ │ • base_code (parent's best)      │ │                  │
  │ │ • current_code (last response)   │ │                  │
  │ │ • trace_logs (eval output)       │ │                  │
  │ │ • attempt / max_rounds           │ │                  │
  │ │ • target_gpu, language           │ │                  │
  │ │ • perf_summary                   │ │                  │
  │ │                                  │ │                  │
  │ │ + world_model_section (appended) │ │                  │
  │ └──────────────────────────────────┘ │                  │
  │              │                       │                  │
  │              ▼ LLM CALL              │                  │
  │ ┌──────────────────────────────────┐ │                  │
  │ │ → Generated Triton/CUDA code     │ │                  │
  │ └──────────────────────────────────┘ │                  │
  └──────────────────────────────────────┘                  │
                  │                                         │
                  ▼                                         │
        ┌───────────────────┐                               │
        │ 4. EVAL PHASE     │                               │
        │ evaluator.eval()  │                               │
        │ scorer.score()    │                               │
        └─────────┬─────────┘                               │
                  │                                         │
                  │ (no LLM - GPU execution)                │
                  │                                         │
                  ▼                                         │
        ┌───────────────────┐                               │
        │ Round(impl,       │                               │
        │   result, prompt, │                               │
        │   score, ...)     │                               │
        └─────────┬─────────┘                               │
                  │                                         │
                  ├──────────────────┐                      │
                  │                  │                      │
                  ▼                  ▼                      │
         ┌──────────────┐    ┌──────────────┐               │
         │ score > best │    │ no_improve   │               │
         │   → UPDATE   │    │   += 1       │               │
         │   best_score │    │              │               │
         └──────────────┘    └──────────────┘               │
                  │                  │                      │
                  │                  ▼                      │
                  │    ┌───────────────────────┐            │
                  │    │ no_improve >= stag?   │────────────┤
                  │    │   → BREAK CYCLE       │            │
                  │    └───────────────────────┘            │
                  │                                         │
                  └──────────────────┬──────────────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │ Cycle complete         │
                        │ → attach to Node       │
                        └───────────┬────────────┘
                                    │
                                    ▼
═══════════════════════════════════════════════════════════════════════════════════════════
                              5. UPDATE PHASE (world_model.update)
═══════════════════════════════════════════════════════════════════════════════════════════
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     │                     ▼
  ┌───────────────────────┐         │       ┌───────────────────────┐
  │ cycle_succeeded?      │         │       │ cycle FAILED          │
  │   → refine()          │         │       │   → note_action_      │
  │                       │         │       │     too_hard()        │
  └───────────┬───────────┘         │       └───────────┬───────────┘
              │                     │                   │
              ▼                     │                   ▼
  ┌──────────────────────────────┐  │   ┌──────────────────────────────┐
  │ ┌──────────────────────────┐ │  │   │ ┌──────────────────────────┐ │
  │ │  PROMPT #3a (REFINE)     │ │  │   │ │  PROMPT #3b (TOO HARD)   │ │
  │ │  build_decision_tree_    │ │  │   │ │  build_decision_tree_    │ │
  │ │    edit_prompt()         │ │  │   │ │    edit_prompt()         │ │
  │ ├──────────────────────────┤ │  │   │ ├──────────────────────────┤ │
  │ │ DEPS:                    │ │  │   │ │ DEPS:                    │ │
  │ │ • world_model_json       │ │  │   │ │ • world_model_json       │ │
  │ │ • definition_text        │ │  │   │ │ • definition_text        │ │
  │ │ • current_tree_path      │ │  │   │ │ • current_tree_path      │ │
  │ │ • chosen_action_text     │ │  │   │ │ • chosen_action_text     │ │
  │ │ • eval_result (PASSED)   │ │  │   │ │ • eval_result (if any)   │ │
  │ │ • current_code           │ │  │   │ │ • current_code           │ │
  │ │ • round_idx              │ │  │   │ │ • debug_and_improve_     │ │
  │ └──────────────────────────┘ │  │   │ │   round/max_rounds       │ │
  │              │               │  │   │ └──────────────────────────┘ │
  │              ▼ LLM CALL      │  │   │              │               │
  │ ┌──────────────────────────┐ │  │   │              ▼ LLM CALL      │
  │ │ → Update node scores,    │ │  │   │ ┌──────────────────────────┐ │
  │ │   insert child actions   │ │  │   │ │ → Downgrade scores,      │ │
  │ └──────────────────────────┘ │  │   │ │   maybe insert recovery  │ │
  └──────────────────────────────┘  │   │ └──────────────────────────┘ │
              │                     │   └──────────────────────────────┘
              │                     │                   │
              └─────────────────────┼───────────────────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │ rounds_used += cycle   │
                        │ → continue outer loop  │
                        └───────────┬────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ tree.get_best_node()│
                         │ → return best       │
                         └─────────────────────┘
```

### Prompt Dependency Summary

| PROMPT | KEY DEPENDENCIES |
|--------|------------------|
| #1 PROPOSE (decision_tree) | world_model_json, definition_text, open_frontier_nodes, current_tree_path, round_idx |
| #2 CODE GEN (6 variants) | definition_text, action_text, base_code (parent's best), current_code (last attempt), trace_logs, attempt, target_gpu, language, + world_model_section (appended) |
| #3a REFINE (on success)* | world_model_json, definition_text, chosen_action_text, current_tree_path, eval_result (PASSED), current_code |
| #3b TOO_HARD (on failure)* | world_model_json, definition_text, chosen_action_text, current_tree_path, debug_round/max_rounds, current_code |

*Prompts #3a/#3b are internal to `WorldModelManager.refine()` and `note_action_too_hard()`, not built by V1PromptBuilder. They're called from `V1WorldModel.update()` and are part of the WorldModel's internal decision tree logic.

### Data Flow

```
TaskDefinition ──┬── definition_text ──────────────────► ALL PROMPTS
                 │
                 └── scorer.score(result) ─────────────► best_score tracking

WorldModelMgr ───┬── world_model_json ─────────────────► #1, #3a, #3b
                 │
                 ├── get_tree_path_text() ─────────────► #1, #3a, #3b
                 │
                 └── render_world_model_section() ─────► #2 (appended)

Selected Node ───┬── action.title ─────────────────────► #2, #3a, #3b
(from select)    │
                 └── parent.cycle.best_round.response ─► #2 (base_code)

Cycle Rounds ────┬── llm_response (current_code) ──────► #2 (next iteration)
(in progress)    │
                 └── result.get_log() (trace_logs) ────► #2 (next iteration)

Evaluator ───────── EvalResult ────────────────────────► #3a (if passed), scoring
```

### Key V1 Components (to be replaced)

| V1 Component | Location | Responsibility |
|--------------|----------|----------------|
| `V1WorldModel` | run.py:108-318 | Wraps WorldModelManager, syncs JSON tree to modular Tree/Node |
| `V1PromptBuilder` | run.py:320-420 | 6-way prompt routing based on cycle state |
| `V1SequentialExecutor` | run.py:422-603 | Outer loop + cycle loop with stagnation detection |
| `WorldModelManager` | kernel_generators/world_model_manager.py | JSON decision tree, propose/refine/note_too_hard |
| World model prompts | kernel_generators/world_model.py | `build_decision_tree_edit_prompt()` |
| Code gen prompts | kernel_generators/world_model_prompts.py | ACTION/DEBUG/IMPROVE prompt templates |

### V1 Coupling Issues

1. **State coupling:** `base_code` comes from parent node's best round (tree structure dependency)
2. **Cross-cutting concerns:** `world_model_section` appended to code prompts
3. **Embedded logic:** Prompt selection is a 6-way branch in `V1PromptBuilder.build()`
4. **Implicit data flow:**
   - `current_code` accumulates through cycle (last LLM response)
   - `trace_logs` comes from previous eval result
   - Tree path text derived from V1's JSON structure

---

## V2 Design

### Design Decisions

1. **Keep `WorldModel` as single protocol** - Don't split into separate ActionProposer/ActionSelector/TreeUpdater. V1's propose/select/update share internal state and are designed to work together. Can revisit if needed.

2. **No prompt builder protocols** - Prompt building is an implementation detail, not formalized.

3. **LLM ownership:**
   - WorldModel receives LLM at construction (implementation detail for how it processes context)
   - Executor owns code generation LLM calls (separate concern, may use different LLM/config)
   - Context provides all data; LLM is just how WorldModel internally processes it

4. **Typed contexts as dataclasses** - Replace `dict[str, Any]` and TypedDict with dataclasses for cleaner attribute access and better IDE support.

### Protocol Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           V2 PROTOCOL ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────┐
                              │      Executor       │
                              │  (orchestrates all) │
                              │                     │
                              │  • owns LLM calls   │
                              │  • runs lifecycle   │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    │                    ▼
         ┌─────────────────────┐         │         ┌─────────────────────┐
         │     WorldModel      │         │         │      Evaluator      │
         │     (Protocol)      │         │         │     (Protocol)      │
         ├─────────────────────┤         │         ├─────────────────────┤
         │ propose(ctx) → [Node]         │         │ evaluate(impl, ctx) │
         │ select(ctx) → [Node]│         │         │ → EvaluationResult  │
         │ update(ctx) → None  │         │         │                     │
         └─────────────────────┘         │         └─────────────────────┘
                                         │
                                         │
                              ┌──────────┴──────────┐
                              │  Code Gen Prompts   │
                              │  (impl detail)      │
                              │                     │
                              │  Executor builds    │
                              │  prompts, calls LLM │
                              └─────────────────────┘
```

**Protocols (formalized):**
- `WorldModel` - propose/select/update with typed contexts
- `Evaluator` - evaluate implementations

**Implementation details (not formalized):**
- Prompt building (whatever classes/functions work)
- LLM invocation (executor owns this)


### Typed Contexts (Dataclasses)

**Current state:** `run.py` uses TypedDict for contexts.

**Decision:** Migrate to dataclasses for:
- Cleaner attribute access (`ctx.round_idx` vs `ctx["round_idx"]`)
- Better IDE support (autocomplete, refactoring)
- Consistency with `V1Node`/`V1Action` which are already dataclasses
- Can add computed properties/methods if needed

**Contexts to define:**

```python
from dataclasses import dataclass

@dataclass
class ProposeContext:
    """Context for WorldModel.propose() - generate new action nodes."""
    tree: Tree
    task: TaskDefinition  # needed for definition_text in prompts
    round_idx: int
    current_code: str = ""  # best code so far, empty string on first call


@dataclass
class SelectContext:
    """Context for WorldModel.select() - pick next action.

    Note: V1 selects internally using WorldModelManager state.
    This context just provides tree for implementations that
    need external selection logic.
    """
    tree: Tree


@dataclass
class UpdateContext:
    """Context for WorldModel.update() - update tree after cycle.

    Fields like round_idx, attempts, logs can be derived from cycle:
    - round_idx: cycle start index (tracked by executor)
    - attempts: len(cycle.rounds)
    - logs: cycle.rounds[-1].result.get_log() if rounds exist
    - code: cycle.best_round.llm_response if best_round exists
    """
    tree: Tree
    node: Node
    cycle: Cycle  # completed cycle
    succeeded: bool
    round_idx: int  # global round index at cycle start


@dataclass
class CodeGenContext:
    """Context for code generation (executor use).

    Note: target_gpu and language are constructor params on the
    prompt building component, not per-call context.
    """
    task: TaskDefinition
    node: Node
    cycle: Cycle  # in-progress, may have 0 rounds
```

### Protocol Definitions

**Note:** `WorldModel` protocol exists at `k_search/modular/protocols/world_model.py`.
It currently takes `tree: Tree` as separate param + `context: dict[str, Any] | None`.
We update it to use `context: Any` (implementation-defined).

**Why `Any` for context?** We don't know what context future implementations will need,
and LSP prevents narrowing parameter types in implementations. The protocol defines the
method shape; implementations define their context requirements via documentation and
typed dataclasses.

**What protocols still enforce:**
- Method names and signatures
- Return types (`list[Node]`, `str`, etc.)
- Structural typing (any class with these methods satisfies the protocol)

```python
from typing import Any, Protocol


class WorldModel(Protocol):
    """World model for search tree operations.

    Context is implementation-defined. See specific implementations
    for their context requirements (e.g., V1WorldModel uses ProposeContext).
    """

    def propose(self, context: Any) -> list[Node]:
        """Generate new frontier nodes with actions."""
        ...

    def select(self, context: Any) -> list[Node]:
        """Select frontier nodes to pursue."""
        ...

    def update(self, context: Any) -> None:
        """Update tree after cycle completes."""
        ...


class Evaluator(Protocol):
    """Evaluate generated code (existing protocol)."""

    def evaluate(self, impl: Implementation, context: Any) -> EvaluationResult:
        """Evaluate implementation."""
        ...
```

### Implementation Notes

Prompt building is an implementation detail. Example patterns (not formalized):

```python
# WorldModel receives LLM at construction, uses internally
class V1WorldModel:
    """WorldModel implementation wrapping V1's WorldModelManager."""

    def __init__(self, manager: WorldModelManager, llm: LLMCall, task: TaskDefinition):
        self._manager = manager
        self._llm = llm
        self._task = task

    def propose(self, context: ProposeContext) -> list[Node]:
        # Build prompt internally (uses build_decision_tree_edit_prompt)
        # Call self._llm internally
        # Parse response, create nodes
        ...


# Executor handles code generation with its own LLM
class Executor:
    def __init__(self, world_model: WorldModel, code_gen_llm: LLMCall, ...):
        self._world_model = world_model
        self._code_gen_llm = code_gen_llm

    def _run_cycle(self, node: Node, ...) -> Cycle:
        # Build code gen prompt (from_spec, debug, improve based on state)
        # Call self._code_gen_llm
        # Evaluate result
        ...
```

### Data Derivation from Existing Abstractions

All prompt inputs can be derived from `CodeGenContext`:

| Raw Input | Derivation from Context |
|-----------|------------------------|
| `definition_text` | `context.task.get_prompt_text()` |
| `action_text` | `context.node.action.title` |
| `base_code` | `context.node.parent.cycle.best_round.llm_response` |
| `current_code` | `context.cycle.rounds[-1].llm_response` (if rounds exist) |
| `trace_logs` | `context.cycle.rounds[-1].result.get_log()` |
| `attempt` | `len(context.cycle.rounds)` |
| `has_passed` | `context.cycle.best_round is not None` |
| `perf_summary` | `context.cycle.rounds[-1].result.get_metrics()` |

**UpdateContext derivation (for prompts #3a/#3b):**

| Raw Input | Derivation from UpdateContext |
|-----------|------------------------------|
| `world_model_json` | Internal to WorldModelManager |
| `definition_text` | WorldModelManager constructor param |
| `chosen_action_text` | `context.node.action.title` |
| `current_tree_path` | `WorldModelManager.get_tree_path_text()` |
| `eval_result` | `context.cycle.best_round.result` if succeeded |
| `current_code` | `context.cycle.best_round.llm_response` if succeeded |
| `debug_round` | `len(context.cycle.rounds)` |

Note: `world_model_json`, `definition_text`, and `current_tree_path` are internal to `WorldModelManager`. The `UpdateContext` provides `tree`, `node`, `cycle`, `succeeded`, and `round_idx` - the WorldModel implementation extracts what it needs for its internal prompts.

---

## Migration Path

### Phase 1: Update Protocols
- [ ] Update `WorldModel` protocol (`k_search/modular/protocols/world_model.py`):
  - Remove `tree: Tree` as separate parameter (fold into context)
  - Change `context: dict[str, Any] | None` to `context: Any`
  - This is a breaking change - update existing implementations:
    - `scripts/gpu_mode_modular_k_search/` - uses `V1WorldModel`
    - `scripts/gpu_mode_async_pipeline_executor/` - uses `AsyncSimpleWorldModel`
    - `scripts/gpu_mode_simple_linear_executor/` - uses `SimpleWorldModel`
  - Note: `scripts/gpu_mode_modular_v1/` does NOT use WorldModel protocol (simple loop)
- [ ] Async handling: The `WorldModel` protocol stays sync. `AsyncSimpleWorldModel` is an async implementation that satisfies the protocol when awaited externally. No separate `AsyncWorldModel` protocol needed - the caller (executor) decides sync/async invocation.

### Phase 2: Create New v2 Directory
- [ ] Create `scripts/gpu_mode_modular_k_search_v2/`
- [ ] Add context dataclasses (`ProposeContext`, `UpdateContext`, `CodeGenContext`, etc.)
- [ ] Implement v2 executor using updated protocols
- [ ] Implement v2 WorldModel wrapper
- [ ] Implement code generation prompt building (internal to executor)

---

## Open Questions

None - all questions resolved. See Resolved Decisions section.

**Resolved:** `TaskDefinition.get_prompt_text()` provides the spec text needed for prompts.

## Resolved Decisions

1. **Keep `WorldModel` as single protocol** - Don't split into ActionProposer/ActionSelector/TreeUpdater (YAGNI).

2. **Protocol context typing** - Use `Any` for context parameters. Protocols define method shape, implementations define context requirements. This is LSP-compliant and maximally flexible.

3. **Tree in context** - Fold `tree: Tree` into context rather than separate parameter.

4. **No prompt builder protocols** - Prompt building is implementation detail, not formalized.

5. **LLM ownership split:**
   - WorldModel receives LLM at construction, uses internally for propose/update
   - Executor owns code generation LLM calls (separate concern, different LLM/config possible)

6. **Dataclasses over TypedDict** - Cleaner attribute access, better IDE support, consistency with V1Node/V1Action.
