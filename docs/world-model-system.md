# World Model System

The world model system maintains a structured JSON representation of kernel optimization strategies across generation rounds. It enables the LLM to build on prior iterations rather than starting fresh each time.

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │      WorldModelKernelGenerator      │
                    │  (kernel_generator_world_model.py)  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       WorldModelManager             │
                    │    (world_model_manager.py)         │
                    │                                     │
                    │  - ensure_initialized()             │
                    │  - refine()                         │
                    │  - choose_next_action_node_id()     │
                    │  - attach_solution_to_active_leaf() │
                    │  - set_active_leaf_id()             │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         world_model.py              │
                    │                                     │
                    │  - JSON schema/parsing              │
                    │  - Rendering for prompts            │
                    │  - Edit ops parsing                 │
                    └─────────────────────────────────────┘
```

## Decision Tree Structure

The world model organizes optimization strategies as a prefix tree where each root-to-leaf path represents a complete plan.

```
                        ┌─────────────────┐
                        │      root       │  decision=null, choice=null
                        │   (baseline)    │  solution_ref → baseline eval
                        └────────┬────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
   ┌────────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
   │  tiled_matmul   │  │  fused_online   │  │  vectorized    │
   │  (SOLVED)       │  │  (OPEN)         │  │  (OPEN)        │
   └────────┬────────┘  └─────────────────┘  └────────────────┘
            │
   ┌────────▼────────┐
   │  shared_mem_opt │
   │  (OPEN)         │  ← continuation node
   └─────────────────┘
```

### Node Schema

Each node in `decision_tree.nodes` contains:

| Field | Purpose |
|-------|---------|
| `node_id` | Unique identifier |
| `parent_id` | Parent node (null for root) |
| `node_type` | `"root"` or `"action"` |
| `decision` | Question at this branch point |
| `choice` | Selected answer (branch taken) |
| `action` | Optimization action to attempt (see below) |
| `solution_ref` | Attached solution and eval result |
| `overall_rating_0_to_10` | LLM-assigned quality estimate |
| `confidence_0_to_1` | LLM confidence in this approach |
| `notes` | Analysis or rationale |

### Action Field

Nodes with `action.title` are executable optimization attempts:

```json
{
  "action": {
    "title": "Use shared memory for A tiles",
    "description": "Stage A matrix tiles in shared memory...",
    "difficulty_1_to_5": 3,
    "score_0_to_1": 0.7,
    "expected_vs_baseline_factor": 1.2,
    "rationale": "Reduces global memory traffic..."
  }
}
```

### Solution Reference

When a solution is evaluated and attached:

```json
{
  "solution_ref": {
    "solution_id": "abc123",
    "solution_name": "tiled_matmul_v2",
    "parent_solution_id": "def456",
    "round_index": 5,
    "eval": {
      "status": "passed",
      "metrics": { "score": 0.85, "score_name": "vs_baseline" }
    }
  }
}
```

## Node States

| State | Condition |
|-------|-----------|
| OPEN | Has `action.title`, no `solution_ref.solution_id` |
| SOLVED | Has attached `solution_ref.solution_id` |
| FRONTIER | OPEN and parent is SOLVED (or root) |

Only FRONTIER nodes are eligible for selection.

## Selection Policy

`WorldModelSelectionPolicy` controls which action to execute next.

### Difficulty Gating

```python
max_difficulty_1_to_5: int = 4          # Prefer actions at or below this
relax_difficulty_if_best_vs_base_ge: float = 0.5  # Threshold to relax
relaxed_max_difficulty_1_to_5: int = 4  # Relaxed max when threshold met
```

Actions with `difficulty > max_difficulty_1_to_5` are filtered out unless the best observed `vs_baseline` score exceeds the relaxation threshold.

### Selection Ranking

FRONTIER nodes are sorted by:

1. `action.score_0_to_1` (descending) - LLM's expected value
2. `difficulty_1_to_5` (ascending) - prefer easier actions
3. `overall_rating_0_to_10` (descending) - node quality
4. `node_id` (ascending) - stable tie-breaker

The top-ranked node becomes the next action.

## WorldModelManager

### Key Methods

| Method | Purpose |
|--------|---------|
| `ensure_initialized()` | Create initial WM via LLM if none exists |
| `refine()` | Update WM after successful eval (edit ops) |
| `choose_next_action_node_id()` | Deterministic frontier selection |
| `attach_solution_to_active_leaf()` | Record eval result on active node |
| `set_active_leaf_id()` | Advance pointer after selection |
| `get()` / `set()` | Read/write WM JSON by task name |

### Configuration

```python
WorldModelConfig(
    enabled=True,
    max_chars_per_block=6000,       # Prompt truncation limit
    max_new_nodes_per_edit=3,       # Cap new nodes per refine
    selection_policy=WorldModelSelectionPolicy(...)
)
```

## Round-by-Round Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                         Round N                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INITIALIZE (if first round)                                 │
│     └─ ensure_initialized() → LLM creates initial WM JSON       │
│                                                                 │
│  2. SELECT ACTION                                               │
│     └─ choose_next_action_node_id() → pick from FRONTIER        │
│     └─ set_active_leaf_id() → advance pointer to selected node  │
│                                                                 │
│  3. GENERATE CODE                                               │
│     └─ Inject chosen action into codegen prompt                 │
│     └─ LLM produces kernel implementation                       │
│                                                                 │
│  4. EVALUATE                                                    │
│     └─ Run kernel, measure latency, check correctness           │
│                                                                 │
│  5. ATTACH RESULT                                               │
│     └─ attach_solution_to_active_leaf() → record eval on node   │
│                                                                 │
│  6. REFINE (only on PASSED with performance data)               │
│     └─ refine() → LLM emits edit ops to update tree             │
│     └─ Continuation rule: ensure ≥1 OPEN child (difficulty < 5) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Continuation Rule

After attaching a PASSED solution, `refine()` enforces that the solved node has at least one OPEN child action with `difficulty < 5`. This prevents the search from stalling:

```
BEFORE refine():                 AFTER refine():

  ┌─────────┐                      ┌─────────┐
  │ node_A  │ ← just solved        │ node_A  │ (SOLVED)
  │ (OPEN)  │                      └────┬────┘
  └─────────┘                           │
                                   ┌────▼────┐
                                   │ node_B  │ (OPEN)
                                   │ diff=3  │ ← continuation
                                   └─────────┘
```

Without continuation, selection would jump to unrelated branches.

## Edit Operations

The LLM refines the decision tree via structured edit ops:

| Op | Purpose |
|----|---------|
| `update_node` | Modify existing node fields |
| `insert_node` | Add new child action |
| `split_node` | Branch a node into alternatives |
| `delete_node` | Remove unused branches |

Validation ensures:
- Root `decision`/`choice` remain null
- Existing attached solutions are not dropped
- Continuation rule is satisfied

## Prompt Rendering

`render_world_model_section()` produces a compact view for prompts:
- `kernel_summary` and `open_questions`
- Active path from root to current leaf
- Top-rated sibling alternatives
- `computed_signals` (last trace metrics)

`render_chosen_action_node_block()` formats the selected action for codegen:

```
World Model: Chosen Next Action (from decision tree node)
- node_id: opt_shared_mem
- base_node_id: tiled_matmul
- title: Use shared memory for A tiles
- difficulty_1_to_5: 3
- description: Stage A matrix tiles in shared memory...
- expected_vs_baseline_factor: 1.20x
```

## File Locations

| File | Contents |
|------|----------|
| `world_model.py` | Schema, parsing, rendering, edit ops |
| `world_model_manager.py` | Manager class, selection policy, lifecycle |
| `kernel_generator_world_model.py` | Integration with kernel generator |
