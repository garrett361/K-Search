# Paper-to-Code Mapping

Reference for the K-Search paper (arXiv:2602.19128) notation and its implementation.

## Notation Table

| Paper | Code | Notes |
|-------|------|-------|
| π_code | `_generate_code_from_prompt()` via `world_model_prompts.py` | Same LLM instance as P_world |
| P_world | `WorldModelManager.{init,refine,propose_action_nodes}()` | Receives full tree state |
| S_t (search state) | World model JSON with `decision_tree.nodes` | Persisted across rounds |
| Open nodes (frontier) | Nodes with `action.title`, no `solution_ref.solution_id` | Executable candidates |
| Closed nodes | Nodes with attached `solution_ref.solution_id` | Have evaluated solutions |
| x_parent | `base_raw_code` from immediate parent's solution | **Not** best/root/lineage |
| δ (optimization intent) | `chosen_action_text` from `render_chosen_action_node_block()` | Title + description |
| Action a_t = (x_parent, δ) | `(base_code, action_text)` passed to prompt | Parent code + intent |
| Priority V | `action.score_0_to_1` | Range [0,1], used in selection |
| Insert/Update/Prune | `DecisionTreeEditOps` | `insert_node`, `update_node`, `delete_node` |

## Are π_code and P_world the Same LLM?

**Yes.** Both use a single LLM client initialized once:

```
kernel_generator_world_model.py:124-135

def _llm_call(prompt: str) -> str:
    if self.use_reasoning_api:
        response = self.client.responses.create(...)
        return (response.output_text or "").strip()
    response = self.client.chat.completions.create(
        model=self.model_name, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()
```

This `_llm_call` is passed to `WorldModelManager` and used for all operations.
The π_code / P_world distinction is **conceptual**, not architectural.

## What Does π_code Receive?

### x_parent = Immediate Parent Only

Code generation receives the **direct parent's code**, not the best solution, root, or lineage.

Retrieval flow (`kernel_generator_world_model.py:513-580`):

```
1. chosen_action_node has parent_id field
2. Look up parent node's solution_ref.solution_id
3. Fetch that solution's code from SolutionDB

parent_id = str((node_obj or {}).get("parent_id") or "root")
sr = self._wm.get_solution_ref_for_node(definition_name=task.name, node_id=parent_id)
sid = sr.get("solution_id")
recb = self._solution_db.get(sid)
base_raw_code = recb.code
```

**No lineage access.** Grandparent code is invisible to π_code.

### Fallback When Parent Has No Solution

If `parent_id == "root"` or parent has no attached solution:
- Use `"(no base code; start from spec)"` placeholder
- Prompt constructed from spec + action only

### Debug/Improve Loop (Attempts 2+)

Shows whichever is better by score (`kernel_generator_world_model.py:661-744`):

```python
base_for_debug = base_raw_code  # parent's code
if cycle_best_score > base_score:
    base_for_debug = cycle_best_raw  # current cycle's best attempt
```

Single reference snippet shown, not multiple.

### Full Context π_code Does NOT Receive

- Full decision tree structure
- Sibling actions or their results
- Grandparent/ancestor code
- Open frontier list

## What Does P_world Receive?

**Full tree state.** World model operations receive the complete `world_model_json`:

| Operation | Call Site | What It Receives |
|-----------|-----------|------------------|
| Init | `world_model_manager.py:185` | Full spec, creates initial tree |
| Refine | `world_model_manager.py:735` | Full tree + eval result |
| Propose | `world_model_manager.py:930` | Full tree |

The `build_decision_tree_edit_prompt()` includes:
- `world_model_json` - Complete tree with all nodes
- `wm_status_text` - Rendered tree status
- `open_frontier_nodes_text` - Available actions
- `eval_result` - Latest evaluation

Contrast: π_code sees only (parent_code, action_text), while P_world sees the entire search state.

## Algorithm 1 → Code Mapping

Paper's Algorithm 1 maps to `generate()` in `kernel_generator_world_model.py`:

```
1. Initialize S_0 with spec          → WorldModelManager.init()
2. While not converged:
   a. Select action a_t              → select_open_action_node() using score_0_to_1
   b. Get x_parent from parent node  → get_solution_ref_for_node(parent_id)
   c. Generate code x_t = π(x_parent, δ) → _generate_code_from_prompt()
   d. Evaluate x_t                   → task.evaluate(solution)
   e. Update S_{t+1}                 → WorldModelManager.refine()
   f. Propose new actions            → WorldModelManager.propose_action_nodes()
3. Return best solution              → SolutionDB.get(best_solution_id)
```

## Key Architectural Decision: Context Isolation

π_code (code generation) deliberately receives less context than P_world (world model):

| Aspect | π_code | P_world |
|--------|--------|---------|
| Tree structure | No | Yes |
| Sibling actions | No | Yes |
| Historical scores | No | Yes |
| Lineage code | No | Yes |
| Parent code | Yes | N/A |
| Action intent | Yes | Yes |

**Intent:** Focus code generation on a single transformation (parent → child).
The world model tracks global search state; code policy focuses on local edits.

## Source File Reference

- `kernel_generator_world_model.py:124-135` - Single LLM client
- `kernel_generator_world_model.py:513-580` - x_parent retrieval
- `kernel_generator_world_model.py:661-744` - Debug loop base selection
- `world_model_manager.py:185` - WM init
- `world_model_manager.py:735` - WM refine
- `world_model_manager.py:930` - WM propose
- `world_model_manager.py:1881-1901` - `get_solution_ref_for_node()`
- `world_model_prompts.py:16-37` - Action prompt template
- `world_model_prompts.py:64-98` - Debug prompt template
