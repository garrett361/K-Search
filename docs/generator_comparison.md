# KernelGenerator vs WorldModelKernelGenerator: Key Differences

## Architecture Overview

```
KernelGenerator (Base)
├─ Simple linear optimization loop
├─ No persistent state between rounds
└─ Prompts contain: spec + current code + execution logs

WorldModelKernelGeneratorWithBaseline (extends KernelGenerator)
├─ Action-cycle based optimization loop
├─ Persistent world model (decision tree) maintained across all rounds
├─ Solution database tracking all attempts
└─ Prompts contain: spec + current code + logs + WORLD MODEL JSON + SELECTED ACTION
```

## Loop Structure Comparison

### KernelGenerator (kernel_generator.py:261-567)

**Structure:** Simple sequential rounds

```python
for round_num in range(1, max_opt_rounds + 1):
    # 1. Evaluate current solution
    eval_result = task.run_benchmark(solution)

    # 2. Update best if improved
    if passed and score > best_score:
        best_solution = solution
        best_score = score

    # 3. Generate next iteration
    opt_prompt = get_optimization_prompt(
        definition_text=definition_text,
        trace_logs=trace_logs,
        current_code=current_raw_code,
        current_best=current_best_for_prompt,
        previous_round_summary=previous_round_summary
    )

    # 4. Call LLM with simple prompt
    code_result = self._generate_code_from_prompt(opt_prompt)
    current_code = code_result["cleaned"]
```

**Key characteristics:**
- Rounds share best-so-far state and previous round summary
- No structured exploration strategy (LLM decides from recent feedback)
- No decision tree or solution genealogy
- No systematic memory of the broader exploration landscape

---

### WorldModelKernelGenerator (kernel_generator_world_model.py:306-900+)

**Structure:** Nested action-cycles

```python
cycle_start_round = 1
while cycle_start_round <= max_opt_rounds:

    # === CYCLE START: Choose an action from the decision tree ===

    # 1. Propose new action nodes (expand decision tree)
    self._wm.propose_action_nodes(
        definition_name=task.name,
        definition_text=definition_text,
        current_code_excerpt=current_code,
        current_tree_path=self._wm.get_tree_path_text(),
        round_index=cycle_start_round
    )

    # 2. Select highest-utility action node from frontier
    chosen_leaf = self._wm.choose_next_action_node_id(
        definition_name=task.name
    )
    # Utility = 3.0*score - 2.5*difficulty + 0.75*depth + ...

    # 3. Get parent solution code as base (if exists)
    parent_solution = self._solution_db.get(parent_id)
    base_raw_code = parent_solution.code if parent_solution else ""

    # === INNER LOOP: Multiple attempts at same action ===

    no_improve_streak = 0
    rounds_consumed = 0

    while True:
        attempt_idx = rounds_consumed + 1
        round_num = cycle_start_round + rounds_consumed

        # Attempt 1: Generate from parent + action
        if attempt_idx == 1:
            if base_raw_code:
                prompt = get_generate_code_from_action_prompt(
                    definition_text=definition_text,
                    base_code=base_raw_code,
                    action_text=chosen_action_text,
                    target_gpu=self.target_gpu
                )
            else:
                prompt = get_generate_code_from_spec_with_action_prompt(
                    definition_text=definition_text,
                    action_text=chosen_action_text,
                    target_gpu=self.target_gpu
                )

        # Attempts 2+: Debug and improve
        else:
            prompt = get_debug_and_improve_from_spec_prompt(
                definition_text=definition_text,
                trace_logs=trace_logs,
                current_code=current_raw_code,
                action_text=chosen_action_text,
                debug_round=attempt_idx,
                max_rounds=max_dai,
                base_code=base_for_debug
            )

        # CRITICAL: Inject world model JSON into prompt
        prompt = prompt + "\n\n" + render_world_model_section(
            self._wm.get(task.name),
            max_chars=self._world_model_max_chars
        )

        # Generate and evaluate
        code_result = self._generate_code_from_prompt(prompt)
        solution = self._create_solution_from_code(...)
        eval_result = task.run_benchmark(solution)

        # Track in solution database
        self._solution_db.add(
            solution=solution,
            eval_result=eval_result,
            parent_solution_id=parent_solution_id
        )

        # Check stagnation
        if score_improved:
            cycle_best_solution = solution
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        # Exit inner loop if stagnated
        if no_improve_streak >= stagnation_window:
            break

        rounds_consumed += 1
        if cycle_start_round + rounds_consumed > max_opt_rounds:
            break

    # === CYCLE END: Update world model ===

    if cycle_best_solution:
        # Success: attach solution to action node and refine tree
        self._wm.attach_and_refine(
            definition_name=task.name,
            node_id=chosen_leaf,
            solution=cycle_best_solution,
            eval_result=cycle_best_eval,
            code_excerpt=cycle_best_wm_code,
            prediction=prediction
        )
    else:
        # Failure: mark action as "too hard"
        self._wm.note_action_too_hard(
            definition_name=task.name,
            node_id=chosen_leaf,
            feedback="Multiple attempts failed to pass correctness"
        )

    # Persist world model snapshot for resumption
    self._persist_world_model_snapshot(task=task)

    cycle_start_round += rounds_consumed
```

---

## Key Differences Summary

| Aspect | KernelGenerator | WorldModelKernelGenerator |
|--------|-----------------|---------------------------|
| **Loop structure** | Flat: N sequential rounds | Nested: Cycles of action attempts |
| **State between rounds** | Best solution + previous round summary | Persistent decision tree + solution DB |
| **Prompt content** | Spec + code + logs + best-so-far | Spec + code + logs + **WM JSON** + **action** |
| **Optimization strategy** | LLM decides from recent feedback | Tree-guided exploration with utility function |
| **Code ancestry** | No tracking | Full solution genealogy in database |
| **Exploration memory** | Only best-so-far + last round | Full decision tree of all attempts |
| **Stagnation handling** | None (always runs max rounds) | Early cycle termination on plateau |
| **Action selection** | Implicit (LLM decides) | Explicit (choose frontier node by utility) |
| **Tree expansion** | N/A | `propose_action_nodes()` adds candidates |
| **Resumability** | Limited (can continue from solution) | Full (WM + solution DB + genealogy) |
| **Prompting phases** | Generation → Optimization | Generation from spec/action → Debug → Improve |

---

## What the World Model Contains

The world model JSON injected into every prompt includes:

```json
{
  "kernel_summary": "High-level problem description",
  "decision_tree": {
    "root": "root_node_id",
    "nodes": [
      {
        "node_id": "action_fuse_layernorm_fp16",
        "parent_id": "root",
        "action": {
          "title": "Fuse LayerNorm + projection to fp16",
          "difficulty_1_to_5": 3,
          "expected_vs_baseline_factor": 1.15,
          "rationale": "Reduce memory bandwidth..."
        },
        "solution_ref": {
          "solution_id": "sol_r42",
          "eval": {
            "status": "PASSED",
            "latency_ms": 1.05,
            "score": 0.952
          }
        },
        "impacts": {
          "memory_bandwidth": {
            "rating": 8,
            "risk": "Lower precision may affect accuracy"
          }
        },
        "children": ["action_optimize_tiling"]
      }
    ]
  },
  "open_questions": [
    "Can we use tensor cores for the pairwise matmul?",
    "Is there a better memory layout?"
  ],
  "computed_signals": {
    "round_index": 42,
    "best_score": 0.952,
    "trace": {...}
  }
}
```

This gives the LLM:
- **Context**: What strategies have been tried
- **Guidance**: Which action to attempt now
- **History**: What worked/failed and why
- **Uncertainty**: Open questions to explore

---

## Why World Model is Better

**KernelGenerator issues:**
1. Only remembers best-so-far + last round (no long-term exploration memory)
2. May repeat failed strategies from earlier rounds
3. No systematic exploration (LLM decides ad-hoc from recent feedback)
4. Limited resumability and no solution genealogy for analysis

**WorldModelKernelGenerator advantages:**
1. Structured exploration prevents redundant attempts
2. Utility function prioritizes promising actions
3. Decision tree provides transparency
4. Full resumption from snapshots
5. LLM sees "global context" of optimization landscape
6. Solution genealogy enables analysis

---

## Example Flow

**TriMul with KernelGenerator (hypothetical):**
```
Round 1: Generate kernel from spec
Round 2: LLM sees "failed", tries to fix
Round 3: LLM sees "still failing", tries different approach
Round 4: LLM forgets round 2 strategy, might retry it
...
Round 300: Best of 300 independent attempts
```

**TriMul with WorldModelKernelGenerator (actual):**
```
Cycle 1: root → "Fuse LayerNorm+fp16" (utility: 8.2)
  ├─ Attempt 1: Generate from spec + action
  ├─ Attempt 2-5: Debug failing tests
  └─ SUCCESS at attempt 3 → attach to tree, refine WM

Cycle 2: "Fuse LayerNorm+fp16" → "Optimize pairwise matmul layout" (utility: 7.9)
  ├─ Attempt 1: Generate from parent code + action
  ├─ Attempts 2-4: Improve performance
  └─ SUCCESS at attempt 2 → attach to tree

Cycle 3: "Optimize layout" → "Use tensor cores" (utility: 7.5)
  ├─ Too difficult, fails 5 attempts
  └─ Mark "too hard", return to frontier

Cycle 4: "Optimize layout" → "Aggressive register tiling" (utility: 7.1)
  └─ SUCCESS → new SOTA

...continues building the tree systematically...
```

The world model enables **intelligent, structured exploration** vs. **random search**.

---

## Utility Function Details

The action selection utility function (from `world_model_manager.py`):

```python
class WorldModelSelectionPolicy:
    max_difficulty_1_to_5 = 4  # Skip very hard actions initially
    relax_difficulty_if_best_vs_base_ge = 0.5  # Allow harder if >50% speedup

    weight_score = 3.0           # Prioritize high-scoring branches
    weight_difficulty = 2.5      # Penalize difficult actions
    weight_depth = 0.75          # Slight bias toward deeper exploration
    weight_parent_quality = 1.5  # Prefer children of successful nodes
```

**Utility calculation:**
```python
utility = (
    weight_score * score +
    weight_difficulty * (-difficulty) +
    weight_depth * depth +
    weight_parent_quality * parent_quality
)
```

This balances exploration (depth) vs exploitation (score) while managing risk (difficulty).

---

## File References

- **Base generator**: `k_search/kernel_generators/kernel_generator.py`
- **World model generator**: `k_search/kernel_generators/kernel_generator_world_model.py`
- **World model data structures**: `k_search/kernel_generators/world_model.py`
- **World model manager**: `k_search/kernel_generators/world_model_manager.py`
- **Prompt templates**: `k_search/kernel_generators/*_prompts.py`
- **Solution tracking**: `k_search/utils/solution_db.py`
