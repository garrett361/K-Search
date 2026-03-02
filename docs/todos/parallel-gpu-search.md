# TODO: Parallelize World Model Search Across Multiple GPUs

## Problem

The current WorldModelKernelGenerator performs completely serial search:

```python
# kernel_generator_world_model.py:414-710
while cycle_start_round <= max_opt_rounds:
    chosen_leaf = choose_next_action_node_id()  # Pick ONE action

    while True:  # Try this action multiple times
        code = generate_code_from_prompt()      # BLOCKING: LLM call
        eval = task.run_benchmark(solution)     # BLOCKING: GPU evaluation

        if no_improvement:
            break

    update_world_model()
    cycle_start_round += rounds_consumed
```

**Bottleneck:** With N GPUs available, only 1 GPU is active at a time. Search throughput scales with rounds, not hardware.

## Current Architecture Constraints

From kernel_generator_world_model.py:306-710:

1. **Single action selection** - `choose_next_action_node_id()` picks exactly one frontier node
2. **Blocking evaluation** - each `task.run_benchmark()` must complete before next iteration
3. **Sequential world model updates** - `attach_and_refine()` requires eval results before proceeding
4. **No pipelining** - LLM code generation and GPU evaluation cannot overlap

## Parallelization Opportunities

### 1. Parallel Frontier Exploration (Recommended)

**Concept:** Pick top-K frontier actions and evaluate them simultaneously on K GPUs.

```python
while cycle_start_round <= max_opt_rounds:
    # Select K highest-utility actions from frontier
    frontier_actions = choose_top_K_actions(
        K=num_gpus,
        diversity_bonus=0.2  # Penalize similarity to encourage exploration
    )

    # Explore each action in parallel
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(explore_action_cycle, action, gpu_id)
            for gpu_id, action in enumerate(frontier_actions)
        ]
        results = [f.result() for f in futures]

    # Batch update world model with all results
    for action, eval_result, solution in results:
        if eval_result.is_passed():
            attach_and_refine(action, solution, eval_result)
        else:
            note_action_too_hard(action, eval_result)

    cycle_start_round += max(r.rounds_consumed for r in results)
```

**Expected speedup:** Near-linear with K GPUs (e.g., 8x with 8 GPUs)

**Pros:**
- Explores multiple branches simultaneously
- Better coverage of optimization landscape
- Utilizes all available GPUs
- Minimal changes to world model logic (batch updates instead of incremental)

**Cons:**
- World model merging becomes more complex (need conflict resolution if actions overlap)
- Utility function needs diversity mechanism to avoid picking similar actions
- Round accounting gets tricky (do parallel attempts count as 1 round or K rounds?)

### 2. Parallel Attempts Within Action Cycle

**Concept:** Generate K code variants per action and evaluate simultaneously.

```python
for cycle in cycles:
    action = choose_next_action_node_id()

    # Generate K variants in parallel (different temperatures/seeds)
    codes = parallel_generate_codes(action, K=num_gpus)

    # Evaluate all variants simultaneously
    evals = [task.run_benchmark(c, gpu=i) for i, c in enumerate(codes)]

    best = max(zip(codes, evals), key=lambda x: x[1].score)
    update_world_model(action, best)
```

**Expected speedup:** 2-4x (diminishing returns due to code similarity)

**Pros:**
- Simpler to implement (no world model merge logic)
- Faster iteration within each action

**Cons:**
- LLM may generate similar code K times without history
- Wasteful if variants are redundant
- Doesn't increase breadth of exploration

### 3. Pipelined LLM Generation + GPU Evaluation

**Concept:** Overlap LLM code generation with GPU benchmark execution.

```python
# Queue-based pipeline
generation_queue = Queue()
evaluation_queue = Queue()

# Producer: LLM code generation
def generate_worker():
    while True:
        prompt = generation_queue.get()
        code = generate_code_from_prompt(prompt)
        evaluation_queue.put(code)

# Consumer: GPU evaluation
def evaluate_worker(gpu_id):
    while True:
        code = evaluation_queue.get()
        eval = task.run_benchmark(code, gpu=gpu_id)
        results_queue.put(eval)
```

**Expected speedup:** 1.5-2x (depends on LLM latency vs GPU latency)

**Pros:**
- Hides latency
- Better resource utilization

**Cons:**
- Complex queue management
- Ordering issues (evals arrive out-of-order)
- Doesn't fundamentally increase parallelism

## Recommended Implementation: Hybrid Approach

Combine parallel frontier exploration (#1) with limited attempts per action:

```python
def parallel_world_model_search(
    task,
    max_opt_rounds,
    num_gpus=8,
    attempts_per_action=3
):
    cycle_start_round = 1

    while cycle_start_round <= max_opt_rounds:
        # Expand decision tree with new action proposals
        propose_action_nodes(...)

        # Select top-K diverse actions
        frontier = choose_top_K_actions(
            K=num_gpus,
            diversity_bonus=0.2,
            min_utility_gap=0.1  # Ensure actions are sufficiently different
        )

        # Parallel exploration across GPUs
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(
                    explore_action_on_gpu,
                    action=action,
                    gpu_id=gpu_id,
                    max_attempts=attempts_per_action,
                    parent_solution=get_parent_solution(action),
                    world_model_snapshot=serialize_wm_state()
                ): action
                for gpu_id, action in enumerate(frontier)
            }

            results = []
            for future in as_completed(futures):
                action = futures[future]
                result = future.result()
                results.append((action, result))

        # Merge results into world model
        for action, result in sorted(results, key=lambda x: x[1].best_score, reverse=True):
            if result.success:
                attach_and_refine(
                    node_id=action.node_id,
                    solution=result.best_solution,
                    eval_result=result.best_eval
                )
            else:
                note_action_too_hard(
                    node_id=action.node_id,
                    feedback=result.failure_reason
                )

        rounds_consumed = sum(r.rounds_consumed for _, r in results)
        cycle_start_round += rounds_consumed

def explore_action_on_gpu(
    action,
    gpu_id,
    max_attempts,
    parent_solution,
    world_model_snapshot
):
    """
    Isolated worker function for parallel execution.
    Each worker gets a copy of WM state but doesn't modify the shared WM.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    best_solution = None
    best_eval = None
    best_score = -1.0

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            code = generate_from_action(parent_solution, action)
        else:
            code = debug_and_improve(current_code, trace_logs)

        solution = create_solution(code)
        eval_result = benchmark(solution)  # GPU-local evaluation

        if eval_result.score > best_score:
            best_solution = solution
            best_eval = eval_result
            best_score = eval_result.score

        if not eval_result.is_passed():
            continue
        else:
            break  # Success, exit early

    return ExplorationResult(
        success=best_eval.is_passed() if best_eval else False,
        best_solution=best_solution,
        best_eval=best_eval,
        best_score=best_score,
        rounds_consumed=attempt,
        failure_reason=None if best_eval else "all attempts failed"
    )
```

## Implementation Challenges

### 1. World Model Consistency

**Problem:** Parallel updates may conflict if actions modify overlapping parts of the tree.

**Solution:**
- Lock-based: Acquire write lock before `attach_and_refine()`
- Optimistic: Apply all updates sequentially after parallel eval completes
- Copy-on-write: Each worker gets WM snapshot, merge at end

**Recommended:** Optimistic approach (sequential merge after parallel eval)

### 2. Diversity in Action Selection

**Problem:** Utility function may pick K similar actions (e.g., all "use tensor cores" variants).

**Solution:**
```python
def choose_top_K_diverse_actions(K, diversity_bonus=0.2):
    """
    Select K actions with diversity penalty for similarity.
    """
    selected = []
    frontier = get_open_frontier_nodes()

    while len(selected) < K and frontier:
        utilities = []
        for action in frontier:
            base_utility = compute_utility(action)
            diversity_penalty = sum(
                similarity(action, s) * diversity_bonus
                for s in selected
            )
            utilities.append(base_utility - diversity_penalty)

        best_idx = argmax(utilities)
        selected.append(frontier.pop(best_idx))

    return selected

def similarity(action1, action2):
    """
    Compute similarity between actions (0=different, 1=identical).
    - Same parent node: +0.5
    - Similar tactics (cosine similarity of action text): +0.5
    """
    same_parent = 0.5 if action1.parent_id == action2.parent_id else 0.0
    text_sim = cosine_similarity(
        embed(action1.title + action1.rationale),
        embed(action2.title + action2.rationale)
    )
    return same_parent + 0.5 * text_sim
```

### 3. Round Accounting

**Problem:** If K actions run in parallel for N attempts each, do we count this as N rounds or K*N rounds?

**Options:**
- **Wall-clock rounds:** Count max(rounds_per_action) - treats parallel work as "free"
- **Total rounds:** Count sum(rounds_per_action) - preserves budget semantics
- **Hybrid:** Count max + penalty for stragglers

**Recommended:** Wall-clock rounds for fairness (parallelism shouldn't penalize budget)

### 4. GPU Isolation

**Problem:** Ensure each worker uses only its assigned GPU.

**Solution:**
```python
def explore_action_on_gpu(action, gpu_id, ...):
    # Method 1: Environment variable (simplest)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Method 2: PyTorch/Triton device context
    import torch
    with torch.cuda.device(gpu_id):
        benchmark(solution)

    # Method 3: Subprocess isolation (most robust)
    subprocess.run([
        'python', 'evaluate.py',
        '--gpu', str(gpu_id),
        '--solution', solution_path
    ])
```

## Files to Modify

1. **kernel_generator_world_model.py**
   - Add `parallel_world_model_search()` method
   - Refactor `_generate_world_model_cycles_v2()` to extract `explore_action_cycle()`
   - Add `explore_action_on_gpu()` worker function

2. **world_model_manager.py**
   - Add `choose_top_K_actions()` with diversity scoring
   - Add `batch_attach_and_refine()` for merging parallel results
   - Add `serialize_wm_state()` / `deserialize_wm_state()` for worker isolation

3. **world_model.py**
   - Add `action_similarity()` utility function
   - Add diversity penalty to utility calculation

4. **task.py** (benchmark interface)
   - Ensure `run_benchmark()` is thread-safe
   - Add GPU affinity parameter

## Evaluation Plan

**Baseline:** Serial WorldModelKernelGenerator on TriMul (300 rounds, 1 GPU)

**Experiment:** Parallel version with K={2,4,8} GPUs, same 300 total rounds

**Metrics:**
- Wallclock time (expect ~K× speedup)
- Best solution quality (should match or exceed serial)
- Unique actions explored (should increase with K)
- GPU utilization (target >90% per GPU)

## Priority

**High** - This directly addresses hardware underutilization and could significantly accelerate optimization.

## References

- Current serial implementation: kernel_generator_world_model.py:306-710
- Action selection: world_model_manager.py:775-954
- Utility function: world_model_manager.py:318-339
