# Implementation Plan: Bare Minimum V2 Search Loop

Simple sequential optimization loop using task_framework protocols.

## Prerequisites

- Priority 1 (task_framework) complete
- Doc reconciliation (02) complete

## Module Structure

```
k_search/search_v2/
├── __init__.py
├── config.py               # SearchConfig dataclass
├── loop.py                 # run_search() function
└── prompts.py              # Prompt formatting helpers

k_search/task_framework/adapters/
└── gpu_mode_evaluator.py   # GpuModeEvaluator (new)
```

## Tasks

### 1. Create GpuModeEvaluator

**File**: `k_search/task_framework/adapters/gpu_mode_evaluator.py`

```python
class GpuModeEvaluator:
    """Evaluator that delegates to GpuModeTask.run_benchmark()."""

    def __init__(self, task: GpuModeTask) -> None:
        self._task = task

    def evaluate(
        self,
        impl: Implementation,
        *,
        timeout_secs: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        solution = impl.content
        eval_result = self._task.run_benchmark(solution)
        return GpuModeEvaluationResult(eval_result)
```

- [ ] Create `gpu_mode_evaluator.py`
- [ ] Export from `task_framework/adapters/__init__.py`
- [ ] Add unit tests in `tests/task_framework/test_gpu_mode_evaluator.py`

### 2. Create search_v2 package structure

- [ ] Create `k_search/search_v2/__init__.py`
- [ ] Create `k_search/search_v2/config.py`
- [ ] Create `k_search/search_v2/loop.py`
- [ ] Create `k_search/search_v2/prompts.py`

### 3. Implement SearchConfig

**File**: `k_search/search_v2/config.py`

```python
@dataclass
class SearchConfig:
    max_rounds: int = 10
    timeout_secs: int | None = None

@dataclass
class SearchResult:
    impl: Implementation | None
    score: float
    result: EvaluationResult | None
    rounds_completed: int = 0
```

- [ ] Define `SearchConfig` dataclass
- [ ] Define `SearchResult` dataclass

### 4. Implement prompt building

**File**: `k_search/search_v2/prompts.py`

```python
def build_prompt(
    task: TaskDefinition,
    last_outcome: EvalOutcome | None,
) -> str:
    base = task.get_prompt_text()
    if last_outcome:
        feedback = task.feedback_provider.for_codegen(last_outcome)
        return f"{base}\n\n{feedback}"
    return base

def create_implementation(code: str, round_idx: int) -> Implementation:
    # Wrap code in GpuModeImplementation
    ...
```

- [ ] Implement `build_prompt()`
- [ ] Implement `create_implementation()` helper

### 5. Implement run_search

**File**: `k_search/search_v2/loop.py`

```python
LLMCall = Callable[[str], str]

def run_search(
    task: TaskDefinition,
    evaluator: Evaluator,
    llm: LLMCall,
    config: SearchConfig,
) -> SearchResult:
    best_impl: Implementation | None = None
    best_score: float = float("-inf")
    best_result: EvaluationResult | None = None

    for round_idx in range(config.max_rounds):
        # Log best so far
        if best_result:
            metrics = best_result.get_metrics()
            speedup = metrics.get("speedup_factor", "N/A")
            logger.info(
                f"Round {round_idx + 1}/{config.max_rounds} | "
                f"Best: {best_score:.4f} (speedup: {speedup})"
            )
        else:
            logger.info(
                f"Round {round_idx + 1}/{config.max_rounds} | "
                f"No solution found yet"
            )

        round_start = time.perf_counter()

        # Build prompt, generate, evaluate
        outcome = EvalOutcome(impl=best_impl, result=best_result) if best_impl else None
        prompt = build_prompt(task, outcome)
        code = llm(prompt)
        impl = create_implementation(code, round_idx)
        result = evaluator.evaluate(impl)
        score = task.scorer.score(result)

        if score > best_score:
            best_impl, best_score, best_result = impl, score, result

        round_elapsed = time.perf_counter() - round_start
        logger.info(
            f"Round {round_idx + 1} complete | "
            f"Score: {score:.4f} | "
            f"Time: {round_elapsed:.1f}s"
        )

    return SearchResult(
        impl=best_impl,
        score=best_score,
        result=best_result,
        rounds_completed=config.max_rounds,
    )
```

- [ ] Implement `run_search()` with logging
- [ ] Handle EvalOutcome construction

### 6. Add unit tests

**File**: `tests/search_v2/test_loop.py`

- [ ] Test `run_search()` with mock LLM and evaluator
- [ ] Test logging output
- [ ] Test best solution tracking
- [ ] Test config.max_rounds respected

### 7. Add integration test

**File**: `tests/search_v2/test_e2e_search.py`

- [ ] Test with real GpuModeTask (causal_conv1d)
- [ ] Test with mock LLM returning valid Triton code
- [ ] Verify evaluation results flow through correctly

### 8. Export from __init__.py

- [ ] Export `run_search`, `SearchConfig`, `SearchResult` from `search_v2/__init__.py`
- [ ] Export `GpuModeEvaluator` from `task_framework/adapters/__init__.py`

### 9. Create V2 entry point script

**File**: `run_search_v2.py` (or integrate into existing `generate_kernels_and_eval.py`)

- [ ] CLI argument parsing (task, model, max_rounds, etc.)
- [ ] LLM client construction (OpenAI-compatible)
- [ ] Wire up: GpuModeTask → GpuModeAdapter → GpuModeEvaluator → run_search()
- [ ] Print final SearchResult summary

### 10. Add runme recipe for V2

**File**: `k_search_expr/runme.yaml`

```yaml
run_search_v2:
  - recipe: validate_api_env
  - recipe: check_api
  - default: |
      cd {k_search_dir}
      python run_search_v2.py \
        --task {task} \
        --language {language} \
        --model-name {model_name} \
        --max-rounds {max_rounds} \
        --base-url "$RITS_BASE_URL" \
        --api-key "$RITS_API_KEY"
```

- [ ] Add `run_search_v2` recipe
- [ ] Add `run_search_v2_causal_conv1d_e2e` convenience recipe

### 11. Manual e2e testing via runme

**Prerequisites**:
- RITS API credentials configured (`RITS_API_KEY`, `RITS_BASE_URL`, `RITS_MODEL_NAME`)
- GPU available (H100/A100)

**Test commands**:

```bash
# Validate API access
runme run-bash validate_api_env

# Health check API endpoint
runme run-bash check_api

# Run V2 search on causal_conv1d (3 rounds, quick test)
runme run-bash run_search_v2 task=causal_conv1d max_rounds=3

# Run V2 search on trimul (full 10 rounds)
runme run-bash run_search_v2 task=trimul max_rounds=10

# Compare V1 vs V2 on same task
runme run-bash run_causal_conv1d_e2e  # V1
runme run-bash run_search_v2 task=causal_conv1d max_rounds=10  # V2
```

**Success criteria**:
- [ ] V2 completes without errors
- [ ] Logs show round progress with timing and speedup
- [ ] Final solution achieves comparable score to V1 (within 10%)
- [ ] No regressions in task_framework behavior

## Validation

```bash
# Unit tests pass
pytest tests/search_v2/test_loop.py -v

# Integration test passes
pytest tests/search_v2/test_e2e_search.py -v

# All task_framework tests still pass
pytest tests/task_framework/ -v
```

## Estimated Effort

~3-4 hours (implementation + manual e2e testing)

## Dependencies

```
task_framework (complete)
    └─► GpuModeEvaluator
            └─► search_v2/loop.py
                    └─► Unit tests
                    └─► Integration test
                    └─► Entry point script
                            └─► runme recipe
                                    └─► Manual e2e testing
```
