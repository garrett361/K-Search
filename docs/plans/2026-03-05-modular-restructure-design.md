# Modular Restructure Design

Consolidate `task_framework/` and `search_v2/` into a unified `modular/` module with consistent naming.

## Goals

1. Merge artificially separated modules (`task_framework` + `search_v2`) into one cohesive unit
2. Fix confusing naming (`EvalOutcome` → `Round`, generic `types.py` → descriptive names)
3. Align test structure with source structure
4. Update documentation references

## Non-Goals

- Touching v1 code (`k_search/tasks/`, `k_search/kernel_generators/`, `k_search/utils/`)
- Changing protocol interfaces (only renaming files/classes)
- Adding new functionality beyond expanding `Round`

## Directory Structure

### Before

```
k_search/
├── task_framework/
│   ├── __init__.py
│   ├── types.py                    # CheckResult, AnalysisResult, EvalOutcome
│   ├── llm_utils.py
│   ├── protocols/
│   │   ├── results.py              # EvaluationResult, Implementation
│   │   ├── input_generator.py
│   │   ├── reference_impl.py
│   │   ├── correctness.py
│   │   ├── scorer.py
│   │   ├── feedback_provider.py
│   │   ├── evaluator.py
│   │   ├── analyzer.py
│   │   └── task_definition.py
│   └── adapters/
│       └── gpu_mode/
│           ├── types.py            # GpuModeEvaluationResult, GpuModeImplementation
│           ├── task_definition.py
│           └── evaluator.py
├── search_v2/
│   ├── __init__.py
│   ├── loop.py
│   ├── config.py
│   ├── prompts.py
│   ├── metrics/
│   │   ├── protocol.py             # MetricsTracker
│   │   ├── noop.py
│   │   └── wandb.py
│   └── artifacts/
│       ├── protocol.py             # ArtifactStore
│       ├── noop.py
│       ├── local.py
│       └── wandb.py
├── tasks/                          # v1 - unchanged
├── kernel_generators/              # v1 - unchanged
└── utils/                          # unchanged
```

### After

```
k_search/
├── modular/
│   ├── __init__.py
│   ├── round.py                    # Round (expanded container)
│   ├── results.py                  # CheckResult, AnalysisResult
│   ├── llm_utils.py
│   ├── loop.py
│   ├── config.py
│   ├── prompts.py
│   ├── protocols/
│   │   ├── __init__.py
│   │   ├── eval_result.py          # EvaluationResult protocol
│   │   ├── impl.py                 # Implementation protocol
│   │   ├── input_generator.py
│   │   ├── reference_impl.py
│   │   ├── correctness.py
│   │   ├── scorer.py
│   │   ├── feedback_provider.py
│   │   ├── evaluator.py
│   │   ├── analyzer.py
│   │   └── task_definition.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── gpu_mode/
│   │       ├── __init__.py
│   │       ├── wrappers.py         # GpuModeEvaluationResult, GpuModeImplementation
│   │       ├── task_definition.py
│   │       └── evaluator.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── tracker.py              # MetricsTracker protocol
│   │   ├── noop.py
│   │   └── wandb.py
│   └── artifacts/
│       ├── __init__.py
│       ├── store.py                # ArtifactStore protocol
│       ├── noop.py
│       ├── local.py
│       └── wandb.py
├── tasks/                          # v1 - unchanged
├── kernel_generators/              # v1 - unchanged
└── utils/                          # unchanged
```

## Class Changes

### `Round` (formerly `EvalOutcome`)

Expanded to contain complete round information:

```python
@dataclass
class Round:
    # Existing fields
    impl: Implementation
    result: EvaluationResult
    analysis: AnalysisResult | None = None

    # New fields
    prompt: str                    # The prompt sent to LLM
    llm_response: str              # Raw LLM output before parsing
    prompt_tokens: int             # Estimated from chars
    completion_tokens: int         # Estimated from chars
    duration_secs: float           # Round execution time
    score: float                   # From scorer.score()
```

This allows downstream consumers (`FeedbackProvider`, `ArtifactStore`, metrics) to access the full round context.

## File Renames

| From | To | Rationale |
|------|-----|-----------|
| `task_framework/types.py` | `modular/round.py` + `modular/results.py` | Split: Round container vs operation results |
| `protocols/results.py` | `protocols/eval_result.py` + `protocols/impl.py` | Split: distinct protocols get own files |
| `metrics/protocol.py` | `metrics/tracker.py` | Name matches protocol (MetricsTracker) |
| `artifacts/protocol.py` | `artifacts/store.py` | Name matches protocol (ArtifactStore) |
| `adapters/gpu_mode/types.py` | `adapters/gpu_mode/wrappers.py` | These are adapter wrappers |

## Test Structure

### Before

```
tests/
├── task_framework/
│   ├── test_wrappers.py
│   ├── test_gpu_mode_task_definition.py
│   └── test_e2e_causal_conv1d.py
├── search_v2/
│   ├── test_loop.py
│   ├── test_e2e_search.py
│   ├── test_metrics.py
│   └── artifacts/test_stores.py
├── test_gpu_mode_causal_conv1d.py      # misplaced
├── test_gpu_mode_moe.py                # misplaced
├── test_gpu_mode_reference_benchmark.py # misplaced
├── test_kernel_generator.py            # misplaced
├── test_world_model_llm_integration.py # misplaced
└── conftest.py
```

### After

```
tests/
├── modular/
│   ├── test_gpu_mode_wrappers.py
│   ├── test_gpu_mode_task_definition.py
│   ├── test_e2e_causal_conv1d.py
│   ├── test_loop.py
│   ├── test_e2e_search.py
│   ├── test_metrics.py
│   └── test_artifacts.py
├── tasks/
│   └── gpu_mode/
│       ├── test_causal_conv1d.py
│       ├── test_moe.py
│       └── test_reference_benchmark.py
├── kernel_generators/
│   ├── test_generator.py
│   └── test_world_model.py
└── conftest.py
```

## Import Path Changes

| From | To |
|------|-----|
| `k_search.task_framework` | `k_search.modular` |
| `k_search.task_framework.types.EvalOutcome` | `k_search.modular.round.Round` |
| `k_search.task_framework.types.CheckResult` | `k_search.modular.results.CheckResult` |
| `k_search.task_framework.protocols.results.EvaluationResult` | `k_search.modular.protocols.eval_result.EvaluationResult` |
| `k_search.task_framework.protocols.results.Implementation` | `k_search.modular.protocols.impl.Implementation` |
| `k_search.search_v2` | `k_search.modular` |
| `k_search.search_v2.loop.run_search` | `k_search.modular.loop.run_search` |
| `k_search.search_v2.config.SearchConfig` | `k_search.modular.config.SearchConfig` |

## Documentation Updates

23 documentation files reference affected structures. Key updates:

1. **CLAUDE.md** - Update architecture section, key types
2. **README.md** - Update architecture diagram
3. **docs/plans/*.md** - Update import paths, type names
4. **docs/llm-info-routing.md** - Update file references

Search patterns for find/replace:
- `task_framework` → `modular`
- `search_v2` → `modular`
- `EvalOutcome` → `Round`
- `from k_search.task_framework` → `from k_search.modular`
- `from k_search.search_v2` → `from k_search.modular`

## Migration Strategy

1. Create `k_search/modular/` with new structure
2. Move and rename files
3. Update all imports within `modular/`
4. Expand `Round` class with new fields
5. Update `loop.py` to construct full `Round`
6. Update `run_search_v2.py` entry point
7. Move and rename test files
8. Update test imports
9. Update documentation
10. Delete empty `task_framework/` and `search_v2/` directories

## Risks

- **Import breakage**: External code importing from old paths will break. Mitigation: This is internal refactoring; no external consumers.
- **`Round` expansion**: Adding required fields to `Round` requires updating all construction sites. Mitigation: Only one place constructs it (`loop.py`).
- **Doc drift**: Documentation may have references we miss. Mitigation: Grep for old names after migration.
