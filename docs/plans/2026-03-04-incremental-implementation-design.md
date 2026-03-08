# K-Search Incremental Implementation Design

Design for implementing K-Search V2 incrementally, validated against V1 at each step.

## Implementation Priorities

| Priority | Goal | Validation | Status |
|----------|------|------------|--------|
| 1 | Task framework foundation + V1 adapter | e2e causal_conv1d tests | ✅ DONE |
| 2 | Minimal V2 search loop (simple) | Unit tests | ✅ DONE |
| 2b | Full V2 loop (tree + world model) | Parity with V1 | 🔲 NOT STARTED |
| 3 | Wandb logging (minimal → enhanced) | Manual verification | ✅ DONE |
| 4 | Loop feedback extension | Unit tests + e2e with pattern injection | 🔲 NOT STARTED |

Each priority gets its own implementation plan created after the previous one is complete.

**Note:** Priority 2 was split. The simple sequential loop (`run_search()`) is complete. The full tree-based `SearchOrchestrator` with world model is Priority 2b.

---

## Priority 1: Task Framework Foundation ✅

Build protocols and adapters. V1 code unchanged—GpuModeTriMulTask wrapped via adapter.

### Module Structure (as implemented)

```
k_search/modular/
├── __init__.py
├── protocols/
│   ├── __init__.py
│   ├── eval_result.py          # EvaluationResult protocol
│   ├── impl.py                 # Implementation protocol
│   ├── input_generator.py
│   ├── reference_impl.py
│   ├── correctness.py          # CorrectnessChecker
│   ├── scorer.py
│   ├── feedback_provider.py
│   ├── evaluator.py
│   ├── analyzer.py
│   ├── task_definition.py
│   ├── metrics_tracker.py
│   └── artifact_store.py
├── adapters/
│   ├── __init__.py
│   └── gpu_mode/
│       ├── __init__.py
│       ├── task_definition.py  # GpuModeTriMulTaskDefinition
│       ├── evaluator.py        # GpuModeEvaluator
│       └── wrappers.py         # GpuModeImplementation, GpuModeEvaluationResult
├── round.py                    # Round dataclass
├── results.py                  # CheckResult, AnalysisResult
├── config.py                   # SearchConfig, SearchResult
├── loop.py                     # run_search() function
├── prompts.py                  # build_prompt()
├── llm_utils.py
├── metrics/
│   ├── __init__.py
│   ├── noop.py
│   └── wandb.py
└── artifacts/
    ├── __init__.py
    ├── noop.py
    ├── local.py
    └── wandb.py
```

### Key Protocols

```python
class EvaluationResult(Protocol):
    def is_success(self) -> bool: ...
    def get_metrics(self) -> dict[str, Any]: ...
    def get_log(self) -> str: ...

class Implementation(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def content(self) -> Any: ...

class Scorer(Protocol):
    def score(self, result: EvaluationResult) -> float: ...

class FeedbackProvider(Protocol):
    def for_codegen(self, outcomes: Round | list[Round]) -> str: ...
    def for_world_model(self, outcomes: Round | list[Round]) -> list[dict[str, Any]]: ...

class Analyzer(Protocol):
    def analyze(
        self,
        solution: Implementation,
        result: EvaluationResult,
        context: dict[str, Any] | None = None,
    ) -> AnalysisResult: ...
```

### Analyzer Context

Simple analyzers ignore context. Advanced analyzers (like FailurePatternAnalyzer) access:

```python
context = {
    'tree': SolutionTree,
    'recent_failures': list[Round],
}
```

### AnalysisResult

```python
@dataclass
class AnalysisResult:
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_artifact: bytes | None = None
    strategic_guidance: str | None = None  # For feedback to world model/codegen
```

---

## Priority 2: Minimal V2 Search Loop ✅

A simple sequential `run_search()` function was implemented in `loop.py`. This greedy loop doesn't use tree structures or world model—it just tracks the best result across rounds.

See `k_search/modular/loop.py` for the implementation.

---

## Priority 2b: Full V2 Search Loop (Tree + World Model) 🔲

### Module Structure (NOT YET IMPLEMENTED)

```
k_search/modular/
├── model/
│   ├── node.py                 # SolutionNode, ActionNode
│   ├── tree.py                 # SolutionTree
│   └── config.py               # RetryConfig (extends existing config.py)
├── protocols/
│   ├── action_selector.py      # ActionSelector
│   └── formatter.py            # StateFormatter
├── selectors/
│   └── world_model.py          # LLMWorldModel
├── formatters/
│   └── legacy_json.py          # V1-compatible schema
├── parsing/
│   ├── result.py               # ParseResult[T]
│   └── json_parser.py
└── search.py                   # SearchOrchestrator
```

### SearchOrchestrator

```python
class SearchOrchestrator:
    def __init__(
        self,
        task: TaskDefinition,
        evaluator: Evaluator,
        codegen_llm: LLMCall,
        selector: ActionSelector,
        formatter: StateFormatter,
        config: SearchConfig,
        # LLM routing (with defaults)
        world_model_llm: LLMCall | None = None,   # defaults to codegen_llm
        analyzer_llm: LLMCall | None = None,      # defaults to world_model_llm
        # Optional components
        analyzer: Analyzer | None = None,
        artifact_writer: ArtifactWriter | None = None,
        metrics_tracker: MetricsTracker | None = None,
    ): ...

    def run(self) -> SolutionNode | None:
        self._initialize_tree()
        for round_idx in range(self.config.max_rounds):
            actions = self.selector.select(self.tree, k=1)
            if not actions:
                break
            outcome = self._execute_action(actions[0])
            self.selector.update(self.tree, actions[0], outcome)
            self._run_analyzer_if_needed(outcome)
            self._write_artifacts(outcome)
            self._log_metrics(round_idx, outcome)
        return self.tree.get_best_solution()
```

### Parity Testing

Replay recorded V1 runs through V2, compare outputs.

---

## Priority 3: Wandb Logging ✅

**Phase 3a (minimal)**: Round count, best score, pass/fail, wall clock timing — DONE

**Phase 3b (enhanced)**: Token counting, loop timing breakdown, code artifact saving — DONE

Implemented in `k_search/modular/metrics/wandb.py` and `k_search/modular/artifacts/wandb.py`.

---

## Priority 4: Loop Feedback Extension

### FailurePatternAnalyzer

```python
class FailurePatternAnalyzer:
    def __init__(self, llm: LLMCall | None = None):  # Defaults to world_model_llm
        ...

    def analyze(
        self,
        solution: Implementation,
        result: EvaluationResult,
        context: dict[str, Any] | None = None,  # tree + recent_failures
    ) -> AnalysisResult:
        # Summarize failure patterns across recent_failures
        # Return strategic_guidance for next round
        ...
```

### Data Flow

```
Round N fails
     │
     ├──► FeedbackProvider.for_codegen(outcome)
     │         "Error: index out of bounds at line 42"
     │
     └──► FailurePatternAnalyzer.analyze(outcome, context)
               strategic_guidance: "Last 5 failures all hit shared memory limits"
     │
     ▼
Round N+1 prompt includes BOTH
```

### LLM Routing

```
codegen_llm        ─── required
world_model_llm    ─── defaults to codegen_llm
analyzer_llm       ─── defaults to world_model_llm
```

---

## Artifact Storage

```
<artifacts>/<run_name>/
├── nodes/
│   └── <node_id>/
│       ├── metadata.json       # {node_id, parent_id, status, score}
│       └── kernel.py           # (or multiple files for CUDA)
└── logs/
    └── rounds.jsonl            # Per-round events with eval details
```

- **run_name**: User-provided or auto-generated `<task>_<ts>_<uuid>`
- **Write timing**: Incremental after each round

---

## Deferred

- Tree serialization / checkpointing
- Run reproducibility
- Resume from checkpoint

---

## References

- Task framework details: `2026-03-04-task-framework-design.md`
- Search V2 details: `2026-03-04-search-v2-design.md`
- Extensions: `2026-03-04-task-framework-extensions.md`
- Modular restructure: `2026-03-05-modular-restructure-design.md`

### Implementation Plans

- `impls/01-task-framework-foundation.md` — Priority 1
- `impls/03-bare-minimum-search-loop.md` — Priority 2
- `impls/04a-wandb-metrics-integration.md` — Priority 3a
- `impls/04b-wandb-artifacts-integration.md` — Priority 3b
- `impls/05-modular-restructure.md` — File reorganization
