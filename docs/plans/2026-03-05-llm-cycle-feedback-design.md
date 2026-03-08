# LLM Cycle Feedback Design

Design for LLM-based failure summarization within search cycles.

## Problem

Within a cycle (multiple rounds pursuing one action), repeated failures often share patterns. Without feedback, the codegen LLM may repeat the same mistakes. This design adds an LLM-based analyzer that summarizes failure patterns and feeds them to subsequent attempts.

## Overview

```
Failed round
    │
    ▼
cycle_outcomes.append(outcome)
    │
    ▼
LLMFailureAnalyzer.analyze(context={cycle_outcomes})
    │
    ▼
AnalysisResult(kind="failure_summary", summary="...")
    │
    ▼
outcome.analysis = analysis
    │
    ▼
FeedbackProvider.for_codegen(outcome)
    │
    ▼
Codegen prompt with "Previous Issues" section
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration point | LLMAnalyzer implementing Analyzer protocol | Matches existing extension point, testable in isolation |
| Cycle state | Local `list[Round]` passed via context | Simple, no new dataclass needed for MVP |
| LLM for summarization | Same as codegen model | No separate model needed initially |
| Analyzer configuration | Optional, default none | Explicit opt-in, no magic defaults |
| Trigger | After each failed round | Progressive summary, cumulative history |

## Component Changes

### AnalysisResult Refactor

**File:** `k_search/modular/results.py`

Current specialized fields replaced with generic container:

```python
@dataclass
class AnalysisResult:
    kind: str | None = None       # Discriminator for routing (e.g., "failure_summary", "ncu_profile")
    summary: str | None = None    # Human/LLM-readable text
    data: dict[str, Any] = field(default_factory=dict)  # Structured data
```

All fields optional. The `kind` field enables FeedbackProvider to route different analysis types to appropriate prompt sections.

### LLMFailureAnalyzer

**File:** `k_search/modular/analyzers/llm_failure_analyzer.py` (new)

```python
@dataclass
class LLMFailureAnalyzerConfig:
    min_failures: int = 1              # Minimum failures before generating summary
    last_n_failures: int | None = None # None = all failures

class LLMFailureAnalyzer:
    """Summarizes failure patterns using an LLM.

    Typically uses the same LLM as code generation.
    """

    kind = "failure_summary"

    def __init__(
        self,
        llm: Callable[[str], str],
        config: LLMFailureAnalyzerConfig | None = None,
    ):
        self.llm = llm
        self.config = config or LLMFailureAnalyzerConfig()

    def analyze(
        self,
        impl: Implementation,
        result: EvaluationResult,
        context: dict[str, Any] | None = None,
    ) -> AnalysisResult | None:
        cycle_outcomes = (context or {}).get("cycle_outcomes", [])
        failures = [o for o in cycle_outcomes if not o.result.is_success()]

        if len(failures) < self.config.min_failures:
            return None

        recent = failures
        if self.config.last_n_failures is not None:
            recent = failures[-self.config.last_n_failures:]

        prompt = self._format_summarization_prompt(recent)
        summary = self.llm(prompt)

        return AnalysisResult(
            kind=self.kind,
            summary=summary.strip(),
            data={"failure_count": len(failures)},
        )

    def _format_summarization_prompt(self, failures: list[Round]) -> str:
        """Format failures into prompt for summarization LLM."""
        ...
```

### Available Failure Data (GPU Mode Specifics)

> **Note:** This section documents GPU Mode adapter internals. The v2 core components (LLMFailureAnalyzer, FeedbackProvider, etc.) interact only with protocol interfaces (`EvaluationResult.get_log()`, `EvaluationResult.get_metrics()`). Adapters are responsible for surfacing appropriate data through these interfaces.

The LLMFailureAnalyzer accesses failure data via protocol methods:

| Protocol Method | What Adapters Should Provide |
|-----------------|------------------------------|
| `result.get_log()` | Relevant error/failure information |
| `result.get_metrics()` | Structured timing/status data |
| `impl.content` | Source code of the attempt |

**GPU Mode adapter specifics (for adapter implementers):**

| GpuMode Source | Limitation | Adapter Consideration |
|----------------|------------|----------------------|
| `log_excerpt` | 8KB cap, first failure only | Consider extracting from `raw_result` if richer detail needed |
| `raw_result` dict | Contains `benchmark.{i}.error` per benchmark | Can surface via custom adapter |
| Per-benchmark timing | In `raw_result` as `benchmark.{i}.mean` etc. | Not currently exposed |

**Design principle:** v2 components remain task-agnostic. If GPU Mode needs richer failure data surfaced, implement a custom adapter that extracts from `raw_result` and exposes via `get_log()` or `get_metrics()`.

### Loop Integration

The loop/orchestrator tracks cycle history and invokes analyzers:

```
┌─────────────────────────────────────────────────────────────────┐
│                         run_search / Orchestrator               │
│                                                                 │
│   cycle_outcomes: list[Round] = []                        │
│                                                                 │
│   for round in cycle:                                           │
│       impl = create_impl(llm(prompt))                 │
│       result = evaluator.evaluate(impl)                         │
│       outcome = Round(impl, result)                       │
│                                                                 │
│       if not result.is_success():                                   │
│           cycle_outcomes.append(outcome)                        │
│           if analyzer:                                          │
│               analysis = analyzer.analyze(                      │
│                   impl, result,                                 │
│                   context={"cycle_outcomes": cycle_outcomes}    │
│               )                                                 │
│               if analysis:                                      │
│                   outcome.analysis = analysis                   │
│                                                                 │
│       # outcome (with analysis) used for next prompt            │
│                                                                 │
│   # Reset cycle_outcomes on cycle boundary                      │
└─────────────────────────────────────────────────────────────────┘
```

Analyzers are optional; configuration mechanism TBD as `run_search` evolves into `SearchOrchestrator`.

### FeedbackProvider Integration

`FeedbackProvider.for_codegen()` routes analysis by `kind`:

```python
def for_codegen(self, outcomes: Round | list[Round]) -> str:
    # ...
    for outcome in outcomes:
        if outcome.analysis:
            if outcome.analysis.kind == "failure_summary":
                # Include in "Previous Issues" section
            elif outcome.analysis.kind == "ncu_profile":
                # Include in "Profiling Data" section
    # ...
```

## Future Considerations

### Multiple Analyzers

`Round.analysis` currently holds a single `AnalysisResult | None`. For multiple analyzers:

```python
# Option A: Union type
analysis: AnalysisResult | list[AnalysisResult] | None = None

# Option B: Always list
analyses: list[AnalysisResult] = field(default_factory=list)
```

The `kind` field on `AnalysisResult` distinguishes different analysis types.

### Additional Analyzers

The same pattern supports other analyzers:
- NCU profiler: `kind="ncu_profile"`
- Memory analysis: `kind="memory_analysis"`
- Static analysis: `kind="static_analysis"`

Each produces `AnalysisResult` with appropriate `kind`, routed by FeedbackProvider.

## References

- Task framework design: `2026-03-04-task-framework-design.md`
- Search V2 design: `2026-03-04-search-v2-design.md`
- Minimal V2 loop: `2026-03-04-minimal-v2-search-loop-design.md`
