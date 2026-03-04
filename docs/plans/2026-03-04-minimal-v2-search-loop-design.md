# Minimal V2 Search Loop Design

Design for implementing a bare minimum V2 search loop, validated against V1.

## Overview

Two-phase implementation:
1. **Doc reconciliation** - Fix naming inconsistencies in design docs
2. **Bare minimum search loop** - Simple sequential optimization loop with `run_search()` function

World model integration (SearchOrchestrator, tree structures) is a separate follow-on task.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Loop structure | Function, not class | Simpler; refactor to SearchOrchestrator with world model |
| LLM interface | `Callable[[str], str]` | Decoupled from OpenAI specifics |
| Tree/node abstraction | Deferred | Not needed until world model |
| Evaluator | GpuModeEvaluator delegates to V1 | Reuses battle-tested `task.run_benchmark()` |
| Multi-LLM routing | Deferred | Single LLM for bare minimum |

## Extension Path

FeedbackProvider/Analyzer hook into prompt building via EvalOutcome:

```
Round N completes
    ├─► EvalOutcome(impl, result, analysis=None)
    ├─► [optional] Analyzer populates analysis
    └─► FeedbackProvider.for_codegen(outcome) → prompt feedback
```

## Implementation Plans

- `impls/02-doc-reconciliation.md` - Phase 1: Fix naming
- `impls/03-bare-minimum-search-loop.md` - Phase 2: run_search() + GpuModeEvaluator

## Deferred to World Model Follow-on

- SolutionTree / SolutionNode data structures
- SearchOrchestrator class
- ActionSelector protocol
- LLMWorldModel implementation
- Multi-LLM routing
- Solution genealogy tracking

## References

- Task framework: `2026-03-04-task-framework-design.md`
- Full V2 design: `2026-03-04-search-v2-design.md`
- Incremental roadmap: `2026-03-04-incremental-implementation-design.md`
