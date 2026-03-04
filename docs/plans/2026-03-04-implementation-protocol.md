# Implementation Protocol Refinement

## Context

Follow-up to `2026-03-04-task-framework-design.md`. Clarifies the relationship between references, solutions, and how they're represented.

## Design Decision

**`Implementation` is a data container. `Evaluator` runs it.**

- `Implementation`: flat, generic data (source string, file paths, metadata)
- `Evaluator`: knows how to run implementations for its task type

No separate `ImplementationRunner` needed — the Evaluator already plays this role.

## Implementation Protocol

```python
from typing import Any, Protocol

class Implementation(Protocol):
    """Data container for code to be evaluated."""

    name: str
    """Identifier for this implementation."""

    content: Any
    """
    The implementation data. Format is task-specific:
    - str: single source file
    - dict[str, str]: multiple files {filename: content}
    - Path: reference to file on disk
    - etc.
    """
```

Using attributes (not properties) — simpler for a data container, and dataclasses work directly:

```python
@dataclass
class MyImpl:
    name: str
    content: str  # or dict, Path, etc.
```

This is intentionally flat and generic:
- No `run()` method — Implementation is data, not behavior
- `content` type is `Any` — adapters define what makes sense for their task
- Could add optional metadata (language, entry_point, etc.) as needed

### Reference and Solution Are Both Implementation

| What | Representation |
|------|----------------|
| Reference | `Implementation` (from task definition) |
| Generated solution | `Implementation` (from LLM output) |

Same type. Evaluator handles both the same way.

### Evaluator Runs Implementations

The Evaluator knows how to:
1. Load/prepare an Implementation for execution
2. Run it on input data
3. Check correctness against reference
4. Return results

```python
class Evaluator(Protocol):
    def evaluate(self, impl: Implementation) -> EvaluationResult:
        """Run implementation and return results."""
        ...
```

Internally, the Evaluator handles:
- Loading (file writes, module imports, etc.)
- Execution (subprocess, in-process, etc.)
- Timing and result capture

### Evaluator Flow

```python
def evaluate(self, solution: Implementation) -> EvaluationResult:
    # 1. Generate input
    input_data = self.input_generator.generate(params, seed)

    # 2. Run reference (impl-specific loading/execution)
    expected = self._run_implementation(self.reference, input_data)

    # 3. Run solution (same interface)
    actual = self._run_implementation(solution, input_data)

    # 4. Check and score
    check = self.correctness_checker.check(actual, expected)
    return EvaluationResult(...)
```

### Relationship to Executor

```
Executor              Evaluator              Implementation
────────              ─────────              ──────────────
HOW to orchestrate    HOW to run code        DATA container
(parallel, async)     (load, execute)        (source, files)

execute(impl, eval) → evaluator.evaluate(impl)
                         ├─ _run_implementation(reference, data)
                         └─ _run_implementation(solution, data)
```

- **Executor**: Orchestration (sequential, parallel, pipelined)
- **Evaluator**: Runs implementations (loading, execution, checking)
- **Implementation**: The code data

## EvalOutcome

Pairs a solution with its evaluation result. Extension point for additional data (e.g., analysis results).

```python
@dataclass
class EvalOutcome:
    solution: Implementation
    result: EvaluationResult
    # Extension fields added by subclasses or composition
```

The `solution` field is `Implementation`, not a separate type — consistent with reference and solution being the same protocol.
