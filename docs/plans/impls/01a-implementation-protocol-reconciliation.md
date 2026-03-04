# Implementation Protocol Reconciliation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Rename `SolutionArtifact` to `Implementation` per the refined design in `2026-03-04-implementation-protocol.md`.

**Architecture:** Mechanical renaming across code and docs. Change from properties to attributes for simpler data container semantics.

**Tech Stack:** Python protocols, pytest

---

## Task 1: Rename Protocol in results.py

**Files:**
- Modify: `k_search/task_framework/protocols/results.py`

**Step 1: Rename the protocol and change to attributes**

Change from properties to simple attributes per the design doc:

```python
class Implementation(Protocol):
    """Data container for code to be evaluated.

    Format is task-specific:
    - str: single source file
    - dict[str, str]: multiple files {filename: content}
    - Path: reference to file on disk
    """

    name: str
    """Identifier for this implementation."""

    content: Any
    """The implementation data."""
```

**Step 2: Verify file**

Run: `ruff check k_search/task_framework/protocols/results.py`

---

## Task 2: Update EvalOutcome in types.py

**Files:**
- Modify: `k_search/task_framework/types.py`

**Step 1: Update import and field name**

Change the TYPE_CHECKING import:
```python
if TYPE_CHECKING:
    from k_search.task_framework.protocols.results import EvaluationResult, Implementation
```

Change the EvalOutcome dataclass (using `impl` per naming convention):
```python
@dataclass
class EvalOutcome:
    """Complete result of evaluating an implementation."""

    impl: Implementation
    result: EvaluationResult
    analysis: AnalysisResult | None = None
```

**Step 2: Verify**

Run: `ruff check k_search/task_framework/types.py`

---

## Task 3: Update Wrapper Class

**Files:**
- Modify: `k_search/task_framework/adapters/wrappers.py`

**Step 1: Rename class and change to attributes**

Rename `GpuModeSolutionArtifact` to `GpuModeImplementation`.

Change from properties to attributes set in `__init__`:

```python
class GpuModeImplementation:
    """Wrapper exposing V1 Solution as Implementation protocol."""

    def __init__(self, inner: Solution) -> None:
        self._inner = inner
        self.name = inner.name
        self.content = inner
```

Remove the `@property` decorators for `name` and `content`.

**Step 2: Verify**

Run: `ruff check k_search/task_framework/adapters/wrappers.py`

---

## Task 4: Update Protocol Imports

**Files:**
- Modify: `k_search/task_framework/protocols/analyzer.py`
- Modify: `k_search/task_framework/protocols/evaluator.py`

**Step 1: Update analyzer.py**

Change import and parameter name (using `impl` per naming convention):
```python
from k_search.task_framework.protocols.results import EvaluationResult, Implementation

class Analyzer(Protocol):
    def analyze(
        self,
        impl: Implementation,
        result: EvaluationResult,
        context: dict[str, Any] | None = None,
    ) -> AnalysisResult:
```

**Step 2: Update evaluator.py**

Change import and parameter name (using `impl` per naming convention):
```python
from k_search.task_framework.protocols.results import EvaluationResult, Implementation

class Evaluator(Protocol):
    def evaluate(
        self,
        impl: Implementation,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> EvaluationResult:
```

**Step 3: Verify**

Run: `ruff check k_search/task_framework/protocols/`

---

## Task 5: Update All Exports

**Files:**
- Modify: `k_search/task_framework/protocols/__init__.py`
- Modify: `k_search/task_framework/adapters/__init__.py`
- Modify: `k_search/task_framework/__init__.py`

**Step 1: Update protocols/__init__.py**

Replace `SolutionArtifact` with `Implementation` in import and `__all__`.

**Step 2: Update adapters/__init__.py**

Replace `GpuModeSolutionArtifact` with `GpuModeImplementation` in import and `__all__`.

**Step 3: Update task_framework/__init__.py**

Replace both names in imports and `__all__`.

**Step 4: Verify imports work**

Run: `python -c "from k_search.task_framework import Implementation, GpuModeImplementation; print('OK')"`

---

## Task 6: Update Tests

**Files:**
- Modify: `tests/task_framework/test_wrappers.py`
- Modify: `tests/task_framework/test_gpu_mode_adapter.py`
- Modify: `tests/task_framework/test_e2e_causal_conv1d.py`

**Step 1: Update test_wrappers.py**

- Rename `TestGpuModeSolutionArtifact` → `TestGpuModeImplementation`
- Update imports: `GpuModeSolutionArtifact` → `GpuModeImplementation`
- Update `TestEvalOutcome` to use `.impl` field instead of `.solution`

**Step 2: Update test_gpu_mode_adapter.py**

- Update imports: `GpuModeSolutionArtifact` → `GpuModeImplementation`
- Update all usages in test code

**Step 3: Update test_e2e_causal_conv1d.py**

- Update imports: `GpuModeSolutionArtifact` → `GpuModeImplementation`
- Update all usages in test code

**Step 4: Run all tests**

Run: `CUDA_VISIBLE_DEVICES=3 python -m pytest tests/task_framework/ -v`
Expected: All 23 tests pass

---

## Task 7: Verify Code Quality

**Step 1: Run ruff**

Run: `ruff check k_search/task_framework/ tests/task_framework/`
Expected: All checks passed

**Step 2: Run ty**

Run: `ty check k_search/task_framework/`
Expected: All checks passed

**Step 3: Verify top-level imports**

Run: `python -c "from k_search.task_framework import *; print('All exports OK')"`

---

## Task 8: Update Design Docs

**Files:**
- Modify: `docs/plans/2026-03-04-task-framework-design.md`
- Modify: `docs/plans/2026-03-04-task-framework-extensions.md`
- Modify: `docs/plans/2026-03-04-incremental-implementation-design.md`
- Modify: `docs/plans/2026-03-04-search-v2-design.md`

**Step 1: Global replace in each file**

Replace all occurrences:
- `SolutionArtifact` → `Implementation`
- `GpuModeSolutionArtifact` → `GpuModeImplementation`

For EvalOutcome references, update field name:
- `.solution` → `.impl`

**Step 2: Verify no stale references**

Run: `grep -r "SolutionArtifact" docs/plans/`
Expected: No matches (except possibly in historical notes)

---

## Task 9: Add Historical Note to Original Plan

**Files:**
- Modify: `docs/plans/impls/01-task-framework-foundation.md`

**Step 1: Add note at top after title**

```markdown
> **Note:** This plan used `SolutionArtifact` naming. Post-implementation, this was reconciled to `Implementation` per `2026-03-04-implementation-protocol.md`. See `01a-implementation-protocol-reconciliation.md`.
```

---

## Task 10: Commit

**Step 1: Stage and commit from K-Search directory**

```bash
cd K-Search && git add -A && git commit -m "$(cat <<'EOF'
refactor(task_framework): rename SolutionArtifact to Implementation

Reconciles naming with 2026-03-04-implementation-protocol.md:
- SolutionArtifact → Implementation (now uses attrs, not properties)
- GpuModeSolutionArtifact → GpuModeImplementation
- EvalOutcome.solution → EvalOutcome.impl
- Updated all design docs for consistency

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

After completing all tasks:
- Protocol renamed: `Implementation` with simple `name`/`content` attributes (was `SolutionArtifact` with properties)
- Wrapper renamed: `GpuModeImplementation` (was `GpuModeSolutionArtifact`)
- Field renamed: `EvalOutcome.impl` (was `.solution`)
- All docs updated for consistency
- All 23 tests passing
