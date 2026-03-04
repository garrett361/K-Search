# Implementation Plan: Doc Reconciliation

Fix naming inconsistencies across design docs to align with implemented code.

## Prerequisites

- Priority 1 (task_framework) complete

## Changes

| Find | Replace |
|------|---------|
| `SolutionArtifact` | `Implementation` |
| `EvalOutcome.solution` | `EvalOutcome.impl` |
| `.solution` (in EvalOutcome context) | `.impl` |

## Files to Modify

1. `docs/plans/2026-03-04-search-v2-design.md`
2. `docs/plans/2026-03-04-task-framework-design.md`
3. `docs/plans/2026-03-04-task-framework-north-star.md`
4. `docs/plans/impls/01-task-framework-foundation.md`

## Files to Skip

- `01a-implementation-protocol-reconciliation.md` - reconciliation plan, keep as reference
- `2026-03-04-implementation-protocol.md` - already correct
- `2026-03-04-incremental-implementation-design.md` - already correct

## Tasks

### 1. Update search-v2-design.md
- [ ] Replace `SolutionArtifact` with `Implementation`
- [ ] Replace `.solution` with `.impl` in EvalOutcome references
- [ ] Verify code examples use correct naming

### 2. Update task-framework-design.md
- [ ] Replace `SolutionArtifact` with `Implementation`
- [ ] Replace `.solution` with `.impl` in EvalOutcome references

### 3. Update task-framework-north-star.md
- [ ] Replace naming in architecture diagrams
- [ ] Update any ASCII diagrams showing data flow

### 4. Update 01-task-framework-foundation.md
- [ ] Replace `SolutionArtifact` with `Implementation`
- [ ] Update task descriptions if they reference old naming

### 5. Verify changes
- [ ] `grep -r SolutionArtifact docs/plans/` returns only 01a reconciliation doc
- [ ] `grep -r "\.solution" docs/plans/` shows no EvalOutcome.solution references

## Validation

```bash
# Should return only 01a-implementation-protocol-reconciliation.md
grep -r "SolutionArtifact" K-Search/docs/plans/

# Should return 0 matches for EvalOutcome.solution pattern
grep -r "EvalOutcome\.solution\|outcome\.solution" K-Search/docs/plans/
```

## Estimated Effort

~30 minutes (search/replace + manual verification)
