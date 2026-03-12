# Design: Move Modular Loop to scripts/gpu_mode_modular_v1

**Date:** 2026-03-09
**Status:** Approved

## Summary

Refactor `k_search/modular/loop.py` and `run_search_v2.py` into `scripts/gpu_mode_modular_v1/` following the pattern established by `scripts/gpu_mode_simple_linear_executor/`.

## Motivation

- Consistency: Match the `scripts/` directory pattern where entry points are self-contained
- Visibility: Core loop logic visible in one file rather than split across modules
- Simplicity: Easier to understand and modify as a standalone script

## Design Decisions

1. **Self-contained with selective imports**: Inline `run_search` and `build_prompt` into `run.py`, but keep importing config/metrics/artifacts/adapters from `k_search/modular/`
2. **Delete source files**: Remove `k_search/modular/loop.py` and `k_search/modular/prompts.py`
3. **Move tests**: Consolidate tests into `scripts/gpu_mode_modular_v1/test_run.py`
4. **Remove exports**: Remove `run_search` and `LLMCall` from `k_search/modular/__init__.py`

## File Structure After Refactoring

```
K-Search/
├── scripts/
│   ├── gpu_mode_modular_v1/
│   │   ├── __init__.py          # Empty
│   │   ├── run.py               # CLI + inlined run_search + prompts (~320 lines)
│   │   └── test_run.py          # Merged tests (~500 lines)
│   └── gpu_mode_simple_linear_executor/
│       └── ... (unchanged)
├── k_search/
│   └── modular/
│       ├── __init__.py          # Updated: removes run_search, LLMCall
│       ├── config.py            # Unchanged
│       ├── metrics/             # Unchanged
│       ├── artifacts/           # Unchanged
│       ├── adapters/            # Unchanged
│       └── world/               # Unchanged
│       # Deleted: loop.py, prompts.py
├── tests/
│   └── modular/
│       └── test_metrics.py      # Updated: only tracker tests remain
│       # Deleted: test_loop.py, test_e2e_search.py
└── # Deleted: run_search_v2.py
```

## run.py Contents

```python
#!/usr/bin/env python3
"""V2 Search Loop entry point - modular framework with metrics and artifacts."""

# Imports from k_search.modular (shared infrastructure)
from k_search.modular import SearchConfig, ArtifactConfig
from k_search.modular.artifacts import NoOpArtifactStore, create_artifact_stores
from k_search.modular.config import MetricsConfig, SearchResult, build_run_config
from k_search.modular.metrics import NoOpMetricsTracker, create_metrics_trackers
from k_search.modular.protocols import ArtifactStore, Evaluator, EvaluationResult, MetricsTracker
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.world.round import Round
from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

# Inlined functions:
# - _build_round_metrics() - from loop.py
# - run_search() - from loop.py
# - strip_markdown_fences() - from prompts.py
# - build_prompt() - from prompts.py
# - create_llm_call() - from run_search_v2.py
# - main() - CLI entry point
```

## Test Organization

**`scripts/gpu_mode_modular_v1/test_run.py`** contains:
- `TestRunSearch` - from test_loop.py
- `TestSearchConfig` - from test_loop.py
- `TestSearchResult` - from test_loop.py
- `TestBuildPrompt` - from test_loop.py
- `TestCreateImpl` - from test_loop.py
- `TestStripMarkdownFences` - from test_loop.py
- `TestBuildRoundMetrics` - from test_metrics.py
- `TestE2ESearch` - from test_e2e_search.py (GPU tests)

**Remaining in `tests/modular/test_metrics.py`:**
- `TestWandbMetricsTracker`
- `TestCreateMetricsTrackers`
- `TestLocalMetricsTracker`

## Runme Changes

`k_search_expr/runme.yaml` line 194:
```yaml
# Before:
"$VENV/bin/python" run_search_v2.py \

# After:
"$VENV/bin/python" scripts/gpu_mode_modular_v1/run.py \
```

## Implementation Notes

- All changes to K-Search repo happen in worktree `.claude/worktrees/modular-v1-refactor`
- Multiple implementation commits allowed, will be squashed before merge to main
- Runme changes happen in the main `verl-experiments-k-search` repo
