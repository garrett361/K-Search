# Wandb/Artifact Migration for V1 Case Search

**Date:** 2026-03-12
**Status:** Approved
**Scope:** Port wandb and artifact tracking from `gpu_mode_modular_v1/run.py` to `gpu_mode_modular_k_search/run.py`

## Summary

Add wandb metrics logging and local artifact storage to the V1 case search script (`gpu_mode_modular_k_search/run.py`), using the existing modular infrastructure. Update the runme recipe to expose these options.

## Background

- `gpu_mode_modular_v1/run.py` (simpler V2 loop) has full wandb/artifact support
- `gpu_mode_modular_k_search/run.py` (V1 case search with world model) lacks this tracking
- The modular infrastructure (`MetricsConfig`, `ArtifactConfig`, factory functions) already exists in `k_search/modular/`

## Design

### New CLI Arguments

Add to `gpu_mode_modular_k_search/run.py`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--wandb` | flag | False | Enable wandb logging |
| `--wandb-project` | str | None | Wandb project name |
| `--run-name` | str | None | Wandb run name |
| `--wandb-dir` | str | None | Local wandb files directory |
| `--wandb-group` | str | None | Wandb group (experiment name) |
| `--wandb-tags` | str | None | Comma-separated tags |
| `--artifact-output-dir` | str | None | Local artifact directory |
| `--artifact-mode` | str | "successes" | Which artifacts to store: "successes" or "all" |

### Imports

Add to `gpu_mode_modular_k_search/run.py`:

```python
from k_search.modular.config import SearchConfig, ArtifactConfig, MetricsConfig, build_run_config
from k_search.modular.metrics import create_metrics_trackers
from k_search.modular.artifacts import create_artifact_stores
```

### V1SequentialExecutor Modifications

1. **Constructor** - Accept `metrics_trackers` and `artifact_stores` parameters
2. **State tracking** - Add `global_round_idx` counter incremented across all cycles
3. **Integration in `_run_cycle()`** - After each round:
   - Build metrics dict with round data
   - Call `tracker.log(metrics, step=global_round_idx)` for each tracker
   - Call `store.store(round_, global_round_idx)` for each store

### Metrics Logged Per Round

```python
{
    "round_time_secs": float,
    "score": float,
    "succeeded": int,  # 0 or 1
    "best_score": float,
    "cycle_idx": int,
    "cycle_round": int,
    # From EvaluationResult.get_metrics():
    "latency_ms": float | None,
    "speedup_factor": float | None,
    "status": str,  # "passed" or "failed"
}
```

Note: Token tracking is deferred until accurate token counting is implemented.

### main() Changes

1. Parse new arguments
2. Create `MetricsConfig` and `ArtifactConfig` from args
3. Build `run_config` using `build_run_config()`
4. Initialize wandb if `--wandb` flag set
5. Create trackers/stores using factory functions
6. Pass trackers/stores to `V1SequentialExecutor`

### Runme Recipe Updates

Update `run_modular_k_search` in `k_search_expr/runme.yaml`:

**New parameters:**
- `wandb_project`
- `wandb_group`
- `wandb_tags`

**Script invocation additions:**
```bash
--wandb \
--wandb-project {wandb_project} \
--wandb-group "{wandb_group}" \
--wandb-tags "{wandb_tags}" \
--artifact-output-dir "$ARTIFACTS_DIR" \
--run-name "$RUN_NAME"
```

Note: `$ARTIFACTS_DIR` is already created by the recipe. `$RUN_NAME` is already computed.

## Files Modified

1. `K-Search/scripts/gpu_mode_modular_k_search/run.py`
2. `k_search_expr/runme.yaml`

## Testing

- Run with `--wandb` disabled: verify no errors, artifacts saved locally
- Run with `--wandb` enabled: verify metrics appear in wandb dashboard
- Verify runme recipe works with new parameters
