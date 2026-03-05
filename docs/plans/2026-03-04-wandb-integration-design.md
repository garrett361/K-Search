# Wandb Integration Design

Observability for V2 search loop: scalar metrics tracking and artifact persistence.

## Overview

Two independent concerns, implemented in separate PRs:

1. **Scalar Metrics** (`MetricsTracker`) - per-round numerical metrics to wandb
2. **Artifact Storage** (`ArtifactStore`) - code + metadata persistence to local/wandb

Both use protocol injection with no-op defaults. Wandb is opt-in via config flag.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         run_search()                                 │
│                                                                      │
│  for round in range(max_rounds):                                     │
│      prompt = build_prompt(...)                                      │
│      code = llm(prompt)                                              │
│      result = evaluator.evaluate(impl)                               │
│      outcome = EvalOutcome(impl, result)                             │
│                                                                      │
│      for tracker in metrics_trackers:                                │
│          tracker.log({...}, step=round)       ◄── SCALAR METRICS     │
│      for store in artifact_stores:                                   │
│          store.store(outcome, round)          ◄── ARTIFACT STORE     │
└─────────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌───────────────────────┐    ┌───────────────────────┐
│ MetricsTracker        │    │ ArtifactStore         │
│ (Protocol)            │    │ (Protocol)            │
│                       │    │                       │
│ log(metrics, step)    │    │ store(outcome, round) │
│                       │    │                       │
│ Implementations:      │    │ Implementations:      │
│ - WandbMetricsTracker │    │ - LocalArtifactStore  │
│ - NoOpMetricsTracker  │    │ - WandbArtifactStore  │
└───────────────────────┘    │ - NoOpArtifactStore   │
                             └───────────────────────┘
```

## Protocols

### MetricsTracker

```python
class MetricsTracker(Protocol):
    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None: ...
```

### ArtifactStore

```python
class ArtifactStore(Protocol):
    def store(self, outcome: EvalOutcome, round_idx: int) -> None: ...
```

`EvalOutcome` already contains:
- `impl: Implementation` → code content via `impl.content`
- `result: EvaluationResult` → metrics via `result.get_metrics()`, status via `result.is_success()`

## Configuration

### SearchConfig (unchanged)

```python
@dataclass
class SearchConfig:
    max_rounds: int = 10
    timeout_secs: int | None = None
```

### MetricsConfig (new)

```python
@dataclass
class MetricsConfig:
    chars_per_token: int = 4
    wandb: bool = False
```

### ArtifactConfig (new)

```python
@dataclass
class ArtifactConfig:
    output_dir: Path | str | None = None
    only_store_successes: bool = True
    wandb: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
```

## run_search Signature

```python
def run_search(
    task: TaskDefinition,
    evaluator: Evaluator,
    llm: LLMCall,
    config: SearchConfig,
    metrics_trackers: MetricsTracker | list[MetricsTracker] | None = None,
    artifact_stores: ArtifactStore | list[ArtifactStore] | None = None,
) -> SearchResult:
    # Normalize to lists
    if metrics_trackers is None:
        metrics_trackers = [NoOpMetricsTracker()]
    elif not isinstance(metrics_trackers, list):
        metrics_trackers = [metrics_trackers]

    if artifact_stores is None:
        artifact_stores = [NoOpArtifactStore()]
    elif not isinstance(artifact_stores, list):
        artifact_stores = [artifact_stores]

    ...

    # In loop:
    for tracker in metrics_trackers:
        tracker.log(metrics, step=round_idx)

    for store in artifact_stores:
        store.store(outcome, round_idx)
```

## Metrics Schema

Per-round (step=round_idx), all cumulative where applicable:

| Metric | Source |
|--------|--------|
| `round_time_secs` | this round's duration |
| `score` | `task.scorer.score(result)` |
| `is_success` | `result.is_success()` as 0/1 |
| `best_score` | running best |
| `prompt_tokens_est` | cumulative |
| `completion_tokens_est` | cumulative |
| `total_tokens_est` | cumulative (prompt + completion) |
| `*` | all numerical values from `result.get_metrics()` |

Token estimation: `len(text) // config.chars_per_token`

### Metrics Building

```python
def _build_round_metrics(
    round_time_secs: float,
    score: float,
    result: EvaluationResult,
    best_score: float,
    cumulative_prompt_tokens: int,
    cumulative_completion_tokens: int,
) -> dict[str, float | int]:
    metrics = {
        "round_time_secs": round_time_secs,
        "score": score,
        "is_success": int(result.is_success()),
        "best_score": best_score,
        "prompt_tokens_est": cumulative_prompt_tokens,
        "completion_tokens_est": cumulative_completion_tokens,
        "total_tokens_est": cumulative_prompt_tokens + cumulative_completion_tokens,
    }

    # Merge all numerical values from result
    for key, val in result.get_metrics().items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            metrics[key] = val

    return metrics
```

## Artifact Storage

### Local Storage Structure

```
{output_dir}/
└── round_{idx}/
    ├── solution.py        # outcome.impl.content
    └── metadata.json      # { name, is_success, ...get_metrics() }
```

### Storage Behavior

- `only_store_successes=True` (default): only store when `result.is_success()`
- `only_store_successes=False`: store every round

## Factory Functions

```python
def create_metrics_trackers(config: MetricsConfig | None = None) -> list[MetricsTracker]:
    config = config or MetricsConfig()
    if config.wandb:
        return [WandbMetricsTracker(config)]
    return [NoOpMetricsTracker()]

def create_artifact_stores(config: ArtifactConfig | None = None) -> list[ArtifactStore]:
    config = config or ArtifactConfig()
    stores: list[ArtifactStore] = []

    if config.output_dir:
        stores.append(LocalArtifactStore(config))

    if config.wandb:
        stores.append(WandbArtifactStore(config))

    return stores or [NoOpArtifactStore()]
```

## Wandb Error Handling

If `config.wandb=True` but wandb unavailable or no active run, raise error:

```python
class WandbMetricsTracker:
    def __init__(self, config: MetricsConfig):
        try:
            import wandb
        except ImportError:
            raise RuntimeError("wandb configured but not installed")

        if wandb.run is None:
            raise RuntimeError("wandb configured but no active run (call wandb.init() first)")
```

Same pattern for `WandbArtifactStore`.

## Module Structure

```
k_search/search_v2/
├── __init__.py
├── config.py              # SearchConfig, MetricsConfig, ArtifactConfig
├── loop.py                # run_search()
├── prompts.py
│
├── metrics/
│   ├── __init__.py
│   ├── protocol.py        # MetricsTracker protocol
│   ├── noop.py            # NoOpMetricsTracker
│   └── wandb.py           # WandbMetricsTracker
│
└── artifacts/
    ├── __init__.py
    ├── protocol.py        # ArtifactStore protocol
    ├── noop.py            # NoOpArtifactStore
    ├── local.py           # LocalArtifactStore
    └── wandb.py           # WandbArtifactStore
```

## Implementation Order

### PR1: Scalar Metrics

1. Create `metrics/` subpackage with protocol and implementations
2. Add `MetricsConfig` to `config.py`
3. Update `loop.py` to accept `metrics_trackers` param
4. Add `create_metrics_trackers()` factory
5. Tests

### PR2: Artifact Store

1. Create `artifacts/` subpackage with protocol and implementations
2. Add `ArtifactConfig` to `config.py`
3. Update `loop.py` to accept `artifact_stores` param
4. Add `create_artifact_stores()` factory
5. Tests

## Usage Example

```python
from k_search.search_v2 import run_search, SearchConfig
from k_search.search_v2.config import MetricsConfig, ArtifactConfig
from k_search.search_v2.metrics import create_metrics_trackers
from k_search.search_v2.artifacts import create_artifact_stores

# With wandb
import wandb
wandb.init(project="k-search", name="my-run")

metrics_trackers = create_metrics_trackers(MetricsConfig(wandb=True))
artifact_stores = create_artifact_stores(ArtifactConfig(
    output_dir="artifacts",
    wandb=True,
))

result = run_search(
    task=task,
    evaluator=evaluator,
    llm=llm,
    config=SearchConfig(max_rounds=10),
    metrics_trackers=metrics_trackers,
    artifact_stores=artifact_stores,
)
```

## References

- V1 wandb usage: `k_search/kernel_generators/kernel_generator_world_model.py`
- Extensions doc: `docs/plans/2026-03-04-task-framework-extensions.md` (§8, §9)
- V2 loop: `k_search/search_v2/loop.py`
