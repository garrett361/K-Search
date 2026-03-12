# Wandb/Artifact Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add wandb metrics logging and local artifact storage to V1 case search script.

**Architecture:** Port existing modular infrastructure (MetricsConfig, ArtifactConfig, factory functions) into V1SequentialExecutor. Log metrics per-round with cycle context. Update runme recipe to expose wandb parameters.

**Tech Stack:** Python, wandb, existing k_search.modular infrastructure

**Spec:** `docs/plans/2026-03-12-wandb-artifact-migration-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/gpu_mode_modular_k_search/run.py` | Modify | Add CLI args, wire up trackers/stores |
| `k_search_expr/runme.yaml` (repo root) | Modify | Add wandb parameters to recipe |

---

## Chunk 1: Script Modifications

### Task 1: Add Imports

**Files:**
- Modify: `scripts/gpu_mode_modular_k_search/run.py:18-48`

- [ ] **Step 1: Add config and factory imports**

Add after existing imports (around line 48):

```python
from k_search.modular.config import (
    ArtifactConfig,
    MetricsConfig,
    build_run_config,
)
from k_search.modular.metrics import create_metrics_trackers
from k_search.modular.artifacts import create_artifact_stores
from k_search.modular.protocols import ArtifactStore, MetricsTracker
```

- [ ] **Step 2: Verify imports work**

Run: `cd K-Search && python -c "from scripts.gpu_mode_modular_k_search.run import *"`
Expected: No import errors

---

### Task 2: Add CLI Arguments

**Files:**
- Modify: `scripts/gpu_mode_modular_k_search/run.py:964-1009` (in `main()`)

- [ ] **Step 1: Add wandb arguments after existing args**

Add after `--verbose` argument (around line 1007):

```python
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Wandb project name",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Wandb run name",
    )
    parser.add_argument(
        "--wandb-dir",
        default=None,
        help="Directory for wandb local files",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Wandb group name (e.g., experiment name)",
    )
    parser.add_argument(
        "--wandb-tags",
        default=None,
        help="Comma-separated wandb tags",
    )
    parser.add_argument(
        "--artifact-output-dir",
        default=None,
        help="Directory to save artifacts (code, metadata)",
    )
    parser.add_argument(
        "--artifact-mode",
        default="successes",
        choices=["successes", "all"],
        help="Which artifacts to store: 'successes' or 'all'",
    )
```

- [ ] **Step 2: Verify args parse**

Run: `cd K-Search && python scripts/gpu_mode_modular_k_search/run.py --help | grep -A1 wandb`
Expected: Shows `--wandb` and `--wandb-project` options

---

### Task 3: Modify V1SequentialExecutor Constructor

**Files:**
- Modify: `scripts/gpu_mode_modular_k_search/run.py:626-654` (V1SequentialExecutor.__init__)

- [ ] **Step 1: Add tracker/store parameters and state**

Update `__init__` to accept and store trackers/stores:

```python
class V1SequentialExecutor:
    """Executor implementing V1 case search with world model cycles.

    Operates entirely on modular Tree/Node/Cycle/Round structures.
    All metadata reads come from Node.annotations and Action.annotations.
    """

    def __init__(
        self,
        world_model: V1WorldModel,
        task: TaskDefinition,
        evaluator: Evaluator,
        llm: LLMCall,
        prompt_builder: V1PromptBuilder,
        tree: Tree,
        max_rounds: int,
        cycle_config: CycleConfig | None = None,
        metrics_trackers: list[MetricsTracker] | None = None,
        artifact_stores: list[ArtifactStore] | None = None,
    ):
        self.world_model = world_model
        self.task = task
        self.evaluator = evaluator
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.tree = tree
        self.max_rounds = max_rounds
        self.cycle_config = cycle_config or CycleConfig()
        self.metrics_trackers = metrics_trackers or []
        self.artifact_stores = artifact_stores or []
        self.global_best_round: Round | None = None
        self.global_best_score: float = -1.0
        self.global_round_idx: int = 0
```

---

### Task 4: Add Metrics Logging Helper

**Files:**
- Modify: `scripts/gpu_mode_modular_k_search/run.py` (add after `_get_perf_summary_lines`, around line 511)

- [ ] **Step 1: Add helper function to build round metrics**

```python
def _build_v1_round_metrics(
    round_time_secs: float,
    score: float,
    result: EvaluationResult,
    best_score: float,
    cycle_idx: int,
    cycle_round: int,
) -> dict[str, float | int | str | None]:
    """Build metrics dict for a V1 case search round."""
    metrics: dict[str, float | int | str | None] = {
        "round_time_secs": round_time_secs,
        "score": score,
        "succeeded": int(result.succeeded()),
        "best_score": best_score,
        "cycle_idx": cycle_idx,
        "cycle_round": cycle_round,
    }
    for key, val in result.get_metrics().items():
        if isinstance(val, (int, float, str)) and not isinstance(val, bool):
            metrics[key] = val
    return metrics
```

---

### Task 5: Integrate Logging in _run_cycle

**Files:**
- Modify: `scripts/gpu_mode_modular_k_search/run.py:733-913` (V1SequentialExecutor._run_cycle)

- [ ] **Step 1: Update _run_cycle signature to accept cycle_idx**

Change method signature:

```python
    def _run_cycle(
        self,
        node: Node,
        action_ctx: dict[str, Any],
        rounds_remaining: int,
        rounds_used: int,
        cycle_idx: int,
    ) -> Cycle:
```

- [ ] **Step 2: Add timing and logging after round creation**

After the `Round` object is created (after line 870), add:

```python
            import time as time_module
            round_elapsed = time_module.perf_counter() - round_start
```

Wait - the round already has timing from the prompt building. Let me check the code structure again. Actually, I need to add timing around the LLM call and eval. Add `round_start = time.perf_counter()` before the prompt building (around line 811), and after the Round is created, add the metrics logging:

```python
            # Log metrics
            round_metrics = _build_v1_round_metrics(
                round_time_secs=round_elapsed,
                score=score,
                result=result,
                best_score=best_score,
                cycle_idx=cycle_idx,
                cycle_round=attempt,
            )
            for tracker in self.metrics_trackers:
                tracker.log(round_metrics, step=self.global_round_idx)
            for store in self.artifact_stores:
                store.store(round_, self.global_round_idx)
            self.global_round_idx += 1
```

- [ ] **Step 3: Add round timing**

Add at start of attempt loop (after `for attempt in range(max_attempts):`, around line 764):

```python
            round_start = time.perf_counter()
```

Add after Round creation (after line 870):

```python
            round_elapsed = time.perf_counter() - round_start
```

- [ ] **Step 4: Update call site in run() to pass cycle_idx**

In `run()` method (around line 704), update the call:

```python
            cycle = self._run_cycle(
                node,
                action_ctx,
                rounds_remaining=self.max_rounds - rounds_used,
                rounds_used=rounds_used,
                cycle_idx=len([n for n in self.tree.nodes if n.status == "closed"]),
            )
```

Actually, simpler - track cycle_idx as a counter in run():

Before the while loop (around line 677), add:
```python
        cycle_idx = 0
```

Update the call:
```python
            cycle = self._run_cycle(
                node,
                action_ctx,
                rounds_remaining=self.max_rounds - rounds_used,
                rounds_used=rounds_used,
                cycle_idx=cycle_idx,
            )
```

After cycle completes, increment:
```python
            cycle_idx += 1
```

---

### Task 6: Wire Up in main()

**Files:**
- Modify: `scripts/gpu_mode_modular_k_search/run.py:1086-1102` (after cycle_config, before executor creation)

- [ ] **Step 1: Create configs and initialize wandb**

Add after `cycle_config` creation:

```python
    metrics_config = MetricsConfig(wandb=args.wandb, local=bool(args.artifact_output_dir))
    artifact_config = ArtifactConfig(
        output_dir=args.artifact_output_dir,
        only_store_successes=(args.artifact_mode == "successes"),
        wandb=args.wandb,
    )

    wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else None

    run_config = build_run_config(
        run_id=args.run_name or f"{args.task}-v1-{args.model_name.replace('/', '-')}-r{args.max_rounds}",
        model_name=args.model_name,
        reasoning_effort=args.reasoning_effort,
        search_config=SearchConfig(max_rounds=args.max_rounds),
        metrics_config=metrics_config,
        artifact_config=artifact_config,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        wandb_group=args.wandb_group,
        wandb_tags=wandb_tags,
        task=args.task,
        language=args.language,
        stagnation_rounds=args.stagnation_rounds,
        max_debug_improve_rounds=args.max_debug_improve_rounds,
        max_difficulty=args.max_difficulty,
    )

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            dir=args.wandb_dir,
            config=run_config,
            group=args.wandb_group,
            tags=wandb_tags,
        )
        logger.info(f"Wandb enabled: project={args.wandb_project}, run={args.run_name}")

    metrics_trackers = create_metrics_trackers(
        metrics_config,
        output_dir=args.artifact_output_dir,
        run_config=run_config,
    )
    artifact_stores = create_artifact_stores(artifact_config)
```

- [ ] **Step 2: Add SearchConfig import**

The `SearchConfig` import should already be added in Task 1. Verify it's there.

- [ ] **Step 3: Pass trackers/stores to executor**

Update executor creation:

```python
    executor = V1SequentialExecutor(
        world_model=world_model,
        task=task_def,
        evaluator=evaluator,
        llm=llm,
        prompt_builder=prompt_builder,
        tree=tree,
        max_rounds=args.max_rounds,
        cycle_config=cycle_config,
        metrics_trackers=metrics_trackers,
        artifact_stores=artifact_stores,
    )
```

- [ ] **Step 4: Verify script runs without wandb**

Run: `cd K-Search && python scripts/gpu_mode_modular_k_search/run.py --help`
Expected: Shows all new arguments

---

## Chunk 2: Runme Recipe Update

### Task 7: Update run_modular_k_search Recipe

**Files:**
- Modify: `k_search_expr/runme.yaml` (in repo root, not K-Search)

- [ ] **Step 1: Add new parameters**

Update the parameters line:

```yaml
run_modular_k_search:
  parameters: [k_search_path, task, language, max_rounds, reasoning_effort, max_debug_improve_rounds, stagnation_rounds, max_difficulty, run_name, model_name, wandb_project, wandb_group, wandb_tags]
```

- [ ] **Step 2: Add wandb/artifact args to script invocation**

Update the python invocation to include new args:

```bash
      "$VENV/bin/python" scripts/gpu_mode_modular_k_search/run.py \
        --task {task} \
        --language {language} \
        --model-name "{model_name}" \
        --api-key "$RITS_API_KEY" \
        --max-rounds {max_rounds} \
        --reasoning-effort {reasoning_effort} \
        --stagnation-rounds {stagnation_rounds} \
        --max-debug-improve-rounds {max_debug_improve_rounds} \
        --max-difficulty {max_difficulty} \
        --wandb \
        --wandb-project "{wandb_project}" \
        --wandb-group "{wandb_group}" \
        --wandb-tags "{wandb_tags}" \
        --artifact-output-dir "$ARTIFACTS_DIR" \
        --run-name "$RUN_NAME" \
        -v \
        2>&1 | tee "$LOG"
```

---

### Task 8: Final Commit

- [ ] **Step 1: Stage all changes**

```bash
git add K-Search/scripts/gpu_mode_modular_k_search/run.py k_search_expr/runme.yaml
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(modular): add wandb/artifact tracking to V1 case search

- Add CLI args: --wandb, --wandb-project, --run-name, --wandb-dir,
  --wandb-group, --wandb-tags, --artifact-output-dir, --artifact-mode
- Integrate MetricsConfig, ArtifactConfig, factory functions
- Log metrics per-round with cycle context
- Update run_modular_k_search recipe with wandb parameters"
```

---

## Chunk 3: Integration Test

### Task 9: Manual Integration Test (Post-Merge)

- [ ] **Step 1: Test without wandb**

Run a short test without wandb to verify artifacts are saved:

```bash
cd /proj/data-eng/goon/flim/verl-experiments-k-search
runme run-bash run_modular_k_search \
  k_search_path=K-Search \
  task=causal_conv1d \
  language=triton \
  max_rounds=2 \
  reasoning_effort=medium \
  max_debug_improve_rounds=2 \
  stagnation_rounds=2 \
  max_difficulty=4 \
  run_name=test-wandb-migration \
  model_name=openai/gpt-oss-120b \
  wandb_project=k-search-dev \
  wandb_group=test \
  wandb_tags=test,migration \
  --recipe-file k_search_expr/runme.yaml
```

Expected: Run completes, artifacts in `artifacts/test-wandb-migration/`, wandb run visible in k-search-dev project.

- [ ] **Step 2: Verify artifacts**

```bash
ls artifacts/test-wandb-migration/
```

Expected: `run.log` and potentially round artifacts if any succeeded.

- [ ] **Step 3: Clean up test artifacts**

```bash
rm -rf artifacts/test-wandb-migration
```
