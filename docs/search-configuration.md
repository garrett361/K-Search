# Search Configuration

Configuration options for K-Search kernel optimization runs.

## CLI Options

### Core Settings

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | (required) | LLM model name (e.g., `gpt-4.1`, `gemini-2.5-pro`) |
| `--definition` | (required) | Task/kernel name to optimize |
| `--task-source` | `flashinfer` | Task backend: `flashinfer` or `gpumode` |
| `--language` | `triton` | Target language: `triton`, `cuda`, or `python` |
| `--target-gpu` | `H100` | GPU architecture hint for prompts |
| `--max-opt-rounds` | `5` | Maximum optimization rounds per run |

### World Model Options

| Flag | Default | Description |
|------|---------|-------------|
| `--world-model` | off | Enable world-model prompting |
| `--wm-max-difficulty` | `4` | Max action difficulty (1-5) for selection |
| `--wm-stagnation-window` | `5` | End cycle after N non-improving rounds |
| `--continue-from-world-model` | - | Resume from JSON file (`auto` for default path) |

### Evaluation Settings

| Flag | Default | Description |
|------|---------|-------------|
| `--warmup-runs` | `10` | Benchmark warmup iterations |
| `--iterations` | `10` | Benchmark timing iterations |
| `--num-trials` | `1` | Number of benchmark trials |
| `--rtol` | `1e-2` | Relative tolerance (flashinfer only) |
| `--atol` | `1e-2` | Absolute tolerance (flashinfer only) |
| `--num-eval-workload` | all | Limit workloads per definition |
| `--baseline-solution` | - | Solution name for vs_baseline comparison |

**Note:** GPU mode tasks ignore `--rtol`/`--atol` and use hardcoded `2e-2` tolerances defined in each task's `reference.py`.

### Output & Persistence

| Flag | Default | Description |
|------|---------|-------------|
| `--artifacts-dir` | `.ksearch` | Base directory for artifacts |
| `--save-solutions` | off | Persist solutions to artifacts dir |
| `--no-save-results` | off | Skip writing traces to dataset |
| `--continue-from-solution` | - | Resume from existing solution name |

### LLM API

| Flag | Default | Description |
|------|---------|-------------|
| `--api-key` | `$LLM_API_KEY` | API key for LLM provider |
| `--base-url` | - | OpenAI-compatible endpoint URL |
| `--no-reasoning-api` | off | Use chat completions instead of reasoning API |

## Programmatic Configuration

### WorldModelSelectionPolicy

Controls which action node to execute next.

```python
from k_search.kernel_generators.world_model_manager import WorldModelSelectionPolicy

policy = WorldModelSelectionPolicy(
    # Difficulty gating
    max_difficulty_1_to_5=4,              # Filter actions above this
    relax_difficulty_if_best_vs_base_ge=0.5,  # Relax threshold
    relaxed_max_difficulty_1_to_5=4,      # Max after relaxation

    # Utility weights (higher = more important)
    w_score=3.0,           # action.score_0_to_1
    w_difficulty=2.5,      # Prefer easier actions
    w_depth=0.75,          # Prefer shallower nodes
    w_parent_quality=1.5,  # Parent node quality
    w_overall_rating=0.5,  # Node overall_rating_0_to_10
    w_confidence=0.25,     # Node confidence_0_to_1
    w_root_explore=0.15,   # Bias for root-level exploration
)
```

### WorldModelConfig

Controls world model behavior.

```python
from k_search.kernel_generators.world_model_manager import WorldModelConfig

config = WorldModelConfig(
    enabled=True,                    # Enable world model
    max_chars_per_block=6000,        # Prompt truncation limit
    max_new_nodes_per_edit=3,        # Cap new nodes per refine()
    selection_policy=policy,         # Selection policy instance
)
```

### WorldModelKernelGeneratorWithBaseline

Full generator configuration.

```python
from k_search.kernel_generators.kernel_generator_world_model import (
    WorldModelKernelGeneratorWithBaseline
)

generator = WorldModelKernelGeneratorWithBaseline(
    model_name="gpt-4.1",
    language="triton",
    target_gpu="H100",
    api_key="...",
    base_url=None,                   # Optional: custom endpoint
    reasoning_effort="medium",       # low/medium/high
    use_reasoning_api=True,
    enable_world_model=True,
    world_model_max_chars=50000,     # WM JSON size limit
    artifacts_dir=".ksearch",
    wm_max_difficulty=4,             # Override policy max difficulty
)
```

## Selection Behavior

The selection algorithm:

1. **Collect frontier** - OPEN action nodes whose parent is SOLVED (or root)
2. **Filter by difficulty** - Remove actions above `max_difficulty_1_to_5`
   - Relaxes to `relaxed_max_difficulty_1_to_5` if best `vs_baseline >= relax_threshold`
3. **Rank by utility** - Sort by `(score, difficulty, rating, node_id)`
4. **Select top** - Execute highest-ranked action

### Difficulty Scale

| Value | Meaning |
|-------|---------|
| 1 | Trivial change (config tweak, constant) |
| 2 | Simple optimization (vectorize, unroll) |
| 3 | Moderate refactor (tiling, staging) |
| 4 | Significant rewrite (algorithm change) |
| 5 | Major redesign (new kernel family) |

Lower difficulty actions are preferred early; harder actions unlock as performance improves.
