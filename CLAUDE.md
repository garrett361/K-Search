# K-Search

LLM-driven GPU kernel optimization with co-evolving world model.
Entry points:
- `run_search_v2.py` - Modular search loop
- `generate_kernels_and_eval.py` - V1 kernel generation

## Architecture

```
k_search/
├── modular/            # Protocol-based search framework
│   ├── protocols/      # Protocol definitions (EvaluationResult, Implementation, etc.)
│   ├── adapters/       # Task-specific adapters (gpu_mode/)
│   ├── metrics/        # Metrics tracking (noop, wandb)
│   ├── artifacts/      # Artifact storage (local, wandb)
│   ├── loop.py         # Search loop (run_search)
│   ├── round.py        # Round container
│   └── config.py       # SearchConfig, MetricsConfig, ArtifactConfig
├── kernel_generators/  # LLM generation + world model
├── tasks/              # Task backends
│   ├── task_base.py    # Task protocol, Solution, EvalResult
│   ├── flashinfer_bench_task.py
│   └── gpu_mode/       # GPU Mode competition tasks
└── utils/              # Paths, solution persistence
```

Key types: `Round` (iteration container), `EvaluationResult` (protocol), `Implementation` (protocol), `Solution` (v1 source container), `EvalResult` (v1 status)

## GPU Mode Task Structure

Each task in `k_search/tasks/gpu_mode/<task>/`:
- `task.py` - GpuModeTriMulTask subclass
- `reference.py` - generate_input(), ref_kernel(), check_implementation()
- `submission.py` - Baseline custom_kernel(data)
- `spec.py` - SPEC_TEXT_TRITON/CUDA problem description

Reference implementations use pure PyTorch.

## Development

```bash
uv pip install -e ".[dev]"   # Editable install with dev deps
```

## Code Quality

```bash
ruff check path/to/modified/files/
ruff format path/to/modified/files/
ty check path/to/modified/files/
pytest tests/                 # All tests
pytest -m cuda tests/         # GPU tests only
```

GPU tests use `@pytest.mark.cuda`.

## Commits

Use `(v1)` scope for changes to the original V1 code paths:
- `k_search/kernel_generators/`
- `k_search/tasks/` (except `modular/adapters/`)
- `generate_kernels_and_eval.py`

Example: `fix(v1): remove deprecated use_reasoning_api parameter`

The `modular/` subsystem uses standard scopes.

## Style

- Favor descriptive variable and function names over comments
- Comment only the *why*, never the *what*
- No decorative comment headers or section separators

## Worktrees

Always create worktrees under `.claude/worktrees/`, not `worktrees/`.

## Documentation

Implementation details and design decisions are documented in `docs/` with descriptive filenames.
