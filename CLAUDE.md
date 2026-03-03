# K-Search

LLM-driven GPU kernel optimization with co-evolving world model.
Entry point: `generate_kernels_and_eval.py`

## Architecture

```
k_search/
├── kernel_generators/  # LLM generation + world model
├── tasks/              # Task backends
│   ├── task_base.py    # Task protocol, Solution, EvalResult
│   ├── flashinfer_bench_task.py
│   └── gpu_mode/       # GPU Mode competition tasks
└── utils/              # Paths, solution persistence
```

Key types: `Solution` (source container), `EvalResult` (status, latency_ms, score()), `Task` (protocol)

## GPU Mode Task Structure

Each task in `k_search/tasks/gpu_mode/<task>/`:
- `task.py` - GpuModeTask subclass
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

## Style

- Favor descriptive variable and function names over comments
- Comment only the *why*, never the *what*
- No decorative comment headers or section separators

## Documentation

Implementation details and design decisions are documented in `docs/` with descriptive filenames.
