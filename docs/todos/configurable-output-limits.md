# TODO: Configurable Output Limits

Currently, output truncation limits are hardcoded at multiple stages. These should be made configurable to allow tuning based on model context sizes and task complexity.

## Current Hardcoded Limits

| File | Line | Variable/Default | Purpose |
|------|------|------------------|---------|
| `k_search/tasks/gpu_mode/libkernelbot/run_eval.py` | 102 | `max_len=16384` | subprocess stdout/stderr |
| `k_search/tasks/gpu_mode/evaluator.py` | 273-274 | `[:8000]` | log_excerpt assembly |
| `k_search/tasks/task_base.py` | 37 | `max_str_chars=2000` | EvalResult string fields |
| `k_search/tasks/task_base.py` | 38 | `max_log_chars=800` | EvalResult log_excerpt |

## Suggested Configuration Approach

Add to the existing `SearchConfig` dataclass or create a separate `OutputLimitsConfig`:

```python
@dataclass
class OutputLimitsConfig:
    subprocess_output_bytes: int = 16384      # run_eval._limit_length
    log_excerpt_chars: int = 8000             # evaluator.py log assembly
    eval_result_str_chars: int = 2000         # to_dict max_str_chars
    eval_result_log_chars: int = 800          # to_dict max_log_chars
```

These limits can then be threaded through the call chain or accessed via a module-level config.

## Why This Matters

- Larger context models (100K+) could benefit from more verbose error logs
- Smaller/faster models may need tighter limits
- Different tasks may have different verbosity needs (compiler-heavy vs runtime-heavy)

## Priority

Low — current defaults work well for typical use cases.
