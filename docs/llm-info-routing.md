# LLM Info Routing

K-Search uses two separate LLM roles with distinct information needs. This document explains what information flows to each model and why.

## Two LLM Roles

```
                    ┌─────────────────────┐
                    │   Task Evaluation   │
                    │   (GPU execution)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌─────────────────────┐         ┌─────────────────────┐
    │    World Model      │         │   Codegen Model     │
    │  (action selection) │         │ (kernel generation) │
    └─────────────────────┘         └─────────────────────┘
              │                                 │
              │  Numeric metrics only           │  Full error logs
              │  (no logs)                      │  (for debugging)
              ▼                                 ▼
    ┌─────────────────────┐         ┌─────────────────────┐
    │  Select next action │         │   Generate kernel   │
    │  from solution tree │         │   improvements      │
    └─────────────────────┘         └─────────────────────┘
```

### World Model (Action Selection)

The world model decides which solution to expand and what action to take next. It receives:

- Solution tree context with numeric metrics
- `EvalResult.to_dict(include_log_excerpt=False)` — logs explicitly excluded

Why: The world model reasons about exploration vs exploitation. Verbose error logs would pollute the context without helping strategic decisions.

See: `world_model_manager.py:162`, `world_model_manager.py:221`, `world_model_manager.py:1757`

### Codegen Model (Kernel Generation)

The codegen model writes kernel improvements. It receives:

- Full error logs via `trace_logs` parameter
- `log_excerpt` from `EvalResult` through `_last_round_trace_logs_for_prompt`

Why: The codegen model needs detailed error messages, compiler output, and runtime failures to produce correct fixes.

See: `gpu_mode_task.py:159`, `gpu_mode_task.py:197-198`, `gpu_mode_task.py:420-423`

## Truncation Limits

Information gets truncated at multiple stages to prevent context overflow:

| Location | Limit | What gets truncated |
|----------|-------|---------------------|
| `run_eval.py:102` `_limit_length()` | 16KB | subprocess stdout/stderr from GPU execution |
| `evaluator.py:273-274` | 8KB | `log_excerpt` assembled from compile/run output |
| `task_base.py:36-38` `to_dict()` | 2KB strings, 800 chars log_excerpt | final `EvalResult` serialization |

### Data Flow

```
subprocess stdout/stderr (unbounded)
           │
           ▼
    _limit_length()  ─────────────────────────────  16KB cap
           │
           ▼
    log_excerpt assembly in evaluator  ───────────  8KB cap
           │
           ├───────────────────────────────────┐
           │                                   │
           ▼                                   ▼
    to_dict(include_log_excerpt=False)   trace_logs → codegen prompt
           │                                   │
           │  (log excluded)                   │  (full 8KB available)
           ▼                                   ▼
       World Model                       Codegen Model
```

## Design Rationale

1. **Separation of concerns**: Strategic decisions (world model) vs tactical fixes (codegen) need different information
2. **Context efficiency**: World model context stays lean; codegen gets debugging detail
3. **Progressive truncation**: Each stage applies limits appropriate to its role

The 8KB limit for codegen is typically sufficient for compiler errors and runtime tracebacks while preventing runaway context growth from chatty kernel output.
