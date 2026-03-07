# GPU Mode: Honor --rtol/--atol CLI Flags

## Problem

GPU mode tasks (`trimul`, `causal_conv1d`) ignore the `--rtol`/`--atol` CLI flags and use hardcoded tolerances of `2e-2` in each task's `reference.py`. The flashinfer backend respects these flags.

## Current State

```python
# k_search/tasks/gpu_mode/trimul/reference.py:123
check_implementation = make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)

# k_search/tasks/gpu_mode/causal_conv1d/reference.py:87
check_implementation = make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)
```

## Fix

Thread rtol/atol from CLI through the evaluation pipeline:

1. `generate_kernels_and_eval.py` already parses `--rtol`/`--atol`
2. Pass to `GpuModeTask` constructor or `evaluate()` method
3. Forward to `eval.py` which imports `check_implementation`
4. Make `check_implementation` accept rtol/atol or generate it dynamically

Alternative: define tolerances in task config rather than hardcoding in `reference.py`.
