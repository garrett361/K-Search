# Feed NCU Profiling Data Back to LLM

## Problem Statement

NCU (Nsight Compute) profiling captures detailed GPU performance metrics but this data is not fed back into the optimization loop. The LLM cannot see memory throughput, occupancy, pipe utilization, or other metrics that would directly guide optimization decisions.

## Current State

The `profile` mode in GPU tasks:

1. Runs kernel under `ncu --set full` to capture detailed metrics
2. Extracts key tables (GPU Throughput, Pipe Utilization, Warp State)
3. Base64-encodes the report into `benchmark.0.report`
4. Stores full `.ncu-rep` file for offline analysis

However, `log_excerpt` is **cleared for passing runs** (k_search/tasks/gpu_mode/evaluator.py:221-223):
```python
# Passed run: keep excerpt empty so prompts focus on perf only.
log_excerpt = ""
```

So the LLM never sees the profiling data - only aggregate timing.

## Proposal

Add an option to include NCU metrics in the feedback when profiling:

1. Parse the filtered NCU report for key metrics:
   - Memory throughput (% of peak)
   - Compute throughput (% of peak)
   - Occupancy
   - Top bottlenecks from Speed-of-Light analysis

2. Format as concise feedback for the LLM prompt:
   ```
   Profile: Memory BW 45% | Compute 78% | Occupancy 32%
   Bottleneck: Memory bound - L2 cache hit rate 23%
   ```

3. Include in `log_excerpt` even for passing runs when mode=profile

## Implementation Notes

Files to modify:
- `k_search/tasks/gpu_mode/evaluator.py:219-223` - Don't clear log_excerpt for profile mode
- `k_search/tasks/gpu_mode/libkernelbot/run_eval.py:484-500` - Extract structured metrics from NCU report
- `k_search/tasks/gpu_mode_task.py` - Add profile feedback to prompt construction

## Challenges

- NCU output format varies by GPU architecture
- Need to keep feedback concise (context window budget)
- May need to run profile mode separately from benchmark (NCU overhead)

## Priority

Medium - Would improve optimization guidance but current benchmark-only feedback still works.

## References

- `k_search/tasks/gpu_mode/libkernelbot/run_eval.py:452-511` - NCU profiling implementation
- `k_search/tasks/gpu_mode/libkernelbot/run_eval.py:145-188` - NCU report filtering
- `k_search/tasks/gpu_mode/evaluator.py:219-223` - Where log_excerpt is cleared
- `k_search/tasks/gpu_mode_task.py:179-180` - Where feedback is added to prompt
