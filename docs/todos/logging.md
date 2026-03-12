# Logging TODOs

Future logging improvements for K-Search.

## Performance Metrics

- Latency trends over rounds (min/max/mean across cycle)
- Speedup progression (vs baseline over time)
- Score distribution per action node

## World Model State

- Tree structure updates (node additions, state transitions)
- Node scores and selection probabilities
- Action success/failure rates

## Resource Tracking

- Token usage per LLM call
- Cost accumulation
- API latency breakdown

## Execution Timeline

- Wall-clock time per phase (codegen, eval, WM update)
- Bottleneck identification
- Parallelization opportunities
