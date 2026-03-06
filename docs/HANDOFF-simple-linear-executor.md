# Handoff: SimpleWorldModel + SequentialExecutor

**For Claude Code session in `verl-experiments-k-search` repo.**

## What Was Implemented

A new execution path in K-Search that validates the WorldModel + Executor protocol pattern:

```
K-Search/
├── k_search/modular/
│   ├── protocols/executor.py      # Executor protocol
│   ├── world_models/simple.py     # SimpleWorldModel
│   └── executors/sequential.py    # SequentialExecutor
└── scripts/gpu_mode_simple_linear_executor/
    ├── run.py                     # Entry point script
    └── test_run.py                # Tests for prompt functions
```

## How It Works

**Flow per round:**
- Round 0: Generic initial action ("Write an optimized implementation.") → LLM generates code → evaluate
- Round 1+: LLM proposes specific action (based on feedback) → LLM generates code from action → evaluate

**Key difference from `run_search_v2.py`:**
- Uses two-step LLM pattern (action proposal + code generation) vs single code generation
- Uses Tree/Node structure instead of flat Round tracking
- Reference implementation for future parallel/branching executors

## CLI Args (matches run_search_v2.py)

```bash
python K-Search/scripts/gpu_mode_simple_linear_executor/run.py \
  --task causal_conv1d \
  --language triton \
  --max-rounds 5 \
  --model-name "$RITS_MODEL_NAME" \
  --base-url "$RITS_BASE_URL" \
  --api-key "$RITS_API_KEY"
```

## Suggested Runme Recipe

Add to `k_search_expr/runme.yaml`:

```yaml
run_simple_linear_executor:
  parameters: [task, language, max_rounds]
  code:
    - method: recipe
      code: [validate_api_env]
    - |
      MODEL_SAFE=$(echo "$RITS_MODEL_NAME" | tr '/' '-')
      RUN_NAME="{task}-simple-linear-$MODEL_SAFE-{language}-r{max_rounds}-$(date +%Y%m%d_%H%M%S)"
      ARTIFACTS_DIR="./artifacts/$RUN_NAME"
      mkdir -p "$ARTIFACTS_DIR"
      LOG="$ARTIFACTS_DIR/run.log"
      echo "Run: $RUN_NAME"
      echo "Artifacts: $ARTIFACTS_DIR"

      cd K-Search
      python scripts/gpu_mode_simple_linear_executor/run.py \
        --task {task} \
        --language {language} \
        --max-rounds {max_rounds} \
        --model-name "$RITS_MODEL_NAME" \
        --base-url "$RITS_BASE_URL" \
        --api-key "$RITS_API_KEY" \
        2>&1 | tee "../$LOG"

      echo ""
      echo "=== Run Complete ==="
      echo "Artifacts: $ARTIFACTS_DIR"
      echo "Log: $LOG"

run_simple_linear_e2e:
  parameters: []
  code:
    - method: recipe
      code:
        - run_simple_linear_executor
        - task=causal_conv1d
        - language=triton
        - max_rounds=3
```

## Run Command

From `verl-experiments-k-search` root:

```bash
runme run-bash run_simple_linear_e2e --recipe-file k_search_expr/runme.yaml
```

Or with custom params:

```bash
runme run-bash run_simple_linear_executor --recipe-file k_search_expr/runme.yaml task=trimul max_rounds=5
```

## Environment Required

Same as other K-Search runs:
- `RITS_API_KEY`
- `RITS_BASE_URL`
- `RITS_MODEL_NAME`

## Notes

- Script uses standard `chat.completions` API (not reasoning API)
- No wandb integration yet (add if needed)
- No artifact storage yet (uses NoOp stores)
- The script is at `K-Search/scripts/gpu_mode_simple_linear_executor/run.py`

## What to Test

1. Basic e2e run with 1-2 rounds
2. Verify action proposals appear in logs
3. Verify code generation uses the proposed action
4. Check that feedback from round N appears in round N+1 action prompt
