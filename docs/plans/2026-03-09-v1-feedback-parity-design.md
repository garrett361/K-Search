# V1-Style Feedback Parity for Simple Linear Executor

## Goal

Add V1-style feedback to code generation prompts in `scripts/gpu_mode_simple_linear_executor/run.py`.

## V1 Prompt Structure

V1 optimization prompts include these sections:
- "Current Implementation Status:" - trace_logs (error/compilation output)
- "Current Implementation:" - the code that was tried
- "Last Round Summary:" - pass/fail status, latency, score
- "Current Best Solution So Far:" - best code + metrics (only if different from last)

## Changes

### 1. Enhance `_FeedbackProvider`

Location: `k_search/modular/adapters/gpu_mode/task_definition.py`

Combined format for action prompts (status + code + logs):

```python
class _FeedbackProvider:
    def for_codegen(self, rounds: Round | list[Round]) -> str:
        # Returns: "Status: ... | Latency: ... | Score: ...\n\nCode:\n...\n\nLogs:\n..."
```

### 2. Update `create_code_prompt_fn` in run.py

V1-style separate sections for code prompts:

```python
def create_code_prompt_fn(task_def, tree):
    def code_prompt_fn(node: Node, task: TaskDefinition) -> str:
        prompt = task_def.get_prompt_text()

        if node.action:
            prompt += f"\n\nAction: {node.action.title}"

        # Last round - V1-style separate sections
        last_node = _get_last_evaluated_node()
        if last_node and last_node.cycle:
            last_round = last_node.cycle.rounds[-1]
            prompt += f"\n\nCurrent Implementation Status:\n{logs}"
            prompt += f"\n\nCurrent Implementation:\n{code}"
            prompt += f"\n\nLast Round Summary:\n{summary}"

        # Best so far (only if different from last)
        best_node = tree.get_best_node()
        if best_node and best_node._id != last_node._id:
            prompt += f"\n\nCurrent Best Solution So Far:\n{summary}\n\nCode:\n{code}"

        prompt += "\n\nGenerate the corrected and optimized implementation:"
        return prompt
```

## Files Changed

- `k_search/modular/adapters/gpu_mode/task_definition.py` - enhance `_FeedbackProvider.for_codegen()`
- `k_search/modular/executors/sequential.py` - add char count to debug log
- `scripts/gpu_mode_simple_linear_executor/run.py` - V1-style code prompt with separate sections
- `scripts/gpu_mode_simple_linear_executor/test_run.py` - update tests for new signature

## Not in Scope

- Protocol changes
- New abstractions (SearchContext, PromptBuilder)
- Executor signature changes
