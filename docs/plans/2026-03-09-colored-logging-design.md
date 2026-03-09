# Colored Logging for Prompts and Responses

Adds ANSI color codes to distinguish prompts (cyan) from LLM responses (green) in log output.

## Decision Summary

- **ANSI codes embedded in output**: Works with `tail -f`, colors show in terminal
- **Colors**: Cyan for prompts, green for responses
- **Scope**: Only `[*_PROMPT]` and `[*_RESPONSE]` DEBUG messages

## API

New module `k_search/modular/logging.py`:

```python
def prompt_color(text: str) -> str:
    """Wrap text in cyan ANSI codes."""

def response_color(text: str) -> str:
    """Wrap text in green ANSI codes."""
```

## Call Sites

| File | Line | Type |
|------|------|------|
| `k_search/modular/executors/sequential.py` | 88 | `[CODE_RESPONSE]` |
| `k_search/modular/world_models/simple.py` | 44 | `[ACTION_RESPONSE]` |
| `scripts/gpu_mode_simple_linear_executor/run.py` | 72 | `[ACTION_PROMPT]` |
| `scripts/gpu_mode_simple_linear_executor/run.py` | 137 | `[CODE_PROMPT]` |

## Usage

When running with `-v` (DEBUG level), prompts appear in cyan and responses in green. Works when tailing log files: `tail -f run.log`.

## Conventions

Logging conventions added to `CLAUDE.md` for future development.
