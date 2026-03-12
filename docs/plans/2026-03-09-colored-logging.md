# Colored Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ANSI color codes to distinguish prompts (cyan) from LLM responses (green) in log output.

**Architecture:** Simple helper functions wrap text in ANSI codes. Call sites import and use these helpers.

**Tech Stack:** Python logging, ANSI escape codes

---

### Task 1: Create logging module with color helpers

**Files:**
- Create: `k_search/modular/logging.py`

```python
"""Logging utilities for colored output."""

CYAN = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"


def prompt_color(text: str) -> str:
    """Wrap text in cyan ANSI codes for prompt content."""
    return f"{CYAN}{text}{RESET}"


def response_color(text: str) -> str:
    """Wrap text in green ANSI codes for LLM response content."""
    return f"{GREEN}{text}{RESET}"
```

---

### Task 2: Color CODE_RESPONSE in sequential executor

**Files:**
- Modify: `k_search/modular/executors/sequential.py:88`

Add import:
```python
from k_search.modular.logging import response_color
```

Change line 88 from:
```python
logger.debug("[CODE_RESPONSE] (%d chars, ~%d toks):\n\n%s\n", len(code), len(code) // 4, code)
```

To:
```python
logger.debug(response_color(f"[CODE_RESPONSE] ({len(code)} chars, ~{len(code) // 4} toks):\n\n{code}\n"))
```

---

### Task 3: Color ACTION_RESPONSE in world model

**Files:**
- Modify: `k_search/modular/world_models/simple.py:44`

Add import:
```python
from k_search.modular.logging import response_color
```

Change line 44 from:
```python
logger.debug("[ACTION_RESPONSE] %s", raw_response.strip())
```

To:
```python
logger.debug(response_color(f"[ACTION_RESPONSE] {raw_response.strip()}"))
```

---

### Task 4: Color prompts in simple linear executor

**Files:**
- Modify: `scripts/gpu_mode_simple_linear_executor/run.py:72,137`

Add import:
```python
from k_search.modular.logging import prompt_color
```

Change line 72 from:
```python
logger.debug("[ACTION_PROMPT] (%d chars, ~%d toks):\n\n%s\n", len(prompt), len(prompt) // 4, prompt)
```

To:
```python
logger.debug(prompt_color(f"[ACTION_PROMPT] ({len(prompt)} chars, ~{len(prompt) // 4} toks):\n\n{prompt}\n"))
```

Change line 137 from:
```python
logger.debug("[CODE_PROMPT] (%d chars, ~%d toks):\n\n%s\n", len(prompt), len(prompt) // 4, prompt)
```

To:
```python
logger.debug(prompt_color(f"[CODE_PROMPT] ({len(prompt)} chars, ~{len(prompt) // 4} toks):\n\n{prompt}\n"))
```

---

### Task 5: Commit all changes

```bash
git add k_search/modular/logging.py \
        k_search/modular/executors/sequential.py \
        k_search/modular/world_models/simple.py \
        scripts/gpu_mode_simple_linear_executor/run.py
git commit -m "feat(modular): add colored logging for prompts and responses

Cyan for prompts, green for LLM responses. Helps distinguish
input/output when tailing logs with -v."
```

---

### Task 6: Manual verification

Run with verbose logging and verify colors render in terminal and when tailing log files.
