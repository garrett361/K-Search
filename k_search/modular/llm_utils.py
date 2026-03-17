"""Utilities for LLM input/output processing."""

import re


def strip_markdown_fences(code: str | None) -> str | None:
    """Strip markdown code fences from LLM output.

    Args:
        code: Raw code string, possibly wrapped in ```python ... ```

    Returns:
        Code with markdown fences removed
    """
    if not code or "```" not in code:
        return code

    m = re.search(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)\n```", code)
    if m:
        return (m.group(1) or "").strip()

    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        code = "\n".join(lines)

    if code.endswith("```"):
        lines = code.split("\n")
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code.replace("```", "").strip()
