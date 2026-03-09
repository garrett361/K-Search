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
