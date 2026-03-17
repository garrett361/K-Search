"""ParseResult for fallible operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ParseResult(Generic[T]):
    """Result of parsing/applying a tool call."""

    success: bool
    value: T | None = None
    error: str | None = None

    @classmethod
    def ok(cls, value: T) -> ParseResult[T]:
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error: str) -> ParseResult[T]:
        return cls(success=False, error=error)
