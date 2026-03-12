"""Result types for evaluation outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    """Result of correctness check."""

    passed: bool
    message: str = ""
    criteria: dict[str, Any] | None = None


@dataclass
class AnalysisResult:
    """Result of post-evaluation analysis."""

    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_artifact: bytes | None = None
    strategic_guidance: str | None = None
