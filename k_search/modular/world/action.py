"""Action dataclass for search proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Action:
    """Proposal for what to try next."""

    title: str
    annotations: dict[str, Any] | None = None
