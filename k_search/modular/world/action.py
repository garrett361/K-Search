"""Action dataclass for search proposals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Action:
    """Proposal for what to try next."""

    title: str
