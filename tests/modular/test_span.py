"""Tests for Span class."""

import time
from unittest.mock import MagicMock

from k_search.modular.span import Span


def test_span_get_metrics_delegates_to_timer():
    node = MagicMock()
    span = Span(node=node)

    span.timer.start()
    with span.timer["llm"]:
        time.sleep(0.01)
    span.timer.stop()

    metrics = span.get_metrics()
    assert "total" in metrics
    assert "llm" in metrics
    assert metrics["llm"] >= 0.01
