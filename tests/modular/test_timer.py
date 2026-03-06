"""Tests for Timer class."""

import time

from k_search.modular.timer import Timer


def test_timer_total_secs_before_start_is_zero():
    timer = Timer()
    assert timer.total_secs == 0.0


def test_timer_start_stop_captures_elapsed_time():
    timer = Timer()
    timer.start()
    time.sleep(0.01)
    timer.stop()

    assert timer.total_secs >= 0.01
    assert timer.total_secs < 0.1  # sanity bound


def test_timer_single_category_tracking():
    timer = Timer()
    timer.start()

    with timer["llm"]:
        time.sleep(0.01)

    timer.stop()

    timing = timer.get_timing_secs()
    assert "llm" in timing
    assert timing["llm"] >= 0.01
    assert timing["total"] >= timing["llm"]
