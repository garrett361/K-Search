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


def test_timer_multi_tag_tracking():
    timer = Timer()
    timer.start()

    with timer["llm", "codegen"]:
        time.sleep(0.01)

    timer.stop()

    timing = timer.get_timing_secs()
    assert timing["llm"] >= 0.01
    assert timing["codegen"] >= 0.01
    assert timing["llm"] == timing["codegen"]  # same elapsed time for both


def test_timer_overhead_calculation():
    timer = Timer()
    timer.start()

    time.sleep(0.01)  # overhead
    with timer["work"]:
        time.sleep(0.01)

    timer.stop()

    timing = timer.get_timing_secs()
    assert "overhead" in timing
    assert timing["overhead"] >= 0.01
    assert timing["total"] >= timing["work"] + timing["overhead"]


def test_timer_no_overhead_when_no_tags():
    timer = Timer()
    timer.start()
    time.sleep(0.01)
    timer.stop()

    timing = timer.get_timing_secs()
    assert "overhead" not in timing
    assert timing["total"] >= 0.01


def test_timer_rejects_non_string_tag():
    import pytest

    timer = Timer()
    with pytest.raises(TypeError, match="tag must be str, got int: 123"):
        with timer[123]:
            pass


def test_timer_accepts_list_of_tags():
    timer = Timer()
    timer.start()

    tags = ["llm", "world_model"]
    with timer[tags]:
        time.sleep(0.01)

    timer.stop()

    timing = timer.get_timing_secs()
    assert "llm" in timing
    assert "world_model" in timing
