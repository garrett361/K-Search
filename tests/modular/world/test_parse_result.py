"""Tests for ParseResult[T]."""

from k_search.modular.world.parse_result import ParseResult


def test_parse_result_ok():
    result = ParseResult.ok("value")
    assert result.success is True
    assert result.value == "value"
    assert result.error is None


def test_parse_result_fail():
    result = ParseResult.fail("something went wrong")
    assert result.success is False
    assert result.value is None
    assert result.error == "something went wrong"
