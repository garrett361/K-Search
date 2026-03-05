"""Tests for Cycle behavior (best_round, succeeded)."""

from unittest.mock import MagicMock

from k_search.modular.world.cycle import Cycle


def _mock_round(success: bool, score: float) -> MagicMock:
    r = MagicMock()
    r.result.is_success.return_value = success
    r.score = score
    return r


def test_best_round_returns_highest_scoring_success():
    r1 = _mock_round(success=True, score=0.5)
    r2 = _mock_round(success=False, score=0.9)  # failed, ignored
    r3 = _mock_round(success=True, score=0.8)
    cycle = Cycle(rounds=[r1, r2, r3])

    assert cycle.best_round is r3
    assert cycle.succeeded is True


def test_best_round_none_when_all_failed():
    cycle = Cycle(rounds=[_mock_round(False, 0.0), _mock_round(False, 0.0)])
    assert cycle.best_round is None
    assert cycle.succeeded is False
