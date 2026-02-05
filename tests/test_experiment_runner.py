"""Tests for experiment_runner (score key uses metric.id)."""
import pytest
from aiqa.experiment_runner import _metric_score_key


def test_metric_score_key_prefers_id():
    """Scores are keyed by metric.id so server/webapp lookup by id works."""
    assert _metric_score_key({"id": "m1", "name": "Accuracy"}) == "m1"
    assert _metric_score_key({"id": "m2", "name": "Relevance"}) == "m2"


def test_metric_score_key_fallback_to_name():
    """When id is missing, fall back to name for backwards compatibility."""
    assert _metric_score_key({"name": "Accuracy"}) == "Accuracy"
    assert _metric_score_key({"id": None, "name": "F1"}) == "F1"


def test_metric_score_key_empty():
    """Empty or missing id/name returns empty string."""
    assert _metric_score_key({}) == ""
    assert _metric_score_key({"id": "", "name": ""}) == ""
