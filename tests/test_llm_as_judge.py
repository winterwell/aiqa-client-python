"""Tests for llm_as_judge (parse_llm_response, etc.)."""

import pytest
from aiqa.llm_as_judge import parse_llm_response


class TestParseLlmResponse:
    """Tests for parse_llm_response."""

    def test_accepts_score_zero(self):
        """Score 0 is valid and must be accepted (was previously rejected as falsy)."""
        content = '{"score": 0, "message": "No relevant content"}'
        result = parse_llm_response(content)
        assert result is not None
        assert result["score"] == 0.0
        assert result.get("message") == "No relevant content"

    def test_accepts_score_one(self):
        """Score 1 is accepted."""
        content = '{"score": 1, "message": "Perfect"}'
        result = parse_llm_response(content)
        assert result is not None
        assert result["score"] == 1.0

    def test_missing_score_returns_none(self):
        """Missing score key returns None."""
        content = '{"message": "No score field"}'
        assert parse_llm_response(content) is None

    def test_clamps_score_to_range(self):
        """Score is clamped to [0, 1]."""
        assert parse_llm_response('{"score": 1.5}')["score"] == 1.0
        assert parse_llm_response('{"score": -0.1}')["score"] == 0.0
