"""Tests for tracing_llm_utils (LLM attribute extraction)."""

import pytest
from unittest.mock import MagicMock

from aiqa.tracing_llm_utils import _extract_and_set_token_usage


class TestExtractAndSetTokenUsage:
    """Tests for _extract_and_set_token_usage."""

    def test_extracts_usage_from_dict(self):
        """Usage is extracted when result is a dict with 'usage' key."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {}
        mock_span._attributes = {}

        result = {"usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}
        _extract_and_set_token_usage(mock_span, result)

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.usage.input_tokens") == 100
        assert calls.get("gen_ai.usage.output_tokens") == 50
        assert calls.get("gen_ai.usage.total_tokens") == 150

    def test_extracts_usage_from_json_string(self):
        """Usage is extracted when result is raw JSON response text (e.g. response.text)."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {}
        mock_span._attributes = {}

        result = '{"id": "resp_123", "usage": {"input_tokens": 6561, "output_tokens": 125, "total_tokens": 6686}}'
        _extract_and_set_token_usage(mock_span, result)

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.usage.input_tokens") == 6561
        assert calls.get("gen_ai.usage.output_tokens") == 125
        assert calls.get("gen_ai.usage.total_tokens") == 6686

    def test_json_string_without_usage_does_not_set_attributes(self):
        """When result is JSON string with no usage key, no usage attributes are set."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {}
        mock_span._attributes = {}

        result = '{"id": "resp_123", "output": []}'
        _extract_and_set_token_usage(mock_span, result)

        usage_calls = [c for c in mock_span.set_attribute.call_args_list if "gen_ai.usage" in str(c[0][0])]
        assert len(usage_calls) == 0

    def test_invalid_json_string_does_not_raise(self):
        """When result is string but not valid JSON, no exception is raised."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {}
        mock_span._attributes = {}

        result = "not json at all"
        _extract_and_set_token_usage(mock_span, result)  # should not raise
        # No usage attributes set
        usage_calls = [c for c in mock_span.set_attribute.call_args_list if "gen_ai.usage" in str(c[0][0])]
        assert len(usage_calls) == 0

    def test_zero_usage_values_preserved(self):
        """Zero token counts are preserved and set on the span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {}
        mock_span._attributes = {}

        result = {"usage": {"input_tokens": 0, "output_tokens": 5, "total_tokens": 5}}
        _extract_and_set_token_usage(mock_span, result)

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.usage.input_tokens") == 0
        assert calls.get("gen_ai.usage.output_tokens") == 5
        assert calls.get("gen_ai.usage.total_tokens") == 5
