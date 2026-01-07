"""
Unit tests for tracing.py functions.
"""

import os
import inspect
import pytest
from unittest.mock import patch, MagicMock
from aiqa.tracing import get_span
from aiqa.tracing import _prepare_input, _prepare_and_filter_input


class TestGetSpan:
    """Tests for get_span function."""

    def test_get_span_success_with_span_id(self):
        """Test successful retrieval of span using spanId query."""
        span_data = {
            "id": "test-span-123",
            "name": "test_span",
            "trace_id": "abc123",
            "attributes": {"key": "value"},
        }
        mock_response_data = {"hits": [span_data]}

        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
                "AIQA_ORGANISATION_ID": "test-org",
            },
        ):
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data

                mock_get.return_value = mock_response

                result = get_span("test-span-123")

                assert result == span_data
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                assert call_args[0][0] == "http://localhost:3000/span"
                assert "q" in call_args[1]["params"]
                assert call_args[1]["params"]["q"] == "spanId:test-span-123"

    def test_get_span_success_with_client_span_id(self):
        """Test successful retrieval of span using clientSpanId query when spanId fails."""
        span_data = {
            "id": "test-span-123",
            "name": "test_span",
            "trace_id": "abc123",
        }
        mock_response_data = {"hits": [span_data]}

        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
                "AIQA_ORGANISATION_ID": "test-org",
            },
        ):
            with patch("requests.get") as mock_get:
                # First call returns 404 (spanId not found), second call succeeds (clientSpanId)
                mock_response_404 = MagicMock()
                mock_response_404.status_code = 404

                mock_response_200 = MagicMock()
                mock_response_200.status_code = 200
                mock_response_200.json.return_value = mock_response_data

                mock_get.side_effect = [mock_response_404, mock_response_200]

                result = get_span("test-span-123")

                assert result == span_data
                assert mock_get.call_count == 2
                # Check that second call uses clientSpanId
                second_call = mock_get.call_args_list[1]
                assert second_call[1]["params"]["q"] == "clientSpanId:test-span-123"

    def test_get_span_not_found(self):
        """Test that get_span returns None when span is not found."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
                "AIQA_ORGANISATION_ID": "test-org",
            },
        ):
            with patch("requests.get") as mock_get:
                # Both queries return 404
                mock_response_404 = MagicMock()
                mock_response_404.status_code = 404

                mock_get.return_value = mock_response_404

                result = get_span("nonexistent-span")

                assert result is None
                assert mock_get.call_count == 2

    def test_get_span_empty_hits(self):
        """Test that get_span returns None when hits array is empty."""
        mock_response_data = {"hits": []}

        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
                "AIQA_ORGANISATION_ID": "test-org",
            },
        ):
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data

                mock_get.return_value = mock_response

                result = get_span("test-span-123")

                assert result is None


    def test_get_span_missing_organisation_id(self):
        """Test that get_span raises ValueError when organisation ID is not provided."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="Organisation ID is required"):
                get_span("test-span-123")

    def test_get_span_missing_api_key(self):
        """Test that get_span raises ValueError when AIQA_API_KEY is not set."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_ORGANISATION_ID": "test-org",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="API key is required"):
                get_span("test-span-123")

    def test_get_span_with_organisation_id_parameter(self):
        """Test that get_span uses organisation_id parameter when provided."""
        span_data = {"id": "test-span-123", "name": "test_span"}
        mock_response_data = {"hits": [span_data]}

        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
            },
            clear=True,
        ):
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data

                mock_get.return_value = mock_response

                result = get_span("test-span-123", organisation_id="param-org")

                assert result == span_data
                call_args = mock_get.call_args
                assert call_args[1]["params"]["organisation"] == "param-org"

    def test_get_span_server_error(self):
        """Test that get_span raises ValueError on server error."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
                "AIQA_ORGANISATION_ID": "test-org",
            },
        ):
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.text = "Internal Server Error"

                mock_get.return_value = mock_response

                with pytest.raises(ValueError, match="Failed to get span: 500"):
                    get_span("test-span-123")

    def test_get_span_authorization_header(self):
        """Test that get_span includes Authorization header with API key."""
        span_data = {"id": "test-span-123"}
        mock_response_data = {"hits": [span_data]}

        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key-123",
                "AIQA_ORGANISATION_ID": "test-org",
            },
        ):
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data

                mock_get.return_value = mock_response

                get_span("test-span-123")

                call_args = mock_get.call_args
                assert call_args[1]["headers"]["Authorization"] == "ApiKey test-api-key-123"


class TestPrepareInput:
    """Tests for _prepare_input function with signature-based arg conversion."""

    def test_positional_args_converted_to_dict(self):
        """Test that positional args are converted to named args using signature."""
        def test_func(x: int, y: str, z: float = 1.0):
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_input((5, "hello"), {}, sig)
        
        assert isinstance(result, dict)
        assert result["x"] == 5
        assert result["y"] == "hello"
        assert result["z"] == 1.0  # default value applied

    def test_kwargs_merged_with_positional(self):
        """Test that kwargs are merged with positional args into single dict."""
        def test_func(x: int, y: str, z: float = 1.0):
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_input((5,), {"y": "hello", "z": 2.0}, sig)
        
        assert isinstance(result, dict)
        assert result["x"] == 5
        assert result["y"] == "hello"
        assert result["z"] == 2.0  # kwargs override defaults

    def test_all_kwargs_only(self):
        """Test that kwargs-only calls produce dict."""
        def test_func(x: int, y: str):
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_input((), {"x": 5, "y": "hello"}, sig)
        
        assert isinstance(result, dict)
        assert result["x"] == 5
        assert result["y"] == "hello"

    def test_no_args_no_kwargs(self):
        """Test that empty args/kwargs returns None."""
        def test_func():
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_input((), {}, sig)
        
        assert result is None

    def test_fallback_when_signature_none(self):
        """Test fallback to legacy behavior when signature is None."""
        result = _prepare_input((5, 3), {}, None)
        
        # Should fall back to legacy: single arg returns as-is, multiple args return list
        assert result == [5, 3]

    def test_fallback_when_binding_fails(self):
        """Test fallback when signature binding fails (e.g., wrong number of args)."""
        def test_func(x: int, y: str):
            pass
        
        sig = inspect.signature(test_func)
        # Pass wrong number of args - binding should fail
        result = _prepare_input((5, 3, 4), {}, sig)
        
        # Should fall back to legacy behavior
        assert isinstance(result, list)
        assert result == [5, 3, 4]

    def test_single_arg_dict_fallback(self):
        """Test that single dict arg returns dict copy (legacy behavior)."""
        result = _prepare_input(({"key": "value"},), {}, None)
        
        assert isinstance(result, dict)
        assert result == {"key": "value"}
        # Should be a copy, not the same object
        assert result is not {"key": "value"}

    def test_single_non_dict_arg_fallback(self):
        """Test that single non-dict arg returns as-is (legacy behavior)."""
        result = _prepare_input((42,), {}, None)
        
        assert result == 42

    def test_kwargs_only_fallback(self):
        """Test kwargs-only fallback behavior."""
        result = _prepare_input((), {"x": 1, "y": 2}, None)
        
        assert isinstance(result, dict)
        assert result == {"x": 1, "y": 2}

    def test_args_and_kwargs_fallback(self):
        """Test args + kwargs fallback combines into dict with 'args' key."""
        result = _prepare_input((1, 2), {"z": 3}, None)
        
        assert isinstance(result, dict)
        assert result["args"] == [1, 2]
        assert result["z"] == 3


class TestPrepareAndFilterInput:
    """Tests for _prepare_and_filter_input function."""

    def test_positional_args_with_signature(self):
        """Test that positional args are converted using signature."""
        def test_func(x: int, y: str):
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_and_filter_input((5, "hello"), {}, None, None, sig)
        
        assert isinstance(result, dict)
        assert result["x"] == 5
        assert result["y"] == "hello"

    def test_ignore_input_filters_dict(self):
        """Test that ignore_input filters keys from dict."""
        def test_func(x: int, y: str, z: float):
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_and_filter_input((5, "hello", 1.0), {}, None, ["y"], sig)
        
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" not in result  # filtered out
        assert "z" in result

    def test_ignore_self_removes_first_arg(self):
        """Test that ignore_input=['self'] removes first arg and adjusts signature."""
        def test_func(self, x: int, y: str):
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_and_filter_input((None, 5, "hello"), {}, None, ["self"], sig)
        
        assert isinstance(result, dict)
        assert "self" not in result
        assert result["x"] == 5
        assert result["y"] == "hello"

    def test_ignore_self_with_no_self_in_signature(self):
        """Test that ignore_input=['self'] removes first arg even if signature doesn't have self."""
        # For bound methods or regular functions without self, signature doesn't include self
        # But if user specifies ignore_input=['self'], we still remove first arg
        def test_func(x: int, y: str):
            pass
        
        sig = inspect.signature(test_func)
        # Signature doesn't have 'self', but we still remove first arg
        # Signature won't be adjusted (only adjusted if first param is 'self')
        # So binding will fail and fall back to legacy behavior
        result = _prepare_and_filter_input((5, "hello"), {}, None, ["self"], sig)
        
        # Binding fails because we removed first arg but signature wasn't adjusted
        # Falls back to legacy: single arg returns as-is
        assert result == "hello"  # Single remaining arg returned as-is

    def test_filter_input_applied(self):
        """Test that filter_input function is applied."""
        def test_func(x: int, y: str):
            pass
        
        sig = inspect.signature(test_func)
        filter_fn = lambda d: {"sum": sum(v for v in d.values() if isinstance(v, (int, float))) if isinstance(d, dict) else d}
        result = _prepare_and_filter_input((5, "hello"), {}, filter_fn, None, sig)
        
        # filter_input should transform the dict
        assert isinstance(result, dict)

    def test_ignore_input_warns_on_non_dict(self):
        """Test that ignore_input warns when input is not a dict."""
        # When signature binding fails, result might not be a dict
        # and ignore_input should warn
        from aiqa import tracing
        with patch.object(tracing.logger, 'warning') as mock_warning:
            result = _prepare_and_filter_input((5, 3), {}, None, ["x"], None)
            # Should have logged a warning since result is a list, not a dict
            # (result will be [5, 3], not a dict, so ignore_input can't apply)
            assert result == [5, 3]
            # Warning should be called because ignore_input is set but input_data is not a dict
            mock_warning.assert_called_once()
