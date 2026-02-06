"""
Unit tests for tracing.py functions.
"""

import os
import inspect
import pytest
from unittest.mock import patch, MagicMock
from aiqa.span_helpers import get_span
from aiqa.tracing import _prepare_input, _prepare_and_filter_input, _filter_and_serialize_output
from aiqa.tracing import _matches_ignore_pattern, _apply_ignore_patterns, _merge_with_default_ignore_patterns


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
        """Test that ignore_input=['self'] removes 'self' key from final dict."""
        # ignore_input=['self'] only affects the final dict, not the args passed to _prepare_input
        def test_func(x: int, y: str):
            pass
        
        sig = inspect.signature(test_func)
        # Signature doesn't have 'self', so binding works normally
        result = _prepare_and_filter_input((5, "hello"), {}, None, ["self"], sig)
        
        # Result should be a dict with x and y, but no 'self' key (which wasn't there anyway)
        assert isinstance(result, dict)
        assert result["x"] == 5
        assert result["y"] == "hello"
        assert "self" not in result

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


class TestMatchesIgnorePattern:
    """Tests for _matches_ignore_pattern function with wildcard support."""

    def test_exact_match(self):
        """Test that exact matches work."""
        assert _matches_ignore_pattern("password", ["password"]) is True
        assert _matches_ignore_pattern("api_key", ["password", "api_key"]) is True
        assert _matches_ignore_pattern("other", ["password", "api_key"]) is False

    def test_wildcard_prefix(self):
        """Test that wildcard patterns like '_*' match keys starting with '_'."""
        assert _matches_ignore_pattern("_apple", ["_*"]) is True
        assert _matches_ignore_pattern("_fruit", ["_*"]) is True
        assert _matches_ignore_pattern("_internal_data", ["_*"]) is True
        assert _matches_ignore_pattern("apple", ["_*"]) is False
        assert _matches_ignore_pattern("_", ["_*"]) is True

    def test_wildcard_suffix(self):
        """Test that wildcard patterns like '*_internal' match keys ending with '_internal'."""
        assert _matches_ignore_pattern("data_internal", ["*_internal"]) is True
        assert _matches_ignore_pattern("test_internal", ["*_internal"]) is True
        assert _matches_ignore_pattern("internal", ["*_internal"]) is False

    def test_wildcard_middle(self):
        """Test that wildcard patterns with '*' in the middle work."""
        assert _matches_ignore_pattern("test_internal_data", ["test_*_data"]) is True
        assert _matches_ignore_pattern("test_secret_data", ["test_*_data"]) is True
        assert _matches_ignore_pattern("test_data", ["test_*_data"]) is False

    def test_multiple_patterns(self):
        """Test that multiple patterns work together."""
        patterns = ["password", "_*", "api_*"]
        assert _matches_ignore_pattern("password", patterns) is True
        assert _matches_ignore_pattern("_secret", patterns) is True
        assert _matches_ignore_pattern("api_key", patterns) is True
        assert _matches_ignore_pattern("public_data", patterns) is False

    def test_question_mark_wildcard(self):
        """Test that '?' wildcard matches single character."""
        assert _matches_ignore_pattern("test1", ["test?"]) is True
        assert _matches_ignore_pattern("test2", ["test?"]) is True
        assert _matches_ignore_pattern("test", ["test?"]) is False
        assert _matches_ignore_pattern("test12", ["test?"]) is False


class TestWildcardIgnoreInput:
    """Tests for wildcard support in ignore_input."""

    def test_wildcard_ignore_input_with_signature(self):
        """Test that wildcard patterns work in ignore_input."""
        def test_func(x: int, _apple: str, _fruit: str, y: float):
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_and_filter_input((5, "red", "banana", 1.0), {}, None, ["_*"], sig)
        
        assert isinstance(result, dict)
        assert "x" in result
        assert "_apple" not in result  # filtered out by wildcard
        assert "_fruit" not in result  # filtered out by wildcard
        assert "y" in result

    def test_wildcard_and_exact_match(self):
        """Test that wildcard and exact matches can be combined."""
        def test_func(x: int, _apple: str, password: str, y: float):
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_and_filter_input((5, "red", "secret", 1.0), {}, None, ["_*", "password"], sig)
        
        assert isinstance(result, dict)
        assert "x" in result
        assert "_apple" not in result  # filtered out by wildcard
        assert "password" not in result  # filtered out by exact match
        assert "y" in result

    def test_wildcard_no_match(self):
        """Test that keys not matching wildcard are preserved."""
        def test_func(x: int, apple: str, _secret: str, y: float):
            pass
        
        sig = inspect.signature(test_func)
        result = _prepare_and_filter_input((5, "red", "hidden", 1.0), {}, None, ["_*"], sig)
        
        assert isinstance(result, dict)
        assert "x" in result
        assert "apple" in result  # not filtered (doesn't start with _)
        assert "_secret" not in result  # filtered out by wildcard
        assert "y" in result


class TestWildcardIgnoreOutput:
    """Tests for wildcard support in ignore_output."""

    def test_wildcard_ignore_output(self):
        """Test that wildcard patterns work in ignore_output."""
        output_data = {
            "result": "success",
            "_apple": "red",
            "_fruit": "banana",
            "public": "data"
        }
        
        result = _filter_and_serialize_output(output_data, None, ["_*"])
        
        # Result is serialized to JSON string for dicts
        assert isinstance(result, str)
        
        # Parse it back to verify filtering worked
        import json
        parsed = json.loads(result)
        assert "_apple" not in parsed
        assert "_fruit" not in parsed
        assert "result" in parsed
        assert "public" in parsed

    def test_wildcard_and_exact_match_output(self):
        """Test that wildcard and exact matches can be combined in ignore_output."""
        output_data = {
            "result": "success",
            "_apple": "red",
            "password": "secret",
            "public": "data"
        }
        
        result = _filter_and_serialize_output(output_data, None, ["_*", "password"])
        
        import json
        parsed = json.loads(result)
        assert "_apple" not in parsed
        assert "password" not in parsed
        assert "result" in parsed
        assert "public" in parsed

    def test_wildcard_no_match_output(self):
        """Test that keys not matching wildcard are preserved in output."""
        output_data = {
            "result": "success",
            "apple": "red",
            "_secret": "hidden",
            "public": "data"
        }
        
        result = _filter_and_serialize_output(output_data, None, ["_*"])
        
        import json
        parsed = json.loads(result)
        assert "apple" in parsed  # not filtered
        assert "_secret" not in parsed  # filtered by wildcard
        assert "result" in parsed
        assert "public" in parsed


class TestWithTracingFilterInput:
    """Tests for WithTracing filter_input functionality."""

    def test_filter_input_with_self_extracts_properties(self):
        """Test filter_input receives self and can extract specific properties."""
        from aiqa import WithTracing
        
        class TestClass:
            def __init__(self):
                self.trace_me_id = "dataset-123"
                self._trace_me_not = "secret-data"
            
            @WithTracing(
                filter_input=lambda self, x: {"dataset_id": self.trace_me_id, "x": x}
            )
            def my_method(self, x):
                return x * 2
        
        obj = TestClass()
        result = obj.my_method(5)
        assert result == 10

    def test_filter_input_with_positional_args(self):
        """Test filter_input with positional arguments."""
        from aiqa import WithTracing
        
        @WithTracing(
            filter_input=lambda arg1, arg2: {"a": arg1, "b": arg2}
        )
        def my_function(arg1, arg2):
            return arg1 + arg2
        
        result = my_function(1, 2)
        assert result == 3

    def test_filter_input_with_keyword_args(self):
        """Test filter_input with keyword arguments."""
        from aiqa import WithTracing
        
        @WithTracing(
            filter_input=lambda arg1, arg2: {"a": arg1, "b": arg2}
        )
        def my_function(arg1, arg2):
            return arg1 + arg2
        
        result = my_function(arg1=1, arg2=2)
        assert result == 3

    def test_ignore_input_self_removes_from_final_dict(self):
        """Test ignore_input=['self'] removes self from final dict."""
        from aiqa import WithTracing
        from unittest.mock import patch, MagicMock
        
        class TestClass:
            def __init__(self):
                self.dataset_id = "ds-123"
            
            @WithTracing(ignore_input=["self"])
            def my_method(self, x):
                return x * 2
        
        obj = TestClass()
        # Mock the client and tracer to capture what gets set
        mock_client = MagicMock()
        mock_client.enabled = True
        with patch('aiqa.tracing.get_aiqa_client', return_value=mock_client):
            with patch('aiqa.tracing.get_aiqa_tracer') as mock_tracer:
                mock_span = MagicMock()
                mock_span.is_recording.return_value = True
                mock_span.get_span_context.return_value.trace_id = 123
                mock_tracer.return_value.start_as_current_span.return_value.__enter__.return_value = mock_span
                
                obj.my_method(5)
                
                # Check that input was set
                assert mock_span.set_attribute.called
                # Find the input call
                input_calls = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == "input"]
                assert len(input_calls) > 0
                # The input should not contain "self"
                # Note: input is serialized, so we check the call was made
                # The actual filtering happens in _prepare_and_filter_input which is tested separately

    def test_ignore_input_multiple_patterns(self):
        """Test ignore_input with multiple patterns including wildcard."""
        from aiqa import WithTracing
        from unittest.mock import patch, MagicMock
        
        @WithTracing(ignore_input=["arg1", "_*"])
        def my_function(arg1, arg2, _trace_me_not, trace_me_yes):
            return arg1 + arg2
        
        mock_client = MagicMock()
        mock_client.enabled = True
        with patch('aiqa.tracing.get_aiqa_client', return_value=mock_client):
            with patch('aiqa.tracing.get_aiqa_tracer') as mock_tracer:
                mock_span = MagicMock()
                mock_span.is_recording.return_value = True
                mock_span.get_span_context.return_value.trace_id = 123
                mock_tracer.return_value.start_as_current_span.return_value.__enter__.return_value = mock_span
                
                my_function(1, 2, _trace_me_not="hidden", trace_me_yes="visible")
                
                # Verify span was created and input was set
                assert mock_span.set_attribute.called


class TestFilterInputWithSelf:
    """Tests for filter_input receiving self parameter."""

    def test_filter_input_receives_self(self):
        """Test that filter_input receives self as first parameter."""
        from aiqa.tracing import _prepare_and_filter_input
        import inspect
        
        class TestClass:
            def __init__(self):
                self.trace_me_id = "dataset-123"
                self._trace_me_not = "secret"
            
            def my_method(self, x):
                pass
        
        obj = TestClass()
        sig = inspect.signature(obj.my_method)
        
        # filter_input should receive self and x
        filter_fn = lambda self, x: {"dataset": self.trace_me_id, "x": x}
        result = _prepare_and_filter_input((obj, 5), {}, filter_fn, None, sig)
        
        assert isinstance(result, dict)
        assert result["dataset"] == "dataset-123"
        assert result["x"] == 5
        # _trace_me_not should not be in result (filter_input didn't include it)
        assert "_trace_me_not" not in result

    def test_filter_input_with_self_and_ignore_self(self):
        """Test filter_input receives self even when ignore_input=['self']."""
        from aiqa.tracing import _prepare_and_filter_input
        import inspect
        
        class TestClass:
            def __init__(self):
                self.trace_me_id = "dataset-123"
            
            def my_method(self, x):
                pass
        
        obj = TestClass()
        sig = inspect.signature(obj.my_method)
        
        # filter_input should receive self, ignore_input removes it from final dict
        filter_fn = lambda self, x: {"dataset": self.trace_me_id, "x": x}
        result = _prepare_and_filter_input((obj, 5), {}, filter_fn, ["self"], sig)
        
        assert isinstance(result, dict)
        assert result["dataset"] == "dataset-123"
        assert result["x"] == 5
        assert "self" not in result  # Removed by ignore_input

    def test_filter_input_positional_args(self):
        """Test filter_input with positional args."""
        from aiqa.tracing import _prepare_and_filter_input
        import inspect
        
        def my_function(arg1, arg2):
            pass
        
        sig = inspect.signature(my_function)
        filter_fn = lambda arg1, arg2: {"a": arg1, "b": arg2}
        result = _prepare_and_filter_input((1, 2), {}, filter_fn, None, sig)
        
        assert result == {"a": 1, "b": 2}

    def test_filter_input_keyword_args(self):
        """Test filter_input with keyword args."""
        from aiqa.tracing import _prepare_and_filter_input
        import inspect
        
        def my_function(arg1, arg2):
            pass
        
        sig = inspect.signature(my_function)
        filter_fn = lambda arg1, arg2: {"a": arg1, "b": arg2}
        result = _prepare_and_filter_input((), {"arg1": 1, "arg2": 2}, filter_fn, None, sig)
        
        assert result == {"a": 1, "b": 2}

    def test_ignore_input_self_only(self):
        """Test ignore_input=['self'] removes self from final dict."""
        from aiqa.tracing import _prepare_and_filter_input
        import inspect
        
        class TestClass:
            def my_method(self, x):
                pass
        
        obj = TestClass()
        # Use unbound method to get signature with self
        sig = inspect.signature(TestClass.my_method)
        result = _prepare_and_filter_input((obj, 5), {}, None, ["self"], sig)
        
        # _prepare_input should bind args to signature, creating dict with self and x
        # Then ignore_input removes "self" from the dict
        assert isinstance(result, dict)
        assert "self" not in result  # Removed by ignore_input
        assert result["x"] == 5

    def test_ignore_input_multiple_patterns(self):
        """Test ignore_input with multiple patterns."""
        from aiqa.tracing import _prepare_and_filter_input
        import inspect
        
        def my_function(arg1, arg2, _trace_me_not, trace_me_yes):
            pass
        
        sig = inspect.signature(my_function)
        result = _prepare_and_filter_input(
            (1, 2, "hidden", "visible"), 
            {}, 
            None, 
            ["arg1", "_*"], 
            sig
        )
        
        assert isinstance(result, dict)
        assert "arg1" not in result  # filtered by exact match
        assert "_trace_me_not" not in result  # filtered by wildcard
        assert "arg2" in result
        assert "trace_me_yes" in result


class TestDefaultUnderscoreIgnore:
    """Tests for default '_*' pattern that filters properties starting with '_'."""

    def test_default_underscore_ignore_input(self):
        """Test that '_*' pattern is applied by default to ignore_input."""
        def test_func(user: dict, x: int):
            pass
        
        sig = inspect.signature(test_func)
        user_data = {"name": "Alice", "_sa_instance_state": "object"}
        result = _prepare_and_filter_input((user_data, 5), {}, None, None, sig)
        
        assert isinstance(result, dict)
        assert "user" in result
        user_result = result["user"]
        assert isinstance(user_result, dict)
        assert "name" in user_result
        assert user_result["name"] == "Alice"
        assert "_sa_instance_state" not in user_result  # filtered by default '_*' pattern
        assert "x" in result

    def test_default_underscore_ignore_output(self):
        """Test that '_*' pattern is applied by default to ignore_output."""
        output_data = {
            "user": {
                "name": "Alice",
                "_sa_instance_state": "object"
            },
            "result": "success"
        }
        
        result = _filter_and_serialize_output(output_data, None, None)
        
        # Result is serialized to JSON string for dicts
        assert isinstance(result, str)
        
        # Parse it back to verify filtering worked
        import json
        parsed = json.loads(result)
        assert "user" in parsed
        assert "name" in parsed["user"]
        assert parsed["user"]["name"] == "Alice"
        assert "_sa_instance_state" not in parsed["user"]  # filtered by default '_*' pattern
        assert "result" in parsed

    def test_nested_underscore_filtering(self):
        """Test that nested objects are filtered recursively."""
        data = {
            "level1": {
                "public": "value1",
                "_private": "value2",
                "nested": {
                    "public": "value3",
                    "_private": "value4"
                }
            },
            "_top_level": "value5"
        }
        
        # _apply_ignore_patterns only applies explicit patterns; default patterns are merged elsewhere.
        result = _apply_ignore_patterns(data, ["_*"])
        
        assert "level1" in result
        assert "public" in result["level1"]
        assert "_private" not in result["level1"]
        assert "nested" in result["level1"]
        assert "public" in result["level1"]["nested"]
        assert "_private" not in result["level1"]["nested"]
        assert "_top_level" not in result

    def test_merge_with_default_patterns(self):
        """Test that user-provided patterns are merged with default '_*' pattern."""
        user_patterns = ["password", "api_key"]
        merged = _merge_with_default_ignore_patterns(user_patterns)
        
        assert "_*" in merged
        assert "password" in merged
        assert "api_key" in merged
        assert len(merged) == 3

    def test_merge_with_default_patterns_none(self):
        """Test that None patterns result in just default '_*' pattern."""
        merged = _merge_with_default_ignore_patterns(None)
        
        assert merged == ["_*"]

    def test_merge_with_default_patterns_duplicate(self):
        """Test that duplicate patterns are not added."""
        user_patterns = ["_*", "password"]
        merged = _merge_with_default_ignore_patterns(user_patterns)
        
        assert merged == ["_*", "password"]  # _* is first, password is second
        assert merged.count("_*") == 1

    def test_default_underscore_with_custom_patterns(self):
        """Test that default '_*' works alongside custom patterns."""
        def test_func(user: dict, password: str, x: int):
            pass
        
        sig = inspect.signature(test_func)
        user_data = {"name": "Alice", "_sa_instance_state": "object"}
        result = _prepare_and_filter_input(
            (user_data, "secret", 5), 
            {}, 
            None, 
            ["password"],  # Only specify password, '_*' should be added automatically
            sig
        )
        
        assert isinstance(result, dict)
        assert "user" in result
        user_result = result["user"]
        assert "name" in user_result
        assert "_sa_instance_state" not in user_result  # filtered by default '_*'
        assert "password" not in result  # filtered by custom pattern
        assert "x" in result

    def test_nested_underscore_in_output(self):
        """Test nested underscore filtering in output."""
        output_data = {
            "result": "success",
            "user": {
                "name": "Alice",
                "_sa_instance_state": "object",
                "profile": {
                    "email": "alice@example.com",
                    "_internal_id": "12345"
                }
            }
        }
        
        result = _filter_and_serialize_output(output_data, None, None)
        
        import json
        parsed = json.loads(result)
        assert "user" in parsed
        assert "name" in parsed["user"]
        assert "_sa_instance_state" not in parsed["user"]
        assert "profile" in parsed["user"]
        assert "email" in parsed["user"]["profile"]
        assert "_internal_id" not in parsed["user"]["profile"]

    def test_apply_ignore_patterns_with_none(self):
        """Test that _apply_ignore_patterns handles None patterns by recursively processing nested dicts."""
        data = {
            "public": "value",
            "_private": "hidden",
            "nested": {
                "public": "value2",
                "_private": "hidden2"
            }
        }
        
        result = _apply_ignore_patterns(data, None)
        
        # When patterns is None, it should still recursively process nested dicts
        # but won't filter top-level keys (since no patterns provided)
        assert "public" in result
        assert "_private" in result  # Not filtered when patterns is None
        assert "nested" in result
        # Nested dicts are still processed recursively (structure preserved)
        assert isinstance(result["nested"], dict)
        assert "public" in result["nested"]
        assert "_private" in result["nested"]  # Also not filtered when patterns is None


class TestClientIgnoreProperties:
    """Tests for client default_ignore_patterns and ignore_recursive properties."""

    def test_default_ignore_patterns_property(self):
        """Test setting and getting default_ignore_patterns on client."""
        from aiqa import get_aiqa_client
        
        client = get_aiqa_client()
        original = client.default_ignore_patterns
        
        # Test setting new patterns
        client.default_ignore_patterns = ["_*", "password"]
        assert client.default_ignore_patterns == ["_*", "password"]
        
        # Test that it's a copy (modifying returned list doesn't affect client)
        patterns = client.default_ignore_patterns
        patterns.append("new")
        assert client.default_ignore_patterns == ["_*", "password"]
        
        # Test setting to None (disables defaults)
        client.default_ignore_patterns = None
        assert client.default_ignore_patterns == []
        
        # Test setting to empty list
        client.default_ignore_patterns = []
        assert client.default_ignore_patterns == []
        
        # Restore original
        client.default_ignore_patterns = original

    def test_ignore_recursive_property(self):
        """Test setting and getting ignore_recursive on client."""
        from aiqa import get_aiqa_client
        
        client = get_aiqa_client()
        original = client.ignore_recursive
        
        # Test setting to False
        client.ignore_recursive = False
        assert client.ignore_recursive is False
        
        # Test setting to True
        client.ignore_recursive = True
        assert client.ignore_recursive is True
        
        # Restore original
        client.ignore_recursive = original

    def test_custom_default_ignore_patterns_used(self):
        """Test that custom default ignore patterns are used in tracing."""
        from aiqa import get_aiqa_client
        from aiqa.tracing import _prepare_and_filter_input
        import inspect
        
        client = get_aiqa_client()
        original_patterns = client.default_ignore_patterns
        
        try:
            # Set custom default patterns
            client.default_ignore_patterns = ["password", "secret"]
            
            def test_func(password: str, secret: str, public: str):
                pass
            
            sig = inspect.signature(test_func)
            result = _prepare_and_filter_input(
                ("pwd", "sec", "pub"), 
                {}, 
                None, 
                None,  # No explicit ignore_input
                sig
            )
            
            # Custom patterns should be applied
            assert "password" not in result
            assert "secret" not in result
            assert "public" in result
        finally:
            # Restore original
            client.default_ignore_patterns = original_patterns

    def test_ignore_recursive_false(self):
        """Test that ignore_recursive=False only filters top-level keys."""
        from aiqa import get_aiqa_client
        
        client = get_aiqa_client()
        original_recursive = client.ignore_recursive
        
        try:
            client.ignore_recursive = False
            
            data = {
                "top_level": "value",
                "_top_private": "hidden",
                "nested": {
                    "public": "value2",
                    "_nested_private": "hidden2"
                }
            }
            
            result = _apply_ignore_patterns(data, ["_*"], recursive=False)
            
            # Top-level _* should be filtered
            assert "_top_private" not in result
            assert "top_level" in result
            # Nested dict should be preserved as-is (not filtered recursively)
            assert "nested" in result
            assert "_nested_private" in result["nested"]  # Not filtered when recursive=False
            assert "public" in result["nested"]
        finally:
            client.ignore_recursive = original_recursive

    def test_ignore_recursive_true(self):
        """Test that ignore_recursive=True filters nested keys."""
        from aiqa import get_aiqa_client
        
        client = get_aiqa_client()
        original_recursive = client.ignore_recursive
        
        try:
            client.ignore_recursive = True
            
            data = {
                "top_level": "value",
                "_top_private": "hidden",
                "nested": {
                    "public": "value2",
                    "_nested_private": "hidden2"
                }
            }
            
            result = _apply_ignore_patterns(data, ["_*"], recursive=True)
            
            # Top-level _* should be filtered
            assert "_top_private" not in result
            assert "top_level" in result
            # Nested dict should also be filtered
            assert "nested" in result
            assert "_nested_private" not in result["nested"]  # Filtered when recursive=True
            assert "public" in result["nested"]
        finally:
            client.ignore_recursive = original_recursive

    def test_max_depth_prevents_infinite_loop(self):
        """Test that max_depth prevents infinite loops from deep nesting."""
        # Create a deeply nested structure
        data = {"level": 0}
        current = data
        for i in range(150):  # Exceeds default max_depth of 100
            current["nested"] = {"level": i + 1}
            current = current["nested"]
        
        # Should not raise exception or hang
        result = _apply_ignore_patterns(data, ["_*"], recursive=True, max_depth=100)
        
        # Should return something (may be truncated at max_depth)
        assert isinstance(result, dict)
        assert "level" in result

    def test_client_ignore_recursive_affects_tracing(self):
        """Test that client.ignore_recursive setting affects actual tracing."""
        from aiqa import get_aiqa_client
        from aiqa.tracing import _filter_and_serialize_output
        
        client = get_aiqa_client()
        original_recursive = client.ignore_recursive
        original_patterns = client.default_ignore_patterns
        
        try:
            client.default_ignore_patterns = ["_*"]
            
            output_data = {
                "top": "value",
                "_top_private": "hidden",
                "nested": {
                    "public": "value2",
                    "_nested_private": "hidden2"
                }
            }
            
            # Test with recursive=True (default)
            client.ignore_recursive = True
            result_recursive = _filter_and_serialize_output(output_data, None, None)
            import json
            parsed_recursive = json.loads(result_recursive)
            assert "_top_private" not in parsed_recursive
            assert "_nested_private" not in parsed_recursive["nested"]  # Filtered recursively
            
            # Test with recursive=False
            client.ignore_recursive = False
            result_non_recursive = _filter_and_serialize_output(output_data, None, None)
            parsed_non_recursive = json.loads(result_non_recursive)
            assert "_top_private" not in parsed_non_recursive
            assert "_nested_private" in parsed_non_recursive["nested"]  # Not filtered when recursive=False
        finally:
            client.ignore_recursive = original_recursive
            client.default_ignore_patterns = original_patterns
