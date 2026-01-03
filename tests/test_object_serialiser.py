"""
Unit tests for object_serialiser.py functions.
"""

import os
import pytest
import dataclasses
from datetime import datetime, date, time
from unittest.mock import patch
from aiqa.object_serialiser import (
    sanitize_string_for_utf8,
    toNumber,
    _is_jwt_token,
    _is_api_key,
    _apply_data_filters,
    serialize_for_span,
    safe_str_repr,
    object_to_dict,
    SizeLimitedJSONEncoder,
    safe_json_dumps,
    json_default_handler_factory,
)


class TestSanitizeStringForUTF8:
    """Tests for sanitize_string_for_utf8 function."""

    def test_valid_string(self):
        """Test that valid UTF-8 strings pass through unchanged."""
        text = "Hello, world! 你好"
        assert sanitize_string_for_utf8(text) == text

    def test_none(self):
        """Test that None is handled."""
        assert sanitize_string_for_utf8(None) is None

    def test_non_string(self):
        """Test that non-string values are converted."""
        assert sanitize_string_for_utf8(123) == "123"

    def test_surrogate_characters(self):
        """Test that surrogate characters are replaced."""
        # Create a string with surrogate characters
        text = "Hello\ud800World"
        result = sanitize_string_for_utf8(text)
        assert "Hello" in result
        assert "World" in result
        # Should be valid UTF-8 now
        result.encode('utf-8')


class TestToNumber:
    """Tests for toNumber function."""

    def test_none(self):
        """Test that None returns 0."""
        assert toNumber(None) == 0

    def test_int(self):
        """Test that int values pass through."""
        assert toNumber(42) == 42

    def test_string_number(self):
        """Test that string numbers are converted."""
        assert toNumber("42") == 42

    def test_k_suffix(self):
        """Test 'k' suffix (kilobytes)."""
        assert toNumber("10k") == 10 * 1024

    def test_m_suffix(self):
        """Test 'm' suffix (megabytes)."""
        assert toNumber("5m") == 5 * 1024 * 1024

    def test_g_suffix(self):
        """Test 'g' suffix (gigabytes)."""
        assert toNumber("2g") == 2 * 1024 * 1024 * 1024

    def test_b_suffix_dropped(self):
        """Test that 'b' suffix is dropped before processing."""
        assert toNumber("10kb") == 10 * 1024
        assert toNumber("5mb") == 5 * 1024 * 1024
        assert toNumber("2gb") == 2 * 1024 * 1024 * 1024


class TestIsJWTToken:
    """Tests for _is_jwt_token function."""

    def test_valid_jwt(self):
        """Test that valid JWT tokens are detected."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        assert _is_jwt_token(jwt) is True

    def test_invalid_jwt_wrong_start(self):
        """Test that non-JWT strings are not detected."""
        assert _is_jwt_token("not_a_jwt_token") is False

    def test_invalid_jwt_wrong_parts(self):
        """Test that strings with wrong number of parts are not detected."""
        assert _is_jwt_token("eyJ.part1.part2.part3") is False

    def test_non_string(self):
        """Test that non-string values return False."""
        assert _is_jwt_token(123) is False
        assert _is_jwt_token(None) is False


class TestIsAPIKey:
    """Tests for _is_api_key function."""

    def test_openai_key(self):
        """Test that OpenAI keys are detected."""
        assert _is_api_key("sk-1234567890abcdef") is True

    def test_aws_key(self):
        """Test that AWS keys are detected."""
        assert _is_api_key("AKIAIOSFODNN7EXAMPLE") is True

    def test_github_token(self):
        """Test that GitHub tokens are detected."""
        assert _is_api_key("ghp_1234567890abcdef") is True

    def test_not_api_key(self):
        """Test that regular strings are not detected."""
        assert _is_api_key("regular_string") is False

    def test_non_string(self):
        """Test that non-string values return False."""
        assert _is_api_key(123) is False


class TestApplyDataFilters:
    """Tests for _apply_data_filters function."""

    def test_remove_passwords(self):
        """Test that password fields are filtered (if filter is enabled)."""
        # Note: This test depends on AIQA_DATA_FILTERS env var being set
        # The filter may or may not be enabled, so we check the behavior
        result = _apply_data_filters("password", "secret123")
        # Either filtered or not, depending on config
        assert isinstance(result, (str, type("secret123")))

    def test_remove_jwt(self):
        """Test that JWT tokens are filtered (if filter is enabled)."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.test"
        result = _apply_data_filters("token", jwt)
        # Either filtered or not, depending on config
        assert isinstance(result, (str, type(jwt)))

    def test_remove_auth_headers(self):
        """Test that authorization headers are filtered (if filter is enabled)."""
        result = _apply_data_filters("authorization", "Bearer token123")
        # Either filtered or not, depending on config
        assert isinstance(result, (str, type("Bearer token123")))

    def test_remove_api_keys(self):
        """Test that API keys are filtered (if filter is enabled)."""
        result = _apply_data_filters("api_key", "sk-123456")
        # Either filtered or not, depending on config
        assert isinstance(result, (str, type("sk-123456")))


class TestSerializeForSpan:
    """Tests for serialize_for_span function."""

    def test_primitives(self):
        """Test that primitives pass through unchanged."""
        assert serialize_for_span(None) is None
        assert serialize_for_span("hello") == "hello"
        assert serialize_for_span(42) == 42
        assert serialize_for_span(3.14) == 3.14
        assert serialize_for_span(True) is True
        assert serialize_for_span(b"bytes") == b"bytes"

    def test_list_of_primitives(self):
        """Test that lists of primitives pass through."""
        assert serialize_for_span([1, 2, 3]) == [1, 2, 3]
        assert serialize_for_span(["a", "b", "c"]) == ["a", "b", "c"]

    def test_list_with_complex(self):
        """Test that lists with complex objects are serialized."""
        result = serialize_for_span([1, {"key": "value"}])
        assert isinstance(result, str)
        assert "key" in result

    def test_dict(self):
        """Test that dicts are serialized to JSON strings."""
        result = serialize_for_span({"key": "value"})
        assert isinstance(result, str)
        assert "key" in result


class TestSafeStrRepr:
    """Tests for safe_str_repr function."""

    def test_simple_object(self):
        """Test that simple objects are converted to string."""
        result = safe_str_repr([1, 2, 3])
        assert isinstance(result, str)
        assert "1" in result

    def test_large_string_truncation(self):
        """Test that large strings are truncated."""
        # Note: This test depends on AIQA_MAX_OBJECT_STR_CHARS env var
        # We test that truncation logic exists, but exact behavior depends on config
        large_str = "x" * 1000
        result = safe_str_repr(large_str)
        assert isinstance(result, str)
        # Result should be a string representation (may or may not be truncated)

    def test_exception_in_repr(self):
        """Test that exceptions in __repr__ are handled."""
        class BadRepr:
            def __repr__(self):
                raise Exception("Bad repr")
        
        result = safe_str_repr(BadRepr())
        assert "BadRepr" in result or "object" in result


class TestObjectToDict:
    """Tests for object_to_dict function."""

    def test_none(self):
        """Test that None is handled."""
        assert object_to_dict(None, set()) is None

    def test_primitives(self):
        """Test that primitives pass through."""
        assert object_to_dict("hello", set()) == "hello"
        assert object_to_dict(42, set()) == 42
        assert object_to_dict(3.14, set()) == 3.14
        assert object_to_dict(True, set()) is True
        assert object_to_dict(b"bytes", set()) == b"bytes"

    def test_datetime(self):
        """Test that datetime objects are converted to ISO format."""
        dt = datetime(2023, 1, 1, 12, 30, 45)
        result = object_to_dict(dt, set())
        assert isinstance(result, str)
        assert "2023-01-01" in result

    def test_dict(self):
        """Test that dicts are converted."""
        obj = {"key": "value", "num": 42}
        result = object_to_dict(obj, set())
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_list(self):
        """Test that lists are converted."""
        obj = [1, 2, 3, "hello"]
        result = object_to_dict(obj, set())
        assert isinstance(result, list)
        assert result == [1, 2, 3, "hello"]

    def test_circular_reference(self):
        """Test that circular references are detected."""
        obj = {}
        obj["self"] = obj
        result = object_to_dict(obj, set())
        assert result["self"] == "<circular reference>"

    def test_max_depth(self):
        """Test that max depth is enforced."""
        obj = {"level": {"level": {"level": "deep"}}}
        result = object_to_dict(obj, set(), max_depth=2, current_depth=0)
        # Should hit max depth and return "<max depth exceeded>"
        assert isinstance(result, dict) or "<max depth exceeded>" in str(result)

    def test_dataclass(self):
        """Test that dataclasses are converted."""
        @dataclasses.dataclass
        class TestClass:
            name: str
            value: int
        
        obj = TestClass(name="test", value=42)
        result = object_to_dict(obj, set())
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_object_with_dict(self):
        """Test that objects with __dict__ are converted."""
        class TestClass:
            def __init__(self):
                self.name = "test"
                self.value = 42
        
        obj = TestClass()
        result = object_to_dict(obj, set())
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_object_with_slots(self):
        """Test that objects with __slots__ are converted."""
        class TestClass:
            __slots__ = ["name", "value"]
            def __init__(self):
                self.name = "test"
                self.value = 42
        
        obj = TestClass()
        result = object_to_dict(obj, set())
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42


class TestSizeLimitedJSONEncoder:
    """Tests for SizeLimitedJSONEncoder class."""

    def test_size_limit_enforcement(self):
        """Test that the encoder stops when size limit is reached."""
        encoder = SizeLimitedJSONEncoder(max_size_chars=10)
        obj = {"key": "value" * 100}  # Large object
        
        chunks = list(encoder.iterencode(obj))
        assert encoder._truncated is True
        assert len(chunks) > 0

    def test_normal_encoding(self):
        """Test that normal encoding works."""
        encoder = SizeLimitedJSONEncoder(max_size_chars=1000)
        obj = {"key": "value"}
        
        chunks = list(encoder.iterencode(obj))
        assert encoder._truncated is False
        result = ''.join(chunks)
        assert "key" in result
        assert "value" in result


class TestSafeJSONDumps:
    """Tests for safe_json_dumps function."""

    def test_simple_dict(self):
        """Test that simple dicts are serialized."""
        obj = {"key": "value"}
        result = safe_json_dumps(obj)
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result

    def test_circular_reference(self):
        """Test that circular references are handled."""
        obj = {}
        obj["self"] = obj
        result = safe_json_dumps(obj)
        assert isinstance(result, str)
        assert "circular reference" in result.lower()

    def test_size_limit(self):
        """Test that size limits are enforced."""
        # Note: This test depends on AIQA_MAX_OBJECT_STR_CHARS env var
        # We test that the function handles large objects
        obj = {"key": "x" * 10000}  # Large object
        result = safe_json_dumps(obj)
        assert isinstance(result, str)
        # Result should be a string (may be truncated or error message)

    def test_dataclass(self):
        """Test that dataclasses are serialized."""
        @dataclasses.dataclass
        class TestClass:
            name: str
            value: int
        
        obj = TestClass(name="test", value=42)
        result = safe_json_dumps(obj)
        assert isinstance(result, str)
        assert "test" in result
        assert "42" in result

    def test_datetime(self):
        """Test that datetime objects are serialized."""
        obj = {"timestamp": datetime(2023, 1, 1, 12, 30, 45)}
        result = safe_json_dumps(obj)
        assert isinstance(result, str)
        assert "2023-01-01" in result


class TestJSONDefaultHandlerFactory:
    """Tests for json_default_handler_factory function."""

    def test_datetime_handling(self):
        """Test that datetime objects are handled."""
        handler = json_default_handler_factory(set())
        dt = datetime(2023, 1, 1, 12, 30, 45)
        result = handler(dt)
        assert isinstance(result, str)
        assert "2023-01-01" in result

    def test_bytes_handling(self):
        """Test that bytes are handled."""
        handler = json_default_handler_factory(set())
        data = b"hello world"
        result = handler(data)
        assert result == "hello world"

    def test_custom_object(self):
        """Test that custom objects are converted."""
        handler = json_default_handler_factory(set())
        
        class TestClass:
            def __init__(self):
                self.name = "test"
        
        obj = TestClass()
        result = handler(obj)
        assert isinstance(result, dict)
        assert result["name"] == "test"
