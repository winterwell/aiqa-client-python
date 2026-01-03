"""
Unit tests for tracing.py functions.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from aiqa.tracing import get_span


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

    def test_get_span_missing_server_url(self):
        """Test that get_span raises ValueError when AIQA_SERVER_URL is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="AIQA_SERVER_URL is not set"):
                get_span("test-span-123")

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
