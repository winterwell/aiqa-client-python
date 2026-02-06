"""
Unit tests for span_helpers.py functions.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from opentelemetry import trace
from opentelemetry.trace import SpanContext, TraceFlags

from aiqa.span_helpers import (
    set_span_attribute,
    set_span_name,
    get_active_span,
    set_conversation_id,
    set_token_usage,
    set_provider_and_model,
    set_component_tag,
    get_active_trace_id,
    get_span_id,
    create_span_from_trace_id,
    inject_trace_context,
    extract_trace_context,
    get_span,
    submit_feedback,
    flush_tracing,
)


class TestSetSpanAttribute:
    """Tests for set_span_attribute function."""

    def test_set_span_attribute_success(self):
        """Test setting attribute on active span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = set_span_attribute("test_key", "test_value")
            
            assert result is True
            mock_span.set_attribute.assert_called_once()

    def test_set_span_attribute_no_span(self):
        """Test set_span_attribute returns False when no active span."""
        with patch('opentelemetry.trace.get_current_span', return_value=None):
            result = set_span_attribute("test_key", "test_value")
            assert result is False

    def test_set_span_attribute_not_recording(self):
        """Test set_span_attribute returns False when span is not recording."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = set_span_attribute("test_key", "test_value")
            assert result is False


class TestSetSpanName:
    """Tests for set_span_name function."""

    def test_set_span_name_success(self):
        """Test setting name on active span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = set_span_name("new_name")
            
            assert result is True
            mock_span.update_name.assert_called_once_with("new_name")

    def test_set_span_name_no_span(self):
        """Test set_span_name returns False when no active span."""
        with patch('opentelemetry.trace.get_current_span', return_value=None):
            result = set_span_name("new_name")
            assert result is False


class TestGetActiveSpan:
    """Tests for get_active_span function."""

    def test_get_active_span(self):
        """Test getting active span."""
        mock_span = MagicMock()
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = get_active_span()
            assert result == mock_span


class TestSetConversationId:
    """Tests for set_conversation_id function."""

    def test_set_conversation_id_success(self):
        """Test setting conversation ID."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = set_conversation_id("conv-123")
            
            assert result is True
            mock_span.set_attribute.assert_called_once()


class TestSetTokenUsage:
    """Tests for set_token_usage function."""

    def test_set_token_usage_all_tokens(self):
        """Test setting all token usage attributes."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = set_token_usage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cached_input_tokens=5,
            )
            
            assert result is True
            assert mock_span.set_attribute.call_count == 4

    def test_set_token_usage_partial(self):
        """Test setting partial token usage."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = set_token_usage(input_tokens=10)
            
            assert result is True
            mock_span.set_attribute.assert_called_once()

    def test_set_token_usage_cached_input_tokens(self):
        """Test setting cached input token usage attribute."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = set_token_usage(cached_input_tokens=42)

            assert result is True
            mock_span.set_attribute.assert_called_once_with(
                "gen_ai.usage.cached_input_tokens", 42
            )

    def test_set_token_usage_no_span(self):
        """Test set_token_usage returns False when no active span."""
        with patch('opentelemetry.trace.get_current_span', return_value=None):
            result = set_token_usage(input_tokens=10)
            assert result is False


class TestSetProviderAndModel:
    """Tests for set_provider_and_model function."""

    def test_set_provider_and_model_both(self):
        """Test setting both provider and model."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = set_provider_and_model(provider="openai", model="gpt-4")
            
            assert result is True
            assert mock_span.set_attribute.call_count == 2

    def test_set_provider_and_model_no_span(self):
        """Test set_provider_and_model returns False when no active span."""
        with patch('opentelemetry.trace.get_current_span', return_value=None):
            result = set_provider_and_model(provider="openai")
            assert result is False


class TestSetComponentTag:
    """Tests for set_component_tag function."""

    def test_set_component_tag(self):
        """Test setting component tag."""
        with patch('aiqa.span_helpers._set_component_tag') as mock_set:
            set_component_tag("test-component")
            mock_set.assert_called_once_with("test-component")


class TestGetActiveTraceId:
    """Tests for get_active_trace_id function."""

    def test_get_active_trace_id_success(self):
        """Test getting active trace ID."""
        mock_span = MagicMock()
        mock_context = MagicMock()
        mock_context.is_valid = True
        mock_context.trace_id = 0x1234567890abcdef1234567890abcdef
        mock_span.get_span_context.return_value = mock_context
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = get_active_trace_id()
            
            assert result == "1234567890abcdef1234567890abcdef"

    def test_get_active_trace_id_no_span(self):
        """Test get_active_trace_id returns None when no active span."""
        with patch('opentelemetry.trace.get_current_span', return_value=None):
            result = get_active_trace_id()
            assert result is None

    def test_get_active_trace_id_invalid_context(self):
        """Test get_active_trace_id returns None when context is invalid."""
        mock_span = MagicMock()
        mock_context = MagicMock()
        mock_context.is_valid = False
        mock_span.get_span_context.return_value = mock_context
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = get_active_trace_id()
            assert result is None


class TestGetSpanId:
    """Tests for get_span_id function."""

    def test_get_span_id_success(self):
        """Test getting span ID."""
        mock_span = MagicMock()
        mock_context = MagicMock()
        mock_context.is_valid = True
        mock_context.span_id = 0x1234567890abcdef
        mock_span.get_span_context.return_value = mock_context
        
        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            result = get_span_id()
            
            assert result == "1234567890abcdef"

    def test_get_span_id_no_span(self):
        """Test get_span_id returns None when no active span."""
        with patch('opentelemetry.trace.get_current_span', return_value=None):
            result = get_span_id()
            assert result is None


class TestCreateSpanFromTraceId:
    """Tests for create_span_from_trace_id function."""

    def test_create_span_from_trace_id_success(self):
        """Test creating span from trace ID."""
        with patch('aiqa.span_helpers.get_aiqa_client') as mock_client:
            with patch('aiqa.span_helpers.get_aiqa_tracer') as mock_tracer:
                with patch('aiqa.span_helpers.get_component_tag', return_value=None):
                    mock_span = MagicMock()
                    mock_tracer.return_value.start_span.return_value = mock_span
                    
                    result = create_span_from_trace_id("1234567890abcdef1234567890abcdef")
                    
                    assert result == mock_span
                    mock_tracer.return_value.start_span.assert_called_once()

    def test_create_span_from_trace_id_with_parent(self):
        """Test creating span from trace ID with parent span ID."""
        with patch('aiqa.span_helpers.get_aiqa_client'):
            with patch('aiqa.span_helpers.get_aiqa_tracer') as mock_tracer:
                with patch('aiqa.span_helpers.get_component_tag', return_value=None):
                    mock_span = MagicMock()
                    mock_tracer.return_value.start_span.return_value = mock_span
                    
                    result = create_span_from_trace_id(
                        "1234567890abcdef1234567890abcdef",
                        parent_span_id="abcdef1234567890"
                    )
                    
                    assert result == mock_span


class TestInjectExtractTraceContext:
    """Tests for trace context injection/extraction."""

    def test_inject_trace_context(self):
        """Test injecting trace context."""
        carrier = {}
        with patch('opentelemetry.propagate.inject') as mock_inject:
            inject_trace_context(carrier)
            mock_inject.assert_called_once_with(carrier)

    def test_extract_trace_context(self):
        """Test extracting trace context."""
        carrier = {}
        with patch('opentelemetry.propagate.extract', return_value="context") as mock_extract:
            result = extract_trace_context(carrier)
            assert result == "context"
            mock_extract.assert_called_once_with(carrier)


class TestGetSpan:
    """Tests for get_span function (server API)."""

    def test_get_span_success(self):
        """Test getting span from server."""
        span_data = {"id": "span-123", "name": "test"}
        mock_response_data = {"hits": [span_data]}

        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-key",
                "AIQA_ORGANISATION_ID": "test-org",
            },
        ):
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data
                mock_get.return_value = mock_response

                result = get_span("span-123")
                assert result == span_data

    def test_get_span_not_found(self):
        """Test get_span returns None when not found."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-key",
                "AIQA_ORGANISATION_ID": "test-org",
            },
        ):
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 404
                mock_get.return_value = mock_response

                result = get_span("nonexistent")
                assert result is None


class TestFlushTracing:
    """Tests for flush_tracing function."""

    @pytest.mark.asyncio
    async def test_flush_tracing_with_provider(self):
        """Test flushing tracing with provider."""
        mock_provider = MagicMock()
        mock_client = MagicMock()
        mock_client.provider = mock_provider
        
        with patch('aiqa.span_helpers.get_aiqa_client', return_value=mock_client):
            await flush_tracing()
            mock_provider.force_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_tracing_no_provider(self):
        """Test flushing tracing without provider."""
        mock_client = MagicMock()
        mock_client.provider = None
        
        with patch('aiqa.span_helpers.get_aiqa_client', return_value=mock_client):
            await flush_tracing()
            # Should not raise error
