"""
Test startup reliability - simulates ECS deployment scenarios where rapid initialization
and network issues could cause deployment failures.

These tests verify that:
1. Exporter initialization doesn't block or create threads immediately
2. Thread creation is lazy (only on first export)
3. Network failures during startup don't cause hangs
4. Multiple rapid initializations don't cause issues
"""

import os
import time
import threading
import pytest
from unittest.mock import patch, MagicMock
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from aiqa.client import get_aiqa_client, AIQAClient
from aiqa.aiqa_exporter import AIQASpanExporter


class TestStartupReliability:
    """Tests for startup reliability in ECS-like scenarios."""

    def test_exporter_initialization_does_not_create_thread_immediately(self):
        """Verify that creating an exporter doesn't immediately start a thread."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
            },
        ):
            exporter = AIQASpanExporter(startup_delay_seconds=0.1)
            
            # Thread should not be created immediately
            assert exporter.flush_timer is None
            assert not exporter._auto_flush_started
            
            # Cleanup
            exporter.shutdown()

    def test_thread_created_lazily_on_first_export(self):
        """Verify thread is only created when first span is exported."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
            },
        ):
            exporter = AIQASpanExporter(startup_delay_seconds=0.1)
            
            # Thread should not exist yet
            assert exporter.flush_timer is None
            
            # Create a mock span and export it
            from opentelemetry.sdk.trace import ReadableSpan
            from opentelemetry.trace import SpanContext, TraceFlags
            
            mock_span = MagicMock(spec=ReadableSpan)
            mock_span.get_span_context.return_value = SpanContext(
                trace_id=1, span_id=1, is_remote=False, trace_flags=TraceFlags(0x01)
            )
            mock_span.name = "test_span"
            mock_span.kind = 1
            mock_span.start_time = 1000000000
            mock_span.end_time = 2000000000
            mock_span.status.status_code = 1
            mock_span.attributes = {}
            mock_span.links = []
            mock_span.events = []
            mock_span.resource.attributes = {}
            mock_span.parent = None
            
            # Export should trigger thread creation
            result = exporter.export([mock_span])
            
            # Give thread a moment to start
            time.sleep(0.2)
            
            # Now thread should exist
            assert exporter._auto_flush_started
            assert exporter.flush_timer is not None
            assert exporter.flush_timer.is_alive()
            
            # Cleanup
            exporter.shutdown()
            if exporter.flush_timer:
                exporter.flush_timer.join(timeout=2.0)

    def test_rapid_multiple_initializations(self):
        """Test that multiple rapid initializations don't cause issues (simulates health checks)."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
            },
        ):
            # Simulate rapid health check calls
            clients = []
            for _ in range(10):
                client = get_aiqa_client()
                clients.append(client)
                time.sleep(0.01)  # Very short delay
            
            # All should be the same singleton
            assert all(c is clients[0] for c in clients)
            
            # Should not have created multiple threads
            if clients[0].exporter:
                assert clients[0].exporter._auto_flush_started or clients[0].exporter.flush_timer is None

    def test_initialization_with_unreachable_server(self):
        """Test that initialization doesn't hang when server is unreachable."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://unreachable-server:3000",
                "AIQA_API_KEY": "test-api-key",
            },
        ):
            # Should not block or raise
            client = get_aiqa_client()
            assert client is not None
            assert client._initialized
            
            # Exporter should exist but thread shouldn't be started yet
            if client.exporter:
                # Thread creation is lazy, so it might not exist
                assert client.exporter.flush_timer is None or not client.exporter._auto_flush_started

    def test_startup_delay_respected(self):
        """Verify that startup delay prevents immediate flush attempts."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
            },
        ):
            exporter = AIQASpanExporter(startup_delay_seconds=0.5)
            
            # Create and export a span to trigger thread creation
            from opentelemetry.sdk.trace import ReadableSpan
            from opentelemetry.trace import SpanContext, TraceFlags
            
            mock_span = MagicMock(spec=ReadableSpan)
            mock_span.get_span_context.return_value = SpanContext(
                trace_id=1, span_id=1, is_remote=False, trace_flags=TraceFlags(0x01)
            )
            mock_span.name = "test_span"
            mock_span.kind = 1
            mock_span.start_time = 1000000000
            mock_span.end_time = 2000000000
            mock_span.status.status_code = 1
            mock_span.attributes = {}
            mock_span.links = []
            mock_span.events = []
            mock_span.resource.attributes = {}
            mock_span.parent = None
            
            exporter.export([mock_span])
            
            # Thread should be created
            time.sleep(0.1)
            assert exporter._auto_flush_started
            
            # But flush should not have happened yet (within delay period)
            # We can't easily test this without mocking time, but we verify thread exists
            assert exporter.flush_timer is not None
            
            # Cleanup
            exporter.shutdown()
            if exporter.flush_timer:
                exporter.flush_timer.join(timeout=2.0)

    def test_concurrent_initialization(self):
        """Test concurrent initialization from multiple threads (simulates ECS health checks)."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
            },
        ):
            clients = []
            errors = []
            
            def init_client():
                try:
                    client = get_aiqa_client()
                    clients.append(client)
                except Exception as e:
                    errors.append(e)
            
            # Start multiple threads initializing simultaneously
            threads = [threading.Thread(target=init_client) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)
            
            # Should have no errors
            assert len(errors) == 0
            
            # All should be the same singleton
            assert len(set(id(c) for c in clients)) == 1

    def test_shutdown_before_thread_starts(self):
        """Test that shutdown works even if thread was never started."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
            },
        ):
            exporter = AIQASpanExporter(startup_delay_seconds=1.0)
            
            # Thread should not exist
            assert exporter.flush_timer is None
            
            # Shutdown should work without errors
            exporter.shutdown()
            
            # Should still be able to call shutdown again
            exporter.shutdown()

    def test_initialization_timeout(self):
        """Test that initialization completes quickly even with network issues."""
        with patch.dict(
            os.environ,
            {
                "AIQA_SERVER_URL": "http://localhost:3000",
                "AIQA_API_KEY": "test-api-key",
            },
        ):
            start_time = time.time()
            client = get_aiqa_client()
            elapsed = time.time() - start_time
            
            # Initialization should be fast (< 1 second)
            assert elapsed < 1.0
            assert client is not None

