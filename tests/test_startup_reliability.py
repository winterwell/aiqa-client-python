"""
Test startup reliability - simulates ECS deployment scenarios where rapid initialization
and network issues could cause deployment failures.

These tests verify that:
1. Client initialization doesn't block or hang
2. Network failures during startup don't cause hangs
3. Multiple rapid initializations don't cause issues
4. Concurrent initialization is safe
"""

import os
import time
import threading
import pytest
from unittest.mock import patch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from aiqa.client import get_aiqa_client, AIQAClient


class TestStartupReliability:
    """Tests for startup reliability in ECS-like scenarios."""

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

