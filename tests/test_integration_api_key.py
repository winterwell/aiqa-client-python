"""
Integration test for get_api_key_info functionality.

This test verifies that get_api_key_info works correctly with environment variables
loaded from .env files.

Prerequisites:
- AIQA server must be running and accessible
- Set AIQA_SERVER_URL and AIQA_API_KEY environment variables in .env
- Server must have PostgreSQL configured
- The API key must be valid and associated with an organisation
"""

import os
import pytest
import requests
from typing import Optional
from aiqa.client import get_api_key_info
from aiqa.http_utils import get_server_url, get_api_key, build_headers
from dotenv import load_dotenv

load_dotenv()


def get_api_key_id_from_list() -> Optional[str]:
    """
    Helper function to get an API key ID by listing API keys.
    
    Returns:
        First API key ID found
    """
    
    server_url = get_server_url()
    api_key = get_api_key()
    
    url = f"{server_url}/api-key"
    params = {}
    headers = build_headers(api_key)
    
    response = requests.get(url, params=params, headers=headers, timeout=5)
    if response.status_code == 200:
        api_keys = response.json()
        if api_keys and len(api_keys) > 0:
            return api_keys[0].get("id")
    return None


@pytest.fixture(scope="function")
def check_server_available():
    """Fixture that skips tests if server is not available."""
    server_url = os.getenv("AIQA_SERVER_URL")
    api_key = os.getenv("AIQA_API_KEY")
    
    if not server_url or not api_key:
        pytest.skip("AIQA_SERVER_URL and AIQA_API_KEY environment variables must be set")
    
    # Try to connect to server
    # Any HTTP response (even error codes) means the server is up and responding
    # Only connection errors (timeout, DNS failure, etc.) mean the server is down
    try:
        url = f"{server_url.rstrip('/')}/span"
        headers = build_headers(api_key)
        response = requests.get(
            url,
            params={"q": "name:nonexistent", "limit": "1"},
            headers=headers,
            timeout=5
        )
        # Any HTTP response means server is up (even 400, 404, 500, etc.)
        # We got a response, so server is available
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Cannot connect to server: {e}")


def test_get_api_key_info(check_server_available):
    """Test that get_api_key_info works correctly with environment variables from .env."""
    # Get an API key ID to test with
    api_key_id = get_api_key_id_from_list()
    
    if not api_key_id:
        pytest.skip("AIQA_ORGANISATION_ID must be set to test get_api_key_info, or no API keys found")
    
    # Test get_api_key_info - it should load server_url and api_key from .env
    api_key_info = get_api_key_info(api_key_id)
    print(f"API key info: {api_key_info}")
    
    # Verify the response structure
    assert api_key_info is not None, "API key info should not be None"
    assert "id" in api_key_info, "API key info should have 'id' field"
    assert api_key_info["id"] == api_key_id, f"API key ID should match: expected {api_key_id}, got {api_key_info['id']}"
    
    # Verify other expected fields
    assert "organisation" in api_key_info, "API key info should have 'organisation' field"
    assert "role" in api_key_info, "API key info should have 'role' field"
    assert api_key_info["role"] in ["trace", "developer", "admin"], "API key role should be valid"

