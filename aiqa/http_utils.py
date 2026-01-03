"""
Shared HTTP utilities for AIQA client.
Provides common functions for building headers, handling errors, and accessing environment variables.
"""

import os
from typing import Dict, Optional


def build_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Build HTTP headers for AIQA API requests.
    
    Args:
        api_key: Optional API key. If not provided, will try to get from AIQA_API_KEY env var.
    
    Returns:
        Dictionary with Content-Type and optionally Authorization header.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    elif os.getenv("AIQA_API_KEY"):
        headers["Authorization"] = f"ApiKey {os.getenv('AIQA_API_KEY')}"
    return headers


def get_server_url(server_url: Optional[str] = None) -> str:
    """
    Get server URL from parameter or environment variable, with trailing slash removed.
    
    Args:
        server_url: Optional server URL. If not provided, will get from AIQA_SERVER_URL env var.
    
    Returns:
        Server URL with trailing slash removed, or empty string if not set.
    """
    url = server_url or os.getenv("AIQA_SERVER_URL", "")
    return url.rstrip("/")


def get_api_key(api_key: Optional[str] = None) -> str:
    """
    Get API key from parameter or environment variable.
    
    Args:
        api_key: Optional API key. If not provided, will get from AIQA_API_KEY env var.
    
    Returns:
        API key or empty string if not set.
    """
    return api_key or os.getenv("AIQA_API_KEY", "")


def format_http_error(response, operation: str) -> str:
    """
    Format an HTTP error message from a response object.
    
    Args:
        response: Response object with status_code, reason, and text attributes
        operation: Description of the operation that failed (e.g., "fetch dataset")
    
    Returns:
        Formatted error message string.
    """
    error_text = response.text if hasattr(response, "text") else "Unknown error"
    status_code = getattr(response, "status_code", getattr(response, "status", "unknown"))
    reason = getattr(response, "reason", "")
    return f"Failed to {operation}: {status_code} {reason} - {error_text}"
