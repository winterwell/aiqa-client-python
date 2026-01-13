"""
Shared HTTP utilities for AIQA client.
Provides common functions for building headers, handling errors, and accessing environment variables.
Supports AIQA-specific env vars (AIQA_SERVER_URL, AIQA_API_KEY) with fallback to OTLP standard vars.
"""

import os
from typing import Dict, Optional


def build_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Build HTTP headers for AIQA API requests.
    
    Checks AIQA_API_KEY first, then falls back to OTEL_EXPORTER_OTLP_HEADERS if not set.
    
    Args:
        api_key: Optional API key. If not provided, will try to get from AIQA_API_KEY env var,
                 then from OTEL_EXPORTER_OTLP_HEADERS.
    
    Returns:
        Dictionary with Content-Type, Accept-Encoding, and optionally Authorization header.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate, br",  # Request compression (aiohttp handles decompression automatically)
    }
    
    # Check parameter first
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
        return headers
    
    # Check AIQA_API_KEY env var
    aiqa_api_key = os.getenv("AIQA_API_KEY")
    if aiqa_api_key:
        headers["Authorization"] = f"ApiKey {aiqa_api_key}"
        return headers
    
    # Fallback to OTLP headers (format: "key1=value1,key2=value2")
    otlp_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    if otlp_headers:
        # Parse comma-separated key=value pairs
        for header_pair in otlp_headers.split(","):
            header_pair = header_pair.strip()
            if "=" in header_pair:
                key, value = header_pair.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.lower() == "authorization":
                    headers["Authorization"] = value
                else:
                    headers[key] = value
    
    return headers


def get_server_url(server_url: Optional[str] = None) -> str:
    """
    Get server URL from parameter or environment variable, with trailing slash removed.
    
    Checks AIQA_SERVER_URL first, then falls back to OTEL_EXPORTER_OTLP_ENDPOINT if not set.
    
    Args:
        server_url: Optional server URL. If not provided, will get from AIQA_SERVER_URL env var,
                    then from OTEL_EXPORTER_OTLP_ENDPOINT.
    
    Returns:
        Server URL with trailing slash removed. Defaults to https://server-aiqa.winterwell.com if not set.
    """
    # Check parameter first
    if server_url:
        return server_url.rstrip("/")
    
    # Check AIQA_SERVER_URL env var
    url = os.getenv("AIQA_SERVER_URL")
    if url:
        return url.rstrip("/")
    
    # Fallback to OTLP endpoint
    url = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if url:
        return url.rstrip("/")
    
    # Default fallback
    return "https://server-aiqa.winterwell.com"


def get_api_key(api_key: Optional[str] = None) -> str:
    """
    Get API key from parameter or environment variable.
    
    Checks AIQA_API_KEY first, then falls back to OTEL_EXPORTER_OTLP_HEADERS if not set.
    
    Args:
        api_key: Optional API key. If not provided, will get from AIQA_API_KEY env var,
                 then from OTEL_EXPORTER_OTLP_HEADERS (looking for Authorization header).
    
    Returns:
        API key or empty string if not set.
    """
    # Check parameter first
    if api_key:
        return api_key
    
    # Check AIQA_API_KEY env var
    aiqa_api_key = os.getenv("AIQA_API_KEY")
    if aiqa_api_key:
        return aiqa_api_key
    
    # Fallback to OTLP headers (look for Authorization header)
    otlp_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    if otlp_headers:
        for header_pair in otlp_headers.split(","):
            header_pair = header_pair.strip()
            if "=" in header_pair:
                key, value = header_pair.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.lower() == "authorization":
                    # Extract API key from "ApiKey <key>" or just return the value
                    if value.startswith("ApiKey "):
                        return value[7:]
                    return value
    
    return ""


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
