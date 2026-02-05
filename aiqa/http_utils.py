"""
Shared HTTP utilities for AIQA client.
Provides common functions for building headers, handling errors, and accessing environment variables.
Supports AIQA-specific env vars (AIQA_SERVER_URL, AIQA_API_KEY) with fallback to OTLP standard vars.
"""

import gzip
import json
import os
import zlib
from typing import Dict, Optional

# Gzip magic bytes: server may send compressed body without Content-Encoding, so requests won't decompress
_GZIP_MAGIC = b"\x1f\x8b"


def _decompress_deflate(content: bytes) -> bytes:
    """Decompress deflate (try zlib wrapper then raw deflate)."""
    try:
        return zlib.decompress(content)
    except zlib.error:
        return zlib.decompress(content, -zlib.MAX_WBITS)  # raw deflate, no header


def _decompress_br(content: bytes) -> bytes:
    try:
        import brotli
        return brotli.decompress(content)
    except ImportError:
        raise Exception(
            "Response is brotli (br) compressed but the 'brotli' package is not installed. pip install brotli"
        )


def _decompress_content(content: bytes, encoding_hint: str) -> bytes:
    """Decompress content. Uses encoding_hint if set, otherwise tries gzip magic, brotli, deflate."""
    # If header says gzip or body has gzip magic, use gzip (proxy may set header but not compress)
    if encoding_hint == "gzip" or (len(content) >= 2 and content[:2] == _GZIP_MAGIC):
        if len(content) >= 2 and content[:2] == _GZIP_MAGIC:
            return gzip.decompress(content)
        # Header said gzip but body isn't gzip (e.g. proxy stripped body but left header)
        return content
    if encoding_hint == "deflate":
        return _decompress_deflate(content)
    if encoding_hint == "br":
        try:
            return _decompress_br(content)
        except Exception:
            # Header said br but body may be gzip/deflate (e.g. proxy mismatch); try others
            if len(content) >= 2 and content[:2] == _GZIP_MAGIC:
                return gzip.decompress(content)
            try:
                return _decompress_deflate(content)
            except Exception:
                raise
    # No hint: try gzip magic, then brotli, then deflate
    if len(content) >= 2 and content[:2] == _GZIP_MAGIC:
        return gzip.decompress(content)
    try:
        return _decompress_br(content)
    except ImportError:
        pass
    except Exception:
        pass
    try:
        return _decompress_deflate(content)
    except Exception:
        pass
    return content  # not compressed


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
    # Prefer gzip for widest compatibility; br can cause proxy/server mismatches (wrong header or double-compression)
    headers = {
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate",
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


def _response_body_text(response) -> str:
    """Decode response body to str, decompressing using Content-Encoding or by trying gzip/brotli/deflate."""
    content = getattr(response, "content", b"") or b""
    headers = getattr(response, "headers", None) or {}
    encoding = (headers.get("Content-Encoding") or "").strip().lower().split(",")[0].strip()
    if encoding == "identity":
        encoding = ""

    decompressed = _decompress_content(content, encoding)
    try:
        return decompressed.decode("utf-8")
    except UnicodeDecodeError as e:
        raise Exception(
            "Response body is not valid UTF-8 after decompression. "
            "Content-Encoding may not match the actual body."
        ) from e


def parse_json_response(response, operation: str):
    """
    Parse response body as JSON. On empty or invalid JSON, raise a clear error
    including status and body snippet. Handles gzip-compressed bodies when the
    server omits Content-Encoding (so requests doesn't auto-decompress).
    """
    text = _response_body_text(response)
    status_code = getattr(response, "status_code", getattr(response, "status", "unknown"))
    if not text.strip():
        raise Exception(
            f"Server returned {status_code} for {operation} but response body is empty. "
            "Check AIQA_SERVER_URL and that the endpoint returns JSON."
        )
    try:
        return json.loads(text)
    except ValueError as e:
        snippet = text[:500] if len(text) > 500 else text
        raise Exception(
            f"Server returned {status_code} for {operation} but body is not valid JSON: {e}. "
            f"Body snippet: {snippet!r}"
        ) from e
