"""
Object serialization utilities for converting Python objects to JSON-safe formats.
Handles objects, dataclasses, circular references, and size limits.
"""

import json
import os
import dataclasses
import logging
from datetime import datetime, date, time
from typing import Any, Callable, Set

logger = logging.getLogger("aiqa")

def toNumber(value: str|int|None) -> int:
    """Convert string to number. handling units like g, m, k, (also mb kb gb though these should be avoided)"""
    if value is None:
        return 0
    if isinstance(value, int):
        return value    
    if value.endswith("b"): # drop the b
        value = value[:-1]
    if value.endswith("g"):
        return int(value[:-1]) * 1024 * 1024 * 1024
    elif value.endswith("m"):
        return int(value[:-1]) * 1024 * 1024
    elif value.endswith("k"):
        return int(value[:-1]) * 1024
    return int(value)


# Configurable limit for object string representation (in characters)
AIQA_MAX_OBJECT_STR_CHARS = toNumber(os.getenv("AIQA_MAX_OBJECT_STR_CHARS", "1m"))

# Data filters configuration
def _get_enabled_filters() -> Set[str]:
    """Get set of enabled filter names from AIQA_DATA_FILTERS env var."""
    filters_env = os.getenv("AIQA_DATA_FILTERS", "RemovePasswords, RemoveJWT, RemoveAuthHeaders, RemoveAPIKeys")
    if not filters_env or filters_env.lower() == "false":
        return set()
    return {f.strip() for f in filters_env.split(",") if f.strip()}

_ENABLED_FILTERS = _get_enabled_filters()

def _is_jwt_token(value: Any) -> bool:
    """Check if a value looks like a JWT token (starts with 'eyJ' and has 3 parts separated by dots)."""
    if not isinstance(value, str):
        return False
    # JWT tokens have format: header.payload.signature (3 parts separated by dots)
    # They typically start with 'eyJ' (base64 encoded '{"')
    parts = value.split('.')
    return len(parts) == 3 and value.startswith('eyJ') and all(len(p) > 0 for p in parts)

def _is_api_key(value: Any) -> bool:
    """Check if a value looks like an API key based on common patterns."""
    if not isinstance(value, str):
        return False
    value = value.strip()
    # Common API key prefixes:
    api_key_prefixes = [
        'sk-',    # OpenAI secret key
        'pk-',    # possibly public key
        'AKIA',   # AWS access key
        'ghp_',   # GitHub personal access token
        'gho_',   # GitHub OAuth token
        'ghu_',   # GitHub unidentified token
        'ghs_',   # GitHub SAML token
        'ghr_'    # GitHub refresh token
    ]
    return any(value.startswith(prefix) for prefix in api_key_prefixes)

def _apply_data_filters(key: str, value: Any) -> Any:
    """Apply data filters to a key-value pair based on enabled filters."""
    if not value:  # Don't filter falsy values
        return value
    
    key_lower = str(key).lower()
    
    # RemovePasswords filter: if key contains "password", replace value with "****"
    if "RemovePasswords" in _ENABLED_FILTERS and "password" in key_lower:
        return "****"
    
    # RemoveJWT filter: if value looks like a JWT token, replace with "****"
    if "RemoveJWT" in _ENABLED_FILTERS and _is_jwt_token(value):
        return "****"
    
    # RemoveAuthHeaders filter: if key is "authorization" (case-insensitive), replace value with "****"
    if "RemoveAuthHeaders" in _ENABLED_FILTERS and key_lower == "authorization":
        return "****"
    
    # RemoveAPIKeys filter: if key contains API key patterns or value looks like an API key
    if "RemoveAPIKeys" in _ENABLED_FILTERS:
        # Check key patterns
        api_key_key_patterns = ['api_key', 'apikey', 'api-key', 'apikey']
        if any(pattern in key_lower for pattern in api_key_key_patterns):
            return "****"
        # Check value patterns
        if _is_api_key(value):
            return "****"
    
    return value


def serialize_for_span(value: Any) -> Any:
    """
    Serialize a value for span attributes.
    OpenTelemetry only accepts primitives (bool, str, bytes, int, float) or sequences of those.
    Complex types (dicts, lists, objects) are converted to JSON strings.
    
    Handles objects by attempting to convert them to dicts, with safeguards against:
    - Circular references
    - Unconvertible parts
    - Large objects (size limits)
    """
    # Keep primitives as is (including None)
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    
    # For sequences, check if all elements are primitives
    if isinstance(value, (list, tuple)):
        # If all elements are primitives, return as list
        if all(isinstance(item, (str, int, float, bool, bytes, type(None))) for item in value):
            return list(value)
        # Otherwise serialize to JSON string
        try:
            return safe_json_dumps(value)
        except Exception:
            return str(value)
    
    # For dicts and other complex types, serialize to JSON string
    try:
        return safe_json_dumps(value)
    except Exception:
        # If JSON serialization fails, convert to string
        return safe_str_repr(value)


def safe_str_repr(value: Any) -> str:
    """
    Safely convert a value to string representation.
    Handles objects with __repr__ that might raise exceptions.
    Uses AIQA_MAX_OBJECT_STR_CHARS environment variable (default: 100000) to limit length.
    """
    try:
        # Try __repr__ first (usually more informative)
        repr_str = repr(value)
        # Limit length to avoid huge strings
        if len(repr_str) > AIQA_MAX_OBJECT_STR_CHARS:
            return repr_str[:AIQA_MAX_OBJECT_STR_CHARS] + "... (truncated)"
        return repr_str
    except Exception:
        # Fallback to type name
        try:
            return f"<{type(value).__name__} object>"
        except Exception:
            return "<unknown object>"


def object_to_dict(obj: Any, visited: Set[int], max_depth: int = 10, current_depth: int = 0) -> Any:
    """
    Convert an object to a dictionary representation.
    
    Args:
        obj: The object to convert
        visited: Set of object IDs to detect circular references
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    
    Returns:
        Dictionary representation of the object, or a string if conversion fails
    """
    if current_depth > max_depth:
        return "<max depth exceeded>"
    
    obj_id = id(obj)
    if obj_id in visited:
        return "<circular reference>"
    
    # Handle None
    if obj is None:
        return None
    
    # Handle primitives
    if isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    
    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, time):
        return obj.isoformat()
    
    # Handle dict
    if isinstance(obj, dict):
        visited.add(obj_id)
        try:
            result = {}
            for k, v in obj.items():
                try:
                    key_str = str(k) if not isinstance(k, (str, int, float, bool)) else k
                    filtered_value = _apply_data_filters(key_str, v)
                    result[key_str] = object_to_dict(filtered_value, visited, max_depth, current_depth + 1)
                except Exception as e:
                    # If one key-value pair fails, log and use string representation for the value
                    key_str = str(k) if not isinstance(k, (str, int, float, bool)) else k
                    logger.debug(f"Failed to convert dict value for key '{key_str}': {e}")
                    result[key_str] = safe_str_repr(v)
            visited.remove(obj_id)
            return result
        except Exception as e:
            visited.discard(obj_id)
            logger.debug(f"Failed to convert dict to dict: {e}")
            return safe_str_repr(obj)
    
    # Handle list/tuple
    if isinstance(obj, (list, tuple)):
        visited.add(obj_id)
        try:
            result = []
            for item in obj:
                try:
                    result.append(object_to_dict(item, visited, max_depth, current_depth + 1))
                except Exception as e:
                    # If one item fails, log and use its string representation
                    logger.debug(f"Failed to convert list item {type(item).__name__} to dict: {e}")
                    result.append(safe_str_repr(item))
            visited.remove(obj_id)
            return result
        except Exception as e:
            visited.discard(obj_id)
            logger.debug(f"Failed to convert list/tuple to dict: {e}")
            return safe_str_repr(obj)
    
    # Handle dataclasses
    if dataclasses.is_dataclass(obj):
        visited.add(obj_id)
        try:
            result = {}
            for field in dataclasses.fields(obj):
                try:
                    value = getattr(obj, field.name, None)
                    filtered_value = _apply_data_filters(field.name, value)
                    result[field.name] = object_to_dict(filtered_value, visited, max_depth, current_depth + 1)
                except Exception as e:
                    # If accessing a field fails, log and skip it
                    logger.debug(f"Failed to access field {field.name} on {type(obj).__name__}: {e}")
                    result[field.name] = "<error accessing field>"
            visited.remove(obj_id)
            return result
        except Exception as e:
            visited.discard(obj_id)
            logger.debug(f"Failed to convert dataclass {type(obj).__name__} to dict: {e}")
            return safe_str_repr(obj)
    
    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        visited.add(obj_id)
        try:
            result = {}
            for key, value in obj.__dict__.items():
                # Skip private attributes that start with __
                if not (isinstance(key, str) and key.startswith("__")):
                    filtered_value = _apply_data_filters(key, value)
                    result[key] = object_to_dict(filtered_value, visited, max_depth, current_depth + 1)
            visited.remove(obj_id)
            return result
        except Exception as e:
            visited.discard(obj_id)
            # Log the error for debugging, but still return string representation
            logger.debug(f"Failed to convert object {type(obj).__name__} to dict: {e}")
            return safe_str_repr(obj)
    
    # Handle objects with __slots__
    if hasattr(obj, "__slots__"):
        visited.add(obj_id)
        try:
            result = {}
            for slot in obj.__slots__:
                try:
                    if hasattr(obj, slot):
                        value = getattr(obj, slot, None)
                        filtered_value = _apply_data_filters(slot, value)
                        result[slot] = object_to_dict(filtered_value, visited, max_depth, current_depth + 1)
                except Exception as e:
                    # If accessing a slot fails, log and skip it
                    logger.debug(f"Failed to access slot {slot} on {type(obj).__name__}: {e}")
                    result[slot] = "<error accessing slot>"
            visited.remove(obj_id)
            return result
        except Exception as e:
            visited.discard(obj_id)
            logger.debug(f"Failed to convert slotted object {type(obj).__name__} to dict: {e}")
            return safe_str_repr(obj)
    
    # Fallback: try to get a few common attributes
    try:
        result = {}
        for attr in ["name", "id", "value", "type", "status"]:
            if hasattr(obj, attr):
                value = getattr(obj, attr, None)
                filtered_value = _apply_data_filters(attr, value)
                result[attr] = object_to_dict(filtered_value, visited, max_depth, current_depth + 1)
        if result:
            return result
    except Exception:
        pass
    
    # Final fallback: string representation
    return safe_str_repr(obj)


def safe_json_dumps(value: Any) -> str:
    """
    Safely serialize a value to JSON string with safeguards against:
    - Circular references
    - Large objects (size limits)
    - Unconvertible parts
    
    Args:
        value: The value to serialize

    Uses AIQA_MAX_OBJECT_STR_CHARS environment variable (default: 1000000) to limit length.
    
    Returns:
        JSON string representation
    """    
    max_size_chars = AIQA_MAX_OBJECT_STR_CHARS
    visited: Set[int] = set()
    
    # Convert the entire structure to ensure circular references are detected
    # across the whole object graph
    try:
        converted = object_to_dict(value, visited)
    except Exception as e:
        # If conversion fails, try with a fresh visited set and json default handler
        logger.debug(f"object_to_dict failed for {type(value).__name__}, trying json.dumps with default handler: {e}")
        try:
            json_str = json.dumps(value, default=json_default_handler_factory(set()))
            if len(json_str) > max_size_chars:
                return f"<object {type(value)} too large: {len(json_str)} chars (limit: {max_size_chars} chars) begins: {json_str[:100]}... conversion error: {e}>"
            return json_str
        except Exception as e2:
            logger.debug(f"json.dumps with default handler also failed for {type(value).__name__}: {e2}")
            return safe_str_repr(value)
    
    # Try JSON serialization of the converted structure
    try:
        json_str = json.dumps(converted, default=json_default_handler_factory(set()))
        # Check size
        if len(json_str) > max_size_chars:
            return f"<object {type(value)} too large: {len(json_str)} chars (limit: {max_size_chars} chars) begins: {json_str[:100]}...>"
        return json_str
    except Exception as e:
        logger.debug(f"json.dumps total fail for {type(value).__name__}: {e2}")
        # Final fallback
        return safe_str_repr(value)


def json_default_handler_factory(visited: Set[int]) -> Callable[[Any], Any]:
    """
    Create a JSON default handler with a shared visited set for circular reference detection.
    """
    def handler(obj: Any) -> Any:
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, time):
            return obj.isoformat()
        
        # Handle bytes
        if isinstance(obj, bytes):
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return f"<bytes: {len(obj)} bytes>"
        
        # Try object conversion with the shared visited set
        try:
            return object_to_dict(obj, visited)
        except Exception:
            return safe_str_repr(obj)
    
    return handler


def json_default_handler(obj: Any) -> Any:
    """
    Default handler for JSON serialization of non-serializable objects.
    This is a fallback that creates its own visited set.
    """
    return json_default_handler_factory(set())(obj)

