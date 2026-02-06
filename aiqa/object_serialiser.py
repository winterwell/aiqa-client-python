"""
Object serialization utilities for converting Python objects to JSON-safe formats.
Handles objects, dataclasses, circular references, and size limits.
"""

import json
import os
import dataclasses
import logging
from .constants import LOG_TAG
from datetime import datetime, date, time
from typing import Any, Callable, Optional, Set
from json.encoder import JSONEncoder

logger = logging.getLogger(LOG_TAG)

def sanitize_string_for_utf8(text: Optional[str]) -> Optional[str]:
    """
    Sanitize a string to remove surrogate characters that can't be encoded to UTF-8.
    Surrogate characters (U+D800 to U+DFFF) are invalid in UTF-8 and can cause encoding errors.
    
    Args:
        text: The string to sanitize
        
    Returns:
        A string with surrogate characters replaced by the Unicode replacement character (U+FFFD)
    """
    if text is None:
        return None
    if not isinstance(text, str): # paranoia
        text = str(text)
    try:
        # Try encoding to UTF-8 to check if there are any issues
        text.encode('utf-8')
        return text
    except UnicodeEncodeError:
        # If encoding fails, replace surrogates with replacement character
        # This handles surrogates that can't be encoded
        return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

def toNumber(value: str|int|None) -> int:
    """Convert string to number. handling units like g, m, k, (also mb kb gb though these should be avoided)"""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    # Convert to string if not already
    if not isinstance(value, str):
        value = str(value)
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

def _is_private_key(key: Any) -> bool:
    """Return True if key should be omitted from serialized output (e.g. leading underscore)."""
    if isinstance(key, str):
        return key.startswith("_")
    # Allow non-string keys (e.g. int) through
    return False


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
        api_key_key_patterns = ['api_key', 'apikey', 'api-key']
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
    Complex types (dicts, objects) are converted to JSON strings.
    
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
        # Use short-circuiting loop instead of all() for better performance on large lists
        # Only iterate until we find a non-primitive
        for item in value:
            if not isinstance(item, (str, int, float, bool, bytes, type(None))):
                # Found non-primitive, serialize to JSON string
                try:
                    return safe_json_dumps(value)
                except Exception:
                    return str(value)
        # All elements are primitives, return as list
        return list(value)
    
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
    Uses AIQA_MAX_OBJECT_STR_CHARS env var (default "1m") to limit length.
    Also sanitizes surrogate characters to prevent UTF-8 encoding errors.
    """
    try:
        # Try __repr__ first (usually more informative)
        repr_str = repr(value)
        # Sanitize surrogate characters that can't be encoded to UTF-8
        repr_str = sanitize_string_for_utf8(repr_str)
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


def object_to_dict(
    obj: Any,
    visited: Set[int],
    max_depth: int = 10,
    current_depth: int = 0,
    strip_private_keys: bool = True,
) -> Any:
    """
    Convert an object to a dictionary representation. Applies data filters to the object.
    
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
    
    obj_id = id(obj) # note: id cannot raise exception
    if obj_id in visited:
        return "<circular reference>"
    
    # Handle None
    if obj is None:
        return None
    
    # Handle primitives
    if isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    
    # Handle datetime objects
    if isinstance(obj, datetime) or isinstance(obj, date) or isinstance(obj, time):
        try:
            return obj.isoformat()
        except Exception: # paranoia if isoformat() fails (e.g., invalid datetime state, custom implementation bug)
            return safe_str_repr(obj)
    
    # Handle dict
    if isinstance(obj, dict):
        visited.add(obj_id)
        result = {}
        for k, v in obj.items():
            try:
                key_str = str(k) if not isinstance(k, (str, int, float, bool)) else k
                if strip_private_keys and _is_private_key(key_str):
                    continue
                filtered_value = _apply_data_filters(key_str, v)
                result[key_str] = object_to_dict(
                    filtered_value, visited, max_depth, current_depth + 1, strip_private_keys
                )
            except Exception as e:
                # If one key-value pair fails, log and use string representation for the value
                key_str = str(k) if not isinstance(k, (str, int, float, bool)) else k
                logger.debug(f"Failed to convert dict value for key '{key_str}': {e}")
                result[key_str] = safe_str_repr(v)
        visited.remove(obj_id)
        return result
    
    # Handle list/tuple
    if isinstance(obj, (list, tuple)):
        visited.add(obj_id)
        result = []
        for item in obj:
            try:
                result.append(
                    object_to_dict(item, visited, max_depth, current_depth + 1, strip_private_keys)
                )
            except Exception as e:
                # If one item fails, log and use its string representation
                logger.debug(f"Failed to convert list item {type(item).__name__} to dict: {e}")
                result.append(safe_str_repr(item))
        visited.remove(obj_id)
        return result
    
    # Handle dataclasses
    if dataclasses.is_dataclass(obj):
        visited.add(obj_id)
        try:
            result = {}
            for field in dataclasses.fields(obj):
                if strip_private_keys and _is_private_key(field.name):
                    continue
                try:
                    value = getattr(obj, field.name, None)
                    filtered_value = _apply_data_filters(field.name, value)
                    result[field.name] = object_to_dict(
                        filtered_value, visited, max_depth, current_depth + 1, strip_private_keys
                    )
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
    
    # Handle objects with __dict__ (e.g. ORM models like SQLAlchemy)
    # Strip private keys (e.g. _sa_instance_state) at the source so they never enter the serialized output.
    if hasattr(obj, "__dict__"):
        visited.add(obj_id)
        try:
            obj_dict = obj.__dict__
            filtered_dict = {}
            for k, v in obj_dict.items():
                key_str = str(k) if not isinstance(k, (str, int, float, bool)) else k
                if strip_private_keys and _is_private_key(key_str):
                    continue
                filtered_dict[key_str] = v
            return object_to_dict(
                filtered_dict, visited, max_depth, current_depth, strip_private_keys
            )  # Don't count __dict__ as +1 depth
        except Exception as e:  # paranoia: object_to_dict should never raise an exception
            visited.discard(obj_id)
            logger.debug(f"Failed to convert object {type(obj).__name__} with __dict__ to dict: {e}")
            return safe_str_repr(obj)
    
    # Handle objects with __slots__
    if hasattr(obj, "__slots__"):
        visited.add(obj_id)
        try:
            result = {}
            for slot in obj.__slots__:
                if strip_private_keys and _is_private_key(slot):
                    continue
                try:
                    if hasattr(obj, slot):
                        value = getattr(obj, slot, None)
                        filtered_value = _apply_data_filters(slot, value)
                        result[slot] = object_to_dict(
                            filtered_value, visited, max_depth, current_depth + 1, strip_private_keys
                        )
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
                result[attr] = object_to_dict(
                    filtered_value, visited, max_depth, current_depth + 1, strip_private_keys
                )
        if result:
            return result
    except Exception:
        pass
    
    # Final fallback: string representation
    return safe_str_repr(obj)


class SizeLimitedJSONEncoder(JSONEncoder):
    """
    Custom JSON encoder that stops serialization early when max_size_chars is reached.
    Tracks output length incrementally and stops yielding chunks when limit is exceeded.
    """
    def __init__(self, max_size_chars: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size_chars = max_size_chars
        self.current_length = 0
        self._truncated = False
    
    def iterencode(self, o, _one_shot=False):
        """
        Encode the object incrementally, checking size after each chunk.
        Stops early if max_size_chars is exceeded.
        """
        self.current_length = 0
        self._truncated = False
        
        # Use _one_shot optimization when possible (faster for simple objects)
        # The parent class will determine if _one_shot is safe
        for chunk in super().iterencode(o, _one_shot):
            self.current_length += len(chunk)
            if self.current_length > self.max_size_chars:
                self._truncated = True
                # Stop yielding chunks when limit is exceeded
                break
            yield chunk


def safe_json_dumps(value: Any, strip_private_keys: bool = True) -> str:
    """
    Safely serialize a value to JSON string with safeguards against:
    - Circular references
    - Large objects (size limits)
    - Unconvertible parts
    
    Args:
        value: The value to serialize

    Uses AIQA_MAX_OBJECT_STR_CHARS env var (default "1m" = 1 MiB in chars) to limit length.
    
    Returns:
        JSON string representation
    """    
    max_size_chars = AIQA_MAX_OBJECT_STR_CHARS
    visited: Set[int] = set()
    
    # Convert the entire structure to json-friendy form, and ensure circular references are detected
    # across the whole object graph
    try:
        converted = object_to_dict(value, visited, strip_private_keys=strip_private_keys)
    except Exception as e:
        # Note: object_to_dict is very defensive but can still raise in rare edge cases:
        # - Objects with corrupted type metadata causing isinstance()/hasattr() to fail
        # - Malformed dataclasses causing dataclasses.fields() to raise
        # - Objects where accessing __dict__ or __slots__ triggers descriptors that raise
        logger.debug(f"object_to_dict failed for {type(value).__name__}, using safe_str_repr. Error: {e}")
        return safe_str_repr(value)
    
    # Try JSON serialization of the converted structure with size-limited encoder
    # After object_to_dict(), converted is a plain dict/list with circular refs already
    # converted to "<circular reference>" strings. We use check_circular=True (default)
    # as an additional safety net, though it's redundant since object_to_dict() already
    # handled circular refs. We don't need a default handler here since converted
    # should be JSON-serializable.
    try:
        encoder = SizeLimitedJSONEncoder(
            max_size_chars=max_size_chars,
            check_circular=True,  # Safety net for dict/list circular refs (redundant but harmless)
            ensure_ascii=False
        )
        # Use iterencode to get chunks and check size incrementally
        chunks = []
        for chunk in encoder.iterencode(converted, _one_shot=True):
            chunks.append(chunk)
            if encoder._truncated:
                # Hit the limit, stop early
                json_str = ''.join(chunks)
                return f"<object {type(value)} too large: {len(json_str)} chars (limit: {max_size_chars} chars) begins: {json_str[:100]}...>"
        json_str = ''.join(chunks)
        # Check if truncation occurred (encoder may have stopped after last chunk)
        if encoder._truncated or len(json_str) > max_size_chars:
            return f"<object {type(value)} too large: {len(json_str)} chars (limit: {max_size_chars} chars) begins: {json_str[:100]}...>"
        return json_str
    except Exception as e:
        logger.debug(f"json.dumps total fail for {type(value).__name__}: {e}")
        # Final fallback
        return safe_str_repr(value)
