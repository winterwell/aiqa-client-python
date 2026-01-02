"""
Python client for AIQA server - OpenTelemetry tracing decorators.

Initialization is automatic - you don't need to call get_aiqa_client() explicitly.
The client initializes automatically when WithTracing is first used.

Set environment variables:
    AIQA_SERVER_URL: URL of the AIQA server
    AIQA_API_KEY: API key for authentication
    AIQA_COMPONENT_TAG: Optional component identifier
    AIQA_STARTUP_DELAY_SECONDS: Optional delay before first flush (default: 10s)

Example:
    from dotenv import load_dotenv
    from aiqa import WithTracing
    
    # Load environment variables from .env file (if using one)
    load_dotenv()
    
    # No explicit initialization needed - it happens automatically when used
    @WithTracing
    def my_function():
        return "Hello, AIQA!"
    
    # Call the function - initialization happens on first use
    result = my_function()
"""

from .tracing import (
    WithTracing,
    flush_tracing,
    set_span_attribute,
    set_span_name,
    get_active_span,
    get_active_trace_id,
    get_span_id,
    create_span_from_trace_id,
    inject_trace_context,
    extract_trace_context,
    set_conversation_id,
    set_component_tag,
    get_span,
)
from .client import get_aiqa_client
from .experiment_runner import ExperimentRunner
from .constants import VERSION

__all__ = [
    "WithTracing",
    "flush_tracing",
    "set_span_attribute",
    "set_span_name",
    "get_active_span",
    "get_aiqa_client",
    "ExperimentRunner",
    "get_active_trace_id",
    "get_span_id",
    "create_span_from_trace_id",
    "inject_trace_context",
    "extract_trace_context",
    "set_conversation_id",
    "set_component_tag",
    "get_span",
    "VERSION",
]

