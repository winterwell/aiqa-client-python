"""
Span manipulation utilities, trace context management, and server API functions.
"""

import os
import logging
import requests
from typing import Any, Optional, List
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanContext, TraceFlags

from .client import get_aiqa_client, get_component_tag, set_component_tag as _set_component_tag, get_aiqa_tracer
from .constants import LOG_TAG
from .object_serialiser import serialize_for_span
from .http_utils import build_headers, get_server_url, get_api_key

logger = logging.getLogger(LOG_TAG)


async def flush_tracing() -> None:
    """
    Flush all pending spans to the server.
    Flushes also happen automatically every few seconds. So you only need to call this function
    if you want to flush immediately, e.g. before exiting a process.
    A common use is if you are tracing unit tests or experiment runs.

    This flushes the BatchSpanProcessor (OTLP exporter doesn't have a separate flush method).
    """
    client = get_aiqa_client()
    if client.provider:
        client.provider.force_flush()  # Synchronous method


def set_span_attribute(attribute_name: str, attribute_value: Any) -> bool:
    """
    Set an attribute on the active span.
    
    Returns:
        True if attribute was set, False if no active span found
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(attribute_name, serialize_for_span(attribute_value))
        return True
    return False


def set_span_name(span_name: str) -> bool:
    """
    Set the name of the active span.
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.update_name(span_name)
        return True
    return False


def get_active_span() -> Optional[trace.Span]:
    """Get the currently active span."""
    return trace.get_current_span()


def set_conversation_id(conversation_id: str) -> bool:
    """
    Naturally a conversation might span several traces. 
    Set the gen_ai.conversation.id attribute on the active span.
    This allows you to group multiple traces together that are part of the same conversation.
    See https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/ for more details.
    
    Args:
        conversation_id: A unique identifier for the conversation (e.g., user session ID, chat ID, etc.)
    
    Returns:
        True if gen_ai.conversation.id was set, False if no active span found
    
    Example:
        from aiqa import WithTracing, set_conversation_id
        
        @WithTracing
        def handle_user_request(user_id: str, request: dict):
            # Set conversation ID to group all traces for this user session
            set_conversation_id(f"user_{user_id}_session_{request.get('session_id')}")
            # ... rest of function
    """
    return set_span_attribute("gen_ai.conversation.id", conversation_id)


def set_token_usage(
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cached_input_tokens: Optional[int] = None,
) -> bool:
    """
    Set token usage attributes on the active span using OpenTelemetry semantic conventions for gen_ai.
    This allows you to explicitly record token usage information.
    AIQA tracing will automatically detect and set token usage from standard OpenAI-like API responses.
    See https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/ for more details.
    
    Args:
        input_tokens: Number of input tokens used (maps to gen_ai.usage.input_tokens)
        output_tokens: Number of output tokens generated (maps to gen_ai.usage.output_tokens)
        total_tokens: Total number of tokens used (maps to gen_ai.usage.total_tokens)
        cached_input_tokens: Number of cached input tokens used
            (maps to gen_ai.usage.cached_input_tokens)
    Zero is valid (e.g. when the traced function did not call an LLM).
    
    Returns:
        True if at least one token usage attribute was set, False if no active span found
    
    Example:
        from aiqa import WithTracing, set_token_usage
        
        @WithTracing
        def call_llm(prompt: str):
            response = openai_client.chat.completions.create(...)
            # Explicitly set token usage
            set_token_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            return response
        # When there was no LLM call: set_token_usage(0, 0, 0)
    """
    span = trace.get_current_span()
    if not span or not span.is_recording():
        return False
    
    set_count = 0
    try:
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            set_count += 1
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            set_count += 1
        if total_tokens is not None:
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            set_count += 1
        if cached_input_tokens is not None:
            span.set_attribute("gen_ai.usage.cached_input_tokens", cached_input_tokens)
            set_count += 1
    except Exception as e:
        logger.warning(f"Failed to set token usage attributes: {e}")
        return False
    
    return set_count > 0


def set_provider_and_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> bool:
    """
    Set provider and model attributes on the active span using OpenTelemetry semantic conventions for gen_ai.
    This allows you to explicitly record provider and model information.
    AIQA tracing will automatically detect and set provider/model from standard API responses.
    See https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/ for more details.
    
    Args:
        provider: Name of the AI provider (e.g., "openai", "anthropic", "google") (maps to gen_ai.provider.name)
        model: Name of the model used (e.g., "gpt-4", "claude-3-5-sonnet") (maps to gen_ai.request.model)
    
    Returns:
        True if at least one attribute was set, False if no active span found
    
    Example:
        from aiqa import WithTracing, set_provider_and_model
        
        @WithTracing
        def call_llm(prompt: str):
            response = openai_client.chat.completions.create(...)
            # Explicitly set provider and model
            set_provider_and_model(
                provider="openai",
                model=response.model
            )
            return response
    """
    span = trace.get_current_span()
    if not span or not span.is_recording():
        return False
    
    set_count = 0
    try:
        if provider is not None:
            span.set_attribute("gen_ai.provider.name", str(provider))
            set_count += 1
        if model is not None:
            span.set_attribute("gen_ai.request.model", str(model))
            set_count += 1
    except Exception as e:
        logger.warning(f"Failed to set provider/model attributes: {e}")
        return False
    
    return set_count > 0


def set_component_tag(tag: str) -> None:
    """
    Set the component tag that will be added to all spans created by AIQA.
    This can also be set via the AIQA_COMPONENT_TAG environment variable.
    The component tag allows you to identify which component/system generated the spans.
    
    Note: Initialization is automatic when WithTracing is first used. You can also call
    get_aiqa_client() explicitly if needed.
    the client and load environment variables.
    
    Args:
        tag: A component identifier (e.g., "mynamespace.mysystem", "backend.api", etc.)
    
    Example:
        from aiqa import get_aiqa_client, set_component_tag, WithTracing
        
        # Initialize client (loads env vars including AIQA_COMPONENT_TAG)
        get_aiqa_client()
        
        # Or set component tag programmatically (overrides env var)
        set_component_tag("mynamespace.mysystem")
        
        @WithTracing
        def my_function():
            pass
    """
    _set_component_tag(tag)


def get_active_trace_id() -> Optional[str]:
    """
    Get the current trace ID as a hexadecimal string (32 characters).
    
    Returns:
        The trace ID as a hex string, or None if no active span exists.
    
    Example:
        trace_id = get_active_trace_id()
        # Pass trace_id to another service/agent
        # e.g., include in HTTP headers, message queue metadata, etc.
        # Within a single thread, OpenTelemetry normally does this for you.
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_span_id() -> Optional[str]:
    """
    Get the current span ID as a hexadecimal string (16 characters).
    
    Returns:
        The span ID as a hex string, or None if no active span exists.
    
    Example:
        span_id = get_span_id()
        # Can be used to create child spans in other services
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, "016x")
    return None


def create_span_from_trace_id(
    trace_id: str,
    parent_span_id: Optional[str] = None,
    span_name: str = "continued_span",
) -> trace.Span:
    """
    Create a new span that continues from an existing trace ID.
    This is useful for linking traces across different services or agents.
    
    Args:
        trace_id: The trace ID as a hexadecimal string (32 characters)
        parent_span_id: Optional parent span ID as a hexadecimal string (16 characters).
            If provided, the new span will be a child of this span.
        span_name: Name for the new span (default: "continued_span")
    
    Returns:
        A new span that continues the trace. Use it in a context manager or call end() manually.
    
    Example:
        # In service A: get trace ID
        trace_id = get_active_trace_id()
        span_id = get_span_id()
        
        # Send to service B (e.g., via HTTP, message queue, etc.)
        # ...
        
        # In service B: continue the trace
        with create_span_from_trace_id(trace_id, parent_span_id=span_id, span_name="service_b_operation"):
            # Your code here
            pass
    """
    # Parse trace ID from hex string
    trace_id_int = int(trace_id, 16)
    
    # Parse parent span ID if provided
    parent_span_id_int = None
    if parent_span_id:
        parent_span_id_int = int(parent_span_id, 16)
    
    # Create a parent span context
    parent_span_context = SpanContext(
        trace_id=trace_id_int,
        span_id=parent_span_id_int if parent_span_id_int else 0,
        is_remote=True,
        trace_flags=TraceFlags(0x01),  # SAMPLED flag
    )
    
    # Create a context with this span context as the parent
    from opentelemetry.trace import set_span_in_context
    parent_context = set_span_in_context(trace.NonRecordingSpan(parent_span_context))
    
    # Ensure initialization before creating span
    get_aiqa_client()
    # Start a new span in this context (it will be a child of the parent span)
    tracer = get_aiqa_tracer()
    span = tracer.start_span(span_name, context=parent_context)
    
    # Set component tag if configured
    component_tag = get_component_tag()
    if component_tag:
        span.set_attribute("gen_ai.component.id", component_tag)
    
    return span



def inject_trace_context(carrier: dict) -> None:
    """
    Inject the current trace context into a carrier (e.g., HTTP headers).
    This allows you to pass trace context to another service.
    
    Args:
        carrier: Dictionary to inject trace context into (e.g., HTTP headers dict)
    
    Example:
        import requests
        
        headers = {}
        inject_trace_context(headers)
        response = requests.get("http://other-service/api", headers=headers)
    """
    try:
        from opentelemetry.propagate import inject
        inject(carrier)
    except Exception as e:
        logger.warning(f"Error injecting trace context: {e}")


def extract_trace_context(carrier: dict) -> Any:
    """
    Extract trace context from a carrier (e.g., HTTP headers).
    Use this to continue a trace that was started in another service.
    
    Args:
        carrier: Dictionary containing trace context (e.g., HTTP headers dict)
    
    Returns:
        A context object that can be used with trace.use_span() or tracer.start_span()
    
    Example:
        from opentelemetry.trace import use_span
        
        # Extract context from incoming request headers
        ctx = extract_trace_context(request.headers)
        
        # Use the context to create a span
        with use_span(ctx):
            # Your code here
            pass
        
        # Or create a span with the context
        tracer = get_aiqa_tracer()
        with tracer.start_as_current_span("operation", context=ctx):
            # Your code here
            pass
    """
    try:
        from opentelemetry.propagate import extract
        return extract(carrier)
    except Exception as e:
        logger.warning(f"Error extracting trace context: {e}")
        return None


def get_span(span_id: str, organisation_id: Optional[str] = None, exclude: Optional[List[str]] = None) -> Optional[dict]:
    """
    Get a span by its ID from the AIQA server.
    
    Expected usage is: re-playing a specific function call in a unit test (either a developer debugging an issue, or as part of a test suite).
    
    Args:
        span_id: The span ID as a hexadecimal string (16 characters) or client span ID
        organisation_id: Optional organisation ID. If not provided, will try to get from
            AIQA_ORGANISATION_ID environment variable. The organisation is typically
            extracted from the API key during authentication, but the API requires it
            as a query parameter.
        exclude: Optional list of fields to exclude from the span data. By default this function WILL return 'attributes' (often large).
    
    Returns:
        The span data as a dictionary, or None if not found
    
    Example:
        from aiqa import get_span
        
        span = get_span('abc123...')
        if span:
            print(f"Found span: {span['name']}")
            my_function(**span['input'])
    """
    server_url = get_server_url()
    api_key = get_api_key()
    org_id = organisation_id or os.getenv("AIQA_ORGANISATION_ID", "")
    
    # Check if server_url is the default (meaning AIQA_SERVER_URL was not set)
    if not os.getenv("AIQA_SERVER_URL"):
        raise ValueError("AIQA_SERVER_URL is not set. Cannot retrieve span.")    
    if not org_id:
        raise ValueError("Organisation ID is required. Provide it as parameter or set AIQA_ORGANISATION_ID environment variable.")
    if not api_key:
        raise ValueError("API key is required. Set AIQA_API_KEY environment variable.")
    
    # Try both spanId and clientSpanId queries
    for query_field in ["spanId", "clientSpanId"]:
        url = f"{server_url}/span"
        params = {
            "q": f"{query_field}:{span_id}",
            "organisation": org_id,
            "limit": "1",
            "exclude": ",".join(exclude) if exclude else None,
            "fields": "*" if not exclude else None,
        }
        
        headers = build_headers(api_key)
        
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            result = response.json()
            hits = result.get("hits", [])
            if hits and len(hits) > 0:
                return hits[0]
        elif response.status_code == 404:
            # Try next query field
            continue
        else:
            error_text = response.text
            raise ValueError(f"Failed to get span: {response.status_code} - {error_text[:500]}")        
    # not found
    return None


async def submit_feedback(
    trace_id: str,
    thumbs_up: Optional[bool] = None,
    comment: Optional[str] = None,
) -> None:
    """
    Submit feedback for a trace by creating a new span with the same trace ID.
    This allows you to add feedback (thumbs-up, thumbs-down, comment) to a trace after it has completed.
    
    Args:
        trace_id: The trace ID as a hexadecimal string (32 characters)
        thumbs_up: True for positive feedback, False for negative feedback, None for neutral
        comment: Optional text comment
    
    Example:
        from aiqa import submit_feedback
        
        # Submit positive feedback
        await submit_feedback('abc123...', thumbs_up=True, comment='Great response!')
        
        # Submit negative feedback
        await submit_feedback('abc123...', thumbs_up=False, comment='Incorrect answer')
    """
    if not trace_id or len(trace_id) != 32:
        raise ValueError('Invalid trace ID: must be 32 hexadecimal characters')
    
    # Create a span for feedback with the same trace ID
    span = create_span_from_trace_id(trace_id, span_name='feedback')
    
    try:
        # Set feedback attributes
        if thumbs_up is not None:
            span.set_attribute('feedback.value', 'positive' if thumbs_up else 'negative')
        else:
            span.set_attribute('feedback.value', 'neutral')
        
        if comment:
            span.set_attribute('feedback.comment', comment)
        
        # Mark as feedback span
        span.set_attribute('aiqa.span_type', 'feedback')
        
        # End the span
        span.end()
        
        # Flush to ensure it's sent immediately
        await flush_tracing()
    except Exception:
        span.end()
        raise

