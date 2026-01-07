"""
OpenTelemetry tracing setup and utilities. Initializes tracer provider on import.
Provides WithTracing decorator to automatically trace function calls.
"""

import json
import logging
import inspect
import os
import copy
import requests
from typing import Any, Callable, Optional, List
from functools import wraps
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode, SpanContext, TraceFlags
from opentelemetry.propagate import inject, extract
from .aiqa_exporter import AIQASpanExporter
from .client import get_aiqa_client, get_component_tag, set_component_tag as _set_component_tag, get_aiqa_tracer
from .constants import AIQA_TRACER_NAME, LOG_TAG
from .object_serialiser import serialize_for_span
from .http_utils import build_headers, get_server_url, get_api_key
from .tracing_llm_utils import _extract_and_set_token_usage, _extract_and_set_provider_and_model

logger = logging.getLogger(LOG_TAG)


async def flush_tracing() -> None:
    """
    Flush all pending spans to the server.
    Flushes also happen automatically every few seconds. So you only need to call this function
    if you want to flush immediately, e.g. before exiting a process.
    A common use is if you are tracing unit tests or experiment runs.

    This flushes both the BatchSpanProcessor and the exporter buffer.
    """
    client = get_aiqa_client()
    if client.provider:
        client.provider.force_flush()  # Synchronous method
    if client.exporter:    
        await client.exporter.flush()


# Export provider and exporter accessors for advanced usage

__all__ = [
    "flush_tracing", "WithTracing",
    "set_span_attribute", "set_span_name", "get_active_span",
    "get_active_trace_id", "get_span_id", "create_span_from_trace_id", "inject_trace_context", "extract_trace_context",
    "set_conversation_id", "set_component_tag", "set_token_usage", "set_provider_and_model", "get_span", "submit_feedback"
]


class TracingOptions:
    """
    Options for WithTracing decorator.
    
    This class is used to configure how function calls are traced and what data
    is recorded in span attributes. All fields are optional.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        ignore_input: Optional[List[str]] = None,
        ignore_output: Optional[List[str]] = None,
        filter_input: Optional[Callable[[Any], Any]] = None,
        filter_output: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Initialize TracingOptions.
        
        Args:
            name: Custom name for the span. If not provided, the function name
                will be used. Useful for renaming spans or providing more
                descriptive names.
            
            ignore_input: Iterable of keys (e.g., list, set) to exclude from
                input data when recording span attributes. Only applies when
                input is a dictionary. For example, use `["password", "api_key"]`
                to exclude sensitive fields from being traced.
            
            ignore_output: Iterable of keys (e.g., list, set) to exclude from
                output data when recording span attributes. Only applies when
                output is a dictionary. Useful for excluding large or sensitive
                fields from traces.
            
            filter_input: Callable function that receives the prepared input data
                and returns a filtered/transformed version to be recorded in the
                span. The function should accept one argument (the input data)
                and return the transformed data. This is applied before
                ignore_input filtering.
            
            filter_output: Callable function that receives the output data and
                returns a filtered/transformed version to be recorded in the span.
                The function should accept one argument (the output data) and
                return the transformed data. This is applied before
                ignore_output filtering.
        
        Example:
            # Exclude sensitive fields from input
            @WithTracing(ignore_input=["password", "secret_key"])
            def authenticate(username, password):
                return {"token": "..."}
            
            # Custom span name and filter output
            @WithTracing(
                name="data_processing",
                filter_output=lambda x: {"count": len(x)} if isinstance(x, list) else x
            )
            def process_data(items):
                return items
        """
        self.name = name
        self.ignore_input = ignore_input
        self.ignore_output = ignore_output
        self.filter_input = filter_input
        self.filter_output = filter_output


def _prepare_input(args: tuple, kwargs: dict, sig: Optional[inspect.Signature] = None) -> Any:
    """Prepare input for span attributes. 
    Converts args and kwargs into a unified dict structure using function signature when available.
    Falls back to legacy behavior for functions without inspectable signatures.
    
    Note: This function does NOT serialize values - it just structures the data.
    Serialization happens later via serialize_for_span() to avoid double-encoding
    (e.g., converting messages to JSON string, then encoding that string again).
    """
    if not args and not kwargs:
        return None
    
    # Try to bind args to parameter names using function signature
    if sig is not None:
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            # Return dict of all arguments (positional args are now named)
            result = bound.arguments.copy()
            # Shallow copy to protect against mutating the input
            return result
        except (TypeError, ValueError):
            # Binding failed (e.g., wrong number of args, *args/**kwargs issues)
            # Fall through to legacy behavior
            pass
    
    # in case binding fails
    if not kwargs:
        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, dict): # shallow copy to protect against mutating the input
                return arg0.copy()
            return arg0
        return list(args)
    if kwargs and len(args) == 0:
        return kwargs.copy() # shallow copy to protect against mutating the input
    # Multiple args and kwargs - combine into dict
    result = kwargs.copy()
    result["args"] = list(args)
    return result


def _prepare_and_filter_input(
    args: tuple,
    kwargs: dict,
    filter_input: Optional[Callable[[Any], Any]],
    ignore_input: Optional[List[str]],
    sig: Optional[inspect.Signature] = None,
) -> Any:
    """
    Prepare and filter input for span attributes - applies the user's filter_input and ignore_input.
    Converts all args to a dict using function signature when available.
    """
    # Handle "self" in ignore_input by skipping the first argument
    filtered_args = args
    filtered_kwargs = kwargs.copy() if kwargs else {}
    filtered_ignore_input = ignore_input
    filtered_sig = sig
    if ignore_input and "self" in ignore_input:
        # Remove "self" from ignore_input list (we'll handle it specially)
        filtered_ignore_input = [key for key in ignore_input if key != "self"]
        # Skip first arg if it exists (typically self for bound methods)
        if args:
            filtered_args = args[1:]
        # Also remove "self" from kwargs if present
        if "self" in filtered_kwargs:
            del filtered_kwargs["self"]
        # Adjust signature to remove "self" parameter if present
        # This is needed because we removed self from args, so signature binding will fail otherwise
        if filtered_sig is not None:
            params = list(filtered_sig.parameters.values())
            if params and params[0].name == "self":
                filtered_sig = filtered_sig.replace(parameters=params[1:])
    # turn args, kwargs into one "nice" object (now always a dict when signature is available)
    input_data = _prepare_input(filtered_args, filtered_kwargs, filtered_sig)
    if filter_input and input_data is not None:
        input_data = filter_input(input_data)
    if filtered_ignore_input and len(filtered_ignore_input) > 0:
        if not isinstance(input_data, dict):
            logger.warning(f"_prepare_and_filter_input: skip: ignore_input is set beyond 'self': {filtered_ignore_input} but input_data is not a dict: {type(input_data)}")
        else:
            for key in filtered_ignore_input:
                if key in input_data:
                    del input_data[key]
    # Also handle case where input_data is just self (single value, not dict)
    # If we filtered out self and there are no remaining args/kwargs, return None
    if ignore_input and "self" in ignore_input and not filtered_args and not filtered_kwargs:
        return None
    return input_data


def _filter_and_serialize_output(
    result: Any,
    filter_output: Optional[Callable[[Any], Any]],
    ignore_output: Optional[List[str]],
) -> Any:
    """Filter and serialize output for span attributes."""
    output_data = result
    if filter_output:
        if isinstance(output_data, dict):
            output_data = output_data.copy() # copy to provide shallow protection against the user accidentally mutating the output with filter_output
        output_data = filter_output(output_data)
    if ignore_output and isinstance(output_data, dict):
        output_data = output_data.copy()
        for key in ignore_output:
            if key in output_data:
                del output_data[key]
    
    # Serialize immediately to create immutable result (removes mutable structures)
    return serialize_for_span(output_data)


def _handle_span_exception(span: trace.Span, exception: Exception) -> None:
    """Record exception on span and set error status."""
    logger.info(f"span end: Handling span exception for {span.name}")
    error = exception if isinstance(exception, Exception) else Exception(str(exception))
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))



def _finalize_span_success_common(
    span: trace.Span,
    result_for_metadata: Any,
    output_data: Any,
    filter_output: Optional[Callable[[Any], Any]] = None,
    ignore_output: Optional[List[str]] = None,
) -> None:
    """
    Common logic for finalizing a span with success status.
    Extracts token usage and provider/model from result, sets output attribute, and sets status to OK.
    
    Serializes output immediately to capture its state when the function returns,
    preventing mutations from affecting the trace.
    
    Args:
        span: The span to finalize
        result_for_metadata: Value to extract token usage and provider/model from
        output_data: The output data to set on the span (will be filtered if needed)
        filter_output: Optional function to filter output data
        ignore_output: Optional list of keys to exclude from output
    """
    logger.info(f"span end: Finalizing for {span.name}")
    _extract_and_set_token_usage(span, result_for_metadata)
    _extract_and_set_provider_and_model(span, result_for_metadata)
    
    # Prepare, filter, and serialize output (serialization happens in _prepare_and_filter_output)
    output_data = _filter_and_serialize_output(output_data, filter_output, ignore_output)
    if output_data is not None:
        # output_data is already serialized (immutable) from _prepare_and_filter_output
        span.set_attribute("output", output_data)
    span.set_status(Status(StatusCode.OK))


class TracedGenerator:
    """Wrapper for sync generators that traces iteration."""
    
    def __init__(
        self,
        generator: Any,
        span: trace.Span,
        fn_name: str,
        filter_output: Optional[Callable[[Any], Any]],
        ignore_output: Optional[List[str]],
        context_token: Any,
    ):
        self._generator = generator
        self._span = span
        self._fn_name = fn_name
        self._filter_output = filter_output
        self._ignore_output = ignore_output
        self._context_token = context_token
        self._yielded_values = []
        self._exhausted = False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._exhausted:
            raise StopIteration
        
        try:
            value = next(self._generator)
            # Serialize immediately to create immutable result (removes mutable structures)
            self._yielded_values.append(serialize_for_span(value))
            return value
        except StopIteration:
            self._exhausted = True
            self._finalize_span_success()
            trace.context_api.detach(self._context_token)
            self._span.end()
            raise
        except Exception as exception:
            self._exhausted = True
            _handle_span_exception(self._span, exception)
            trace.context_api.detach(self._context_token)
            self._span.end()
            raise
    
    def _finalize_span_success(self):
        """Set output and success status on span."""
        # Check last yielded value for token usage (common pattern in streaming responses)
        result_for_metadata = self._yielded_values[-1] if self._yielded_values else None
        
        # Record summary of yielded values
        output_data = {
            "type": "generator",
            "yielded_count": len(self._yielded_values),
        }
        
        # Optionally include sample values (limit to avoid huge spans)
        if self._yielded_values:
            sample_size = min(10, len(self._yielded_values))
            output_data["sample_values"] = [
                serialize_for_span(v) for v in self._yielded_values[:sample_size]
            ]
            if len(self._yielded_values) > sample_size:
                output_data["truncated"] = True
        
        _finalize_span_success_common(
            self._span,
            result_for_metadata,
            output_data,
            self._filter_output,
            self._ignore_output,
        )


class TracedAsyncGenerator:
    """Wrapper for async generators that traces iteration."""
    
    def __init__(
        self,
        generator: Any,
        span: trace.Span,
        fn_name: str,
        filter_output: Optional[Callable[[Any], Any]],
        ignore_output: Optional[List[str]],
        context_token: Any,
    ):
        self._generator = generator
        self._span = span
        self._fn_name = fn_name
        self._filter_output = filter_output
        self._ignore_output = ignore_output
        self._context_token = context_token
        self._yielded_values = []
        self._exhausted = False
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self._exhausted:
            raise StopAsyncIteration
        
        try:
            value = await self._generator.__anext__()
            # Serialize immediately to create immutable result (removes mutable structures)
            self._yielded_values.append(serialize_for_span(value))
            return value
        except StopAsyncIteration:
            self._exhausted = True
            self._finalize_span_success()
            trace.context_api.detach(self._context_token)
            self._span.end()
            raise
        except Exception as exception:
            self._exhausted = True
            _handle_span_exception(self._span, exception)
            trace.context_api.detach(self._context_token)
            self._span.end()
            raise
    
    def _finalize_span_success(self):
        """Set output and success status on span."""
        # Check last yielded value for token usage (common pattern in streaming responses)
        result_for_metadata = self._yielded_values[-1] if self._yielded_values else None
        
        # Record summary of yielded values
        output_data = {
            "type": "async_generator",
            "yielded_count": len(self._yielded_values),
        }
        
        # Optionally include sample values (limit to avoid huge spans)
        if self._yielded_values:
            sample_size = min(10, len(self._yielded_values))
            output_data["sample_values"] = [
                serialize_for_span(v) for v in self._yielded_values[:sample_size]
            ]
            if len(self._yielded_values) > sample_size:
                output_data["truncated"] = True
        
        _finalize_span_success_common(
            self._span,
            result_for_metadata,
            output_data,
            self._filter_output,
            self._ignore_output,
        )


def WithTracing(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    ignore_input: Optional[List[str]] = None,
    ignore_output: Optional[List[str]] = None,
    filter_input: Optional[Callable[[Any], Any]] = None,
    filter_output: Optional[Callable[[Any], Any]] = None,
):
    """
    Decorator to automatically create spans for function calls.
    Records input/output as span attributes. Spans are automatically linked via OpenTelemetry context.
    
    Works with synchronous functions, asynchronous functions, generator functions, and async generator functions.
    
    Args:
        func: The function to trace (when used as @WithTracing)
        name: Optional custom name for the span (defaults to function name)
        ignore_input: List of keys to exclude from input data when recording span attributes.
            Only applies when input is a dictionary. For example, use ["password", "api_key"]
            to exclude sensitive fields from being traced.
        ignore_output: List of keys to exclude from output data when recording span attributes.
            Only applies when output is a dictionary. Useful for excluding large or sensitive
            fields from traces.
        filter_input: Function to filter/transform input before recording
        filter_output: Function to filter/transform output before recording
    
    Example:
        @WithTracing
        def my_function(x, y):
            return x + y
        
        @WithTracing(name="custom_name")
        def another_function():
            pass
    """
    def decorator(fn: Callable) -> Callable:
        fn_name = name or fn.__name__ or "_"
        
        # Check if already traced
        if hasattr(fn, "_is_traced"):
            logger.warning(f"Function {fn_name} is already traced, skipping tracing again")
            return fn
        logger.info(f"WithTracing function {fn_name}")
        is_async = inspect.iscoroutinefunction(fn)
        is_generator = inspect.isgeneratorfunction(fn)
        is_async_generator = inspect.isasyncgenfunction(fn) if hasattr(inspect, 'isasyncgenfunction') else False
        
        # Get function signature once at decoration time for efficient arg name resolution
        fn_sig: Optional[inspect.Signature] = None
        try:
            fn_sig = inspect.signature(fn)
        except (ValueError, TypeError):
            # Some callables (e.g., builtins, C extensions) don't have inspectable signatures
            # Will fall back to legacy behavior
            pass
        
        # Don't get tracer here - get it lazily when function is called
        # This ensures initialization only happens when tracing is actually used
        
        def _setup_span(span: trace.Span, input_data: Any) -> bool:
            """
            Setup span with input data. Returns True if span is recording.
            
            Serializes input immediately to capture its state at function start,
            preventing mutations from affecting the trace.
            """
            if not span.is_recording():
                logger.warning(f"Span {fn_name} is not recording - will not be exported")
                return False
            
            logger.debug(f"Span {fn_name} is recording, trace_id={format(span.get_span_context().trace_id, '032x')}")
            
            # Set component tag if configured
            component_tag = get_component_tag()
            if component_tag:
                span.set_attribute("gen_ai.component.id", component_tag)
            
            if input_data is not None:
                # Serialize input immediately to capture state at function start
                # input_data has already been copied in _prepare_and_filter_input
                span.set_attribute("input", serialize_for_span(input_data))
            
            trace_id = format(span.get_span_context().trace_id, "032x")
            logger.debug(f"do traceable stuff {fn_name} {trace_id}")
            return True
        
        def _finalize_span_success(span: trace.Span, result: Any) -> None:
            """Set output and success status on span."""
            _finalize_span_success_common(
                span,
                result,
                result,
                filter_output,
                ignore_output,
            )
        
        def _execute_with_span_sync(executor: Callable[[], Any], input_data: Any) -> Any:
            """Execute sync function within span context, handling input/output and exceptions.
            Note: input_data has already gone through _prepare_and_filter_input
            """
            # Ensure tracer provider is initialized before creating spans
            # This is called lazily when the function runs, not at decorator definition time
            client = get_aiqa_client()
            if not client.enabled:
                return executor()
            # Get tracer after initialization (lazy)
            tracer = get_aiqa_tracer()
            with tracer.start_as_current_span(fn_name) as span:
                if not _setup_span(span, input_data):
                    return executor() # span is not recording, so just execute the function and return the result                
                try:
                    result = executor()
                    _finalize_span_success(span, result)
                    return result
                except Exception as exception:
                    _handle_span_exception(span, exception)
                    raise
        
        async def _execute_with_span_async(executor: Callable[[], Any], input_data: Any) -> Any:
            """Execute async function within span context, handling input/output and exceptions."""
            # Ensure tracer provider is initialized before creating spans
            # This is called lazily when the function runs, not at decorator definition time
            client = get_aiqa_client()
            if not client.enabled:
                return await executor()
            
            # Get tracer after initialization (lazy)
            tracer = get_aiqa_tracer()
            with tracer.start_as_current_span(fn_name) as span:
                if not _setup_span(span, input_data):
                    return await executor()
                
                try:
                    result = await executor()
                    _finalize_span_success(span, result)
                    logger.debug(f"Span {fn_name} completed successfully, is_recording={span.is_recording()}")
                    return result
                except Exception as exception:
                    _handle_span_exception(span, exception)
                    raise
                finally:
                    logger.debug(f"Span {fn_name} context exiting, is_recording={span.is_recording()}")
        
        def _execute_generator_sync(executor: Callable[[], Any], input_data: Any) -> Any:
            """Execute sync generator function, returning a traced generator."""
            # Ensure tracer provider is initialized before creating spans
            # This is called lazily when the function runs, not at decorator definition time
            client = get_aiqa_client()
            if not client.enabled:
                return executor()
            
            # Get tracer after initialization (lazy)
            tracer = get_aiqa_tracer()
            # Create span but don't use 'with' - span will be closed by TracedGenerator
            span = tracer.start_span(fn_name)
            token = trace.context_api.attach(trace.context_api.set_span_in_context(span))
            
            try:
                if not _setup_span(span, input_data):
                    generator = executor() # span is not recording, so just execute the function and return the result
                    trace.context_api.detach(token)
                    span.end()
                    return generator
                
                generator = executor()
                return TracedGenerator(generator, span, fn_name, filter_output, ignore_output, token)
            except Exception as exception:
                trace.context_api.detach(token)
                _handle_span_exception(span, exception)
                span.end()
                raise
        
        async def _execute_generator_async(executor: Callable[[], Any], input_data: Any) -> Any:
            """Execute async generator function, returning a traced async generator."""
            # Ensure tracer provider is initialized before creating spans
            # This is called lazily when the function runs, not at decorator definition time
            client = get_aiqa_client()
            if not client.enabled:
                return await executor()
            
            # Get tracer after initialization (lazy)
            tracer = get_aiqa_tracer()
            # Create span but don't use 'with' - span will be closed by TracedAsyncGenerator
            span = tracer.start_span(fn_name)
            token = trace.context_api.attach(trace.context_api.set_span_in_context(span))
            
            try:
                if not _setup_span(span, input_data):
                    generator = executor()
                    trace.context_api.detach(token)
                    span.end()
                    return generator
                
                generator = executor()
                return TracedAsyncGenerator(generator, span, fn_name, filter_output, ignore_output, token)
            except Exception as exception:
                trace.context_api.detach(token)
                _handle_span_exception(span, exception)
                span.end()
                raise
        
        if is_async_generator:
            @wraps(fn)
            async def async_gen_traced_fn(*args, **kwargs):
                input_data = _prepare_and_filter_input(args, kwargs, filter_input, ignore_input, fn_sig)
                return await _execute_generator_async(
                    lambda: fn(*args, **kwargs),
                    input_data
                )
            
            async_gen_traced_fn._is_traced = True
            logger.debug(f"Function {fn_name} is now traced (async generator)")
            return async_gen_traced_fn
        elif is_generator:
            @wraps(fn)
            def gen_traced_fn(*args, **kwargs):
                input_data = _prepare_and_filter_input(args, kwargs, filter_input, ignore_input, fn_sig)
                return _execute_generator_sync(
                    lambda: fn(*args, **kwargs),
                    input_data
                )
            
            gen_traced_fn._is_traced = True
            logger.debug(f"Function {fn_name} is now traced (generator)")
            return gen_traced_fn
        elif is_async:
            @wraps(fn)
            async def async_traced_fn(*args, **kwargs):
                input_data = _prepare_and_filter_input(args, kwargs, filter_input, ignore_input, fn_sig)
                return await _execute_with_span_async(
                    lambda: fn(*args, **kwargs),
                    input_data
                )
            
            async_traced_fn._is_traced = True
            logger.debug(f"Function {fn_name} is now traced (async)")
            return async_traced_fn
        else:
            @wraps(fn)
            def sync_traced_fn(*args, **kwargs):
                input_data = _prepare_and_filter_input(args, kwargs, filter_input, ignore_input, fn_sig)
                return _execute_with_span_sync(
                    lambda: fn(*args, **kwargs),
                    input_data
                )
            
            sync_traced_fn._is_traced = True
            logger.debug(f"Function {fn_name} is now traced (sync)")
            return sync_traced_fn
    
    # Support both @WithTracing and @WithTracing(...) syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


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
    try:
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
    except (ValueError, AttributeError) as e:
        logger.error(f"Error creating span from trace_id: {e}")
        # Ensure initialization before creating span
        get_aiqa_client()
        # Fallback: create a new span
        tracer = get_aiqa_tracer()
        span = tracer.start_span(span_name)
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
            span.set_attribute('feedback.thumbs_up', thumbs_up)
            span.set_attribute('feedback.type', 'positive' if thumbs_up else 'negative')
        else:
            span.set_attribute('feedback.type', 'neutral')
        
        if comment:
            span.set_attribute('feedback.comment', comment)
        
        # Mark as feedback span
        span.set_attribute('aiqa.span_type', 'feedback')
        
        # End the span
        span.end()
        
        # Flush to ensure it's sent immediately
        await flush_tracing()
    except Exception as e:
        span.end()
        raise e

