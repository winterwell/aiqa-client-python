"""
OpenTelemetry tracing setup and utilities. Initializes tracer provider on import.
Provides WithTracing decorator to automatically trace function calls.
"""

import json
import logging
import inspect
import os
from typing import Any, Callable, Optional, List
from functools import wraps
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode, SpanContext, TraceFlags
from opentelemetry.propagate import inject, extract
from .aiqa_exporter import AIQASpanExporter
from .client import get_aiqa_client, get_component_tag, set_component_tag as _set_component_tag, get_aiqa_tracer
from .constants import AIQA_TRACER_NAME
from .object_serialiser import serialize_for_span

logger = logging.getLogger("AIQA")


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




def _prepare_input(args: tuple, kwargs: dict) -> Any:
    """Prepare input for span attributes.
    
    Note: This function does NOT serialize values - it just structures the data.
    Serialization happens later via serialize_for_span() to avoid double-encoding
    (e.g., converting messages to JSON string, then encoding that string again).
    """
    if not args and not kwargs:
        return None
    if len(args) == 1 and not kwargs:
        return args[0]  # Don't serialize here - will be serialized later
    # Multiple args or kwargs - combine into dict
    result = {}
    if args:
        result["args"] = list(args)  # Keep as-is, will be serialized later
    if kwargs:
        result["kwargs"] = dict(kwargs)  # Keep as-is, will be serialized later
    return result


def _prepare_and_filter_input(
    args: tuple,
    kwargs: dict,
    filter_input: Optional[Callable[[Any], Any]],
    ignore_input: Optional[List[str]],
) -> Any:
    """Prepare and filter input for span attributes."""
    # Handle "self" in ignore_input by skipping the first argument
    filtered_args = args
    filtered_kwargs = kwargs.copy() if kwargs else {}
    filtered_ignore_input = ignore_input
    if ignore_input and "self" in ignore_input:
        # Remove "self" from ignore_input list (we'll handle it specially)
        filtered_ignore_input = [key for key in ignore_input if key != "self"]
        # Skip first arg if it exists (typically self for bound methods)
        if args:
            filtered_args = args[1:]
        # Also remove "self" from kwargs if present
        if "self" in filtered_kwargs:
            del filtered_kwargs["self"]
    
    input_data = _prepare_input(filtered_args, filtered_kwargs)
    if filter_input:
        input_data = filter_input(input_data)
    if filtered_ignore_input and isinstance(input_data, dict):
        for key in filtered_ignore_input:
            if key in input_data:
                del input_data[key]
    # Also handle case where input_data is just self (single value, not dict)
    # If we filtered out self and there are no remaining args/kwargs, return None
    if ignore_input and "self" in ignore_input and not filtered_args and not filtered_kwargs:
        return None
    return input_data


def _prepare_and_filter_output(
    result: Any,
    filter_output: Optional[Callable[[Any], Any]],
    ignore_output: Optional[List[str]],
) -> Any:
    """Prepare and filter output for span attributes."""
    output_data = result
    if filter_output:
        output_data = filter_output(output_data)
    if ignore_output and isinstance(output_data, dict):
        output_data = output_data.copy()
        for key in ignore_output:
            if key in output_data:
                del output_data[key]
    return output_data


def _handle_span_exception(span: trace.Span, exception: Exception) -> None:
    """Record exception on span and set error status."""
    error = exception if isinstance(exception, Exception) else Exception(str(exception))
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


def _is_attribute_set(span: trace.Span, attribute_name: str) -> bool:
    """
    Check if an attribute is already set on a span.
    Returns True if the attribute exists, False otherwise.
    Safe against exceptions.
    """
    try:
        # Try multiple ways to access span attributes (SDK spans may store them differently)
        # Check public 'attributes' property
        if hasattr(span, "attributes"):
            attrs = span.attributes
            if attrs and attribute_name in attrs:
                return True
        
        # Check private '_attributes' (common in OpenTelemetry SDK)
        if hasattr(span, "_attributes"):
            attrs = span._attributes
            if attrs and attribute_name in attrs:
                return True
        
        # If we can't find the attribute, assume not set (conservative approach)
        return False
    except Exception:
        # If anything goes wrong, assume not set (conservative approach)
        return False


def _extract_and_set_token_usage(span: trace.Span, result: Any) -> None:
    """
    Extract OpenAI API style token usage from result and add to span attributes
    using OpenTelemetry semantic conventions for gen_ai.
    
    Looks for usage dict with prompt_tokens, completion_tokens, and total_tokens.
    Sets gen_ai.usage.input_tokens, gen_ai.usage.output_tokens, and gen_ai.usage.total_tokens.
    Only sets attributes that are not already set.
    
    This function detects token usage from OpenAI API response patterns:
    - OpenAI Chat Completions API: The 'usage' object contains 'prompt_tokens', 'completion_tokens', and 'total_tokens'.
      See https://platform.openai.com/docs/api-reference/chat/object (usage field)
    - OpenAI Completions API: The 'usage' object contains 'prompt_tokens', 'completion_tokens', and 'total_tokens'.
      See https://platform.openai.com/docs/api-reference/completions/object (usage field)
    
    This function is safe against exceptions and will not derail tracing or program execution.
    """
    try:
        if not span.is_recording():
            return
        
        usage = None
        
        # Check if result is a dict with 'usage' key
        try:
            if isinstance(result, dict):
                usage = result.get("usage")
                # Also check if result itself is a usage dict (OpenAI format)
                if usage is None and all(key in result for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
                    usage = result
                # Also check if result itself is a usage dict (Bedrock format)
                elif usage is None and all(key in result for key in ("input_tokens", "output_tokens")):
                    usage = result
            
            # Check if result has a 'usage' attribute (e.g., OpenAI response object)
            elif hasattr(result, "usage"):
                usage = result.usage
        except Exception:
            # If accessing result properties fails, just return silently
            return
        
        # Extract token usage if found
        if isinstance(usage, dict):
            try:
                # Support both OpenAI format (prompt_tokens/completion_tokens) and Bedrock format (input_tokens/output_tokens)
                prompt_tokens = usage.get("prompt_tokens") or usage.get("PromptTokens")
                completion_tokens = usage.get("completion_tokens") or usage.get("CompletionTokens")
                input_tokens = usage.get("input_tokens") or usage.get("InputTokens")
                output_tokens = usage.get("output_tokens") or usage.get("OutputTokens")
                total_tokens = usage.get("total_tokens") or usage.get("TotalTokens")
                
                # Use Bedrock format if OpenAI format not available
                if prompt_tokens is None:
                    prompt_tokens = input_tokens
                if completion_tokens is None:
                    completion_tokens = output_tokens
                
                # Calculate total_tokens if not provided but we have input and output
                if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                    total_tokens = prompt_tokens + completion_tokens
                
                # Only set attributes that are not already set
                if prompt_tokens is not None and not _is_attribute_set(span, "gen_ai.usage.input_tokens"):
                    span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
                if completion_tokens is not None and not _is_attribute_set(span, "gen_ai.usage.output_tokens"):
                    span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
                if total_tokens is not None and not _is_attribute_set(span, "gen_ai.usage.total_tokens"):
                    span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            except Exception:
                # If setting attributes fails, log but don't raise
                logger.debug(f"Failed to set token usage attributes on span")
    except Exception:
        # Catch any other exceptions to ensure this never derails tracing
        logger.debug(f"Error in _extract_and_set_token_usage")


def _extract_and_set_provider_and_model(span: trace.Span, result: Any) -> None:
    """
    Extract provider and model information from result and add to span attributes
    using OpenTelemetry semantic conventions for gen_ai.
    
    Looks for 'model', 'provider', 'provider_name' fields in the result.
    Sets gen_ai.provider.name and gen_ai.request.model.
    Only sets attributes that are not already set.
    
    This function detects model information from common API response patterns:
    - OpenAI Chat Completions API: The 'model' field is at the top level of the response.
      See https://platform.openai.com/docs/api-reference/chat/object
    - OpenAI Completions API: The 'model' field is at the top level of the response.
      See https://platform.openai.com/docs/api-reference/completions/object
    
    This function is safe against exceptions and will not derail tracing or program execution.
    """
    try:
        if not span.is_recording():
            return
        
        model = None
        provider = None
        
        # Check if result is a dict
        try:
            if isinstance(result, dict):
                model = result.get("model") or result.get("Model")
                provider = result.get("provider") or result.get("Provider") or result.get("provider_name") or result.get("providerName")
            
            # Check if result has attributes (e.g., OpenAI response object)
            elif hasattr(result, "model"):
                model = result.model
            if hasattr(result, "provider"):
                provider = result.provider
            elif hasattr(result, "provider_name"):
                provider = result.provider_name
            elif hasattr(result, "providerName"):
                provider = result.providerName
            
            # Check nested structures (e.g., response.data.model)
            if model is None and hasattr(result, "data"):
                data = result.data
                if isinstance(data, dict):
                    model = data.get("model") or data.get("Model")
                elif hasattr(data, "model"):
                    model = data.model
            
            # Check for model in choices (OpenAI pattern)
            if model is None and isinstance(result, dict):
                choices = result.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        model = first_choice.get("model")
                    elif hasattr(first_choice, "model"):
                        model = first_choice.model
        except Exception:
            # If accessing result properties fails, just return silently
            return
        
        # Set attributes if found and not already set
        try:
            if model is not None and not _is_attribute_set(span, "gen_ai.request.model"):
                # Convert to string if needed
                model_str = str(model) if model is not None else None
                if model_str:
                    span.set_attribute("gen_ai.request.model", model_str)
            
            if provider is not None and not _is_attribute_set(span, "gen_ai.provider.name"):
                # Convert to string if needed
                provider_str = str(provider) if provider is not None else None
                if provider_str:
                    span.set_attribute("gen_ai.provider.name", provider_str)
        except Exception:
            # If setting attributes fails, log but don't raise
            logger.debug(f"Failed to set provider/model attributes on span")
    except Exception:
        # Catch any other exceptions to ensure this never derails tracing
        logger.debug(f"Error in _extract_and_set_provider_and_model")


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
            self._yielded_values.append(value)
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
        if self._yielded_values:
            last_value = self._yielded_values[-1]
            _extract_and_set_token_usage(self._span, last_value)
            _extract_and_set_provider_and_model(self._span, last_value)
        
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
        
        output_data = _prepare_and_filter_output(output_data, self._filter_output, self._ignore_output)
        if output_data is not None:
            self._span.set_attribute("output", serialize_for_span(output_data))
        self._span.set_status(Status(StatusCode.OK))


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
            self._yielded_values.append(value)
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
        if self._yielded_values:
            last_value = self._yielded_values[-1]
            _extract_and_set_token_usage(self._span, last_value)
            _extract_and_set_provider_and_model(self._span, last_value)
        
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
        
        output_data = _prepare_and_filter_output(output_data, self._filter_output, self._ignore_output)
        if output_data is not None:
            self._span.set_attribute("output", serialize_for_span(output_data))
        self._span.set_status(Status(StatusCode.OK))


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
        
        @WithTracing
        async def my_async_function(x, y):
            return x + y
        
        @WithTracing
        def my_generator(n):
            for i in range(n):
                yield i * 2
        
        @WithTracing
        async def my_async_generator(n):
            for i in range(n):
                yield i * 2
        
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
        
        is_async = inspect.iscoroutinefunction(fn)
        is_generator = inspect.isgeneratorfunction(fn)
        is_async_generator = inspect.isasyncgenfunction(fn) if hasattr(inspect, 'isasyncgenfunction') else False
        
        # Don't get tracer here - get it lazily when function is called
        # This ensures initialization only happens when tracing is actually used
        
        def _setup_span(span: trace.Span, input_data: Any) -> bool:
            """Setup span with input data. Returns True if span is recording."""
            if not span.is_recording():
                logger.warning(f"Span {fn_name} is not recording - will not be exported")
                return False
            
            logger.debug(f"Span {fn_name} is recording, trace_id={format(span.get_span_context().trace_id, '032x')}")
            
            # Set component tag if configured
            component_tag = get_component_tag()
            if component_tag:
                span.set_attribute("gen_ai.component.id", component_tag)
            
            if input_data is not None:
                span.set_attribute("input", serialize_for_span(input_data))
            
            trace_id = format(span.get_span_context().trace_id, "032x")
            logger.debug(f"do traceable stuff {fn_name} {trace_id}")
            return True
        
        def _finalize_span_success(span: trace.Span, result: Any) -> None:
            """Set output and success status on span."""
            # Extract and set token usage if present (before filtering output)
            _extract_and_set_token_usage(span, result)
            # Extract and set provider/model if present (before filtering output)
            _extract_and_set_provider_and_model(span, result)
            
            output_data = _prepare_and_filter_output(result, filter_output, ignore_output)
            if output_data is not None:
                span.set_attribute("output", serialize_for_span(output_data))
            span.set_status(Status(StatusCode.OK))
        
        def _execute_with_span_sync(executor: Callable[[], Any], input_data: Any) -> Any:
            """Execute sync function within span context, handling input/output and exceptions."""
            # Ensure tracer provider is initialized before creating spans
            # This is called lazily when the function runs, not at decorator definition time
            client = get_aiqa_client()
            if not client.enabled:
                return executor()
            
            # Get tracer after initialization (lazy)
            tracer = get_aiqa_tracer()
            with tracer.start_as_current_span(fn_name) as span:
                if not _setup_span(span, input_data):
                    return executor()
                
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
                    generator = executor()
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
                input_data = _prepare_and_filter_input(args, kwargs, filter_input, ignore_input)
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
                input_data = _prepare_and_filter_input(args, kwargs, filter_input, ignore_input)
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
                input_data = _prepare_and_filter_input(args, kwargs, filter_input, ignore_input)
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
                input_data = _prepare_and_filter_input(args, kwargs, filter_input, ignore_input)
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
    import os
    import requests
    
    server_url = os.getenv("AIQA_SERVER_URL", "").rstrip("/")
    api_key = os.getenv("AIQA_API_KEY", "")
    org_id = organisation_id or os.getenv("AIQA_ORGANISATION_ID", "")
    
    if not server_url:
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
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"ApiKey {api_key}"
        
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

