"""
OpenTelemetry tracing decorator. Provides WithTracing decorator to automatically trace function calls.
"""

import logging
import inspect
import fnmatch
from typing import Any, Callable, Optional, List
from functools import wraps
from opentelemetry import context as otel_context, trace
from opentelemetry.trace import Status, StatusCode

from .client import get_aiqa_client, get_component_tag, get_aiqa_tracer
from .constants import LOG_TAG
from .object_serialiser import serialize_for_span, safe_json_dumps
from .tracing_llm_utils import _extract_and_set_token_usage, _extract_and_set_provider_and_model

logger = logging.getLogger(LOG_TAG)


__all__ = ["WithTracing"]


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
                input data when recording span attributes. Applies after filter_input if both are set. 
                Supports "self" and simple wildcards (e.g., `"_*"` 
                matches `"_apple"`, `"_fruit"`). The pattern `"_*"` is applied by default
                to filter properties starting with '_' in nested objects.
            
            ignore_output: Iterable of keys (e.g., list, set) to exclude from
                output data when recording span attributes. Only applies when
                output is a dictionary. Supports simple wildcards (e.g., `"_*"` 
                matches `"_apple"`, `"_fruit"`). The pattern `"_*"` is applied by default
                to filter properties starting with '_' in nested objects. Useful for excluding 
                large or sensitive fields from traces.
            
            filter_input: Callable function that receives the same arguments as the
                decorated function (*args, **kwargs) and returns a filtered/transformed
                version to be recorded in the span. This allows you to extract specific
                properties from any kind of object, including `self` for methods.
                The function receives the exact same inputs as the decorated function,
                including `self` for bound methods. Returns a dict or any value that
                will be converted to a dict. Applied before ignore_input filtering.
            
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
            
            # Extract properties from self in a method
            class ExperimentRunner:
                def __init__(self, dataset_id, experiment_id):
                    self.dataset_id = dataset_id
                    self.experiment_id = experiment_id
                
                @WithTracing(
                    filter_input=lambda self, example: {
                        "dataset": self.dataset_id,
                        "experiment": self.experiment_id,
                        "example": example.id if hasattr(example, 'id') else None
                    }
                )
                def run_example(self, example):
                    return self.process(example)
        """
        self.name = name
        self.ignore_input = ignore_input
        self.ignore_output = ignore_output
        self.filter_input = filter_input
        self.filter_output = filter_output


def _matches_ignore_pattern(key: str, ignore_patterns: List[str]) -> bool:
    """
    Check if a key matches any pattern in the ignore list.
    Supports simple wildcards (e.g., "_*" matches "_apple", "_fruit").
    """
    for pattern in ignore_patterns:
        if "*" in pattern or "?" in pattern:
            # Use fnmatch for wildcard matching
            if fnmatch.fnmatch(key, pattern):
                return True
        else:
            # Exact match for non-wildcard patterns
            if key == pattern:
                return True
    return False


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


def _apply_ignore_patterns(
    data_dict: dict, 
    ignore_patterns: Optional[List[str]], 
    recursive: bool = True,
    max_depth: int = 100,
    current_depth: int = 0
) -> dict:
    """
    Apply ignore patterns to a dict, optionally recursively.
    Supports string keys, wildcard patterns (*), and list of patterns.
    Used for both ignore_input and ignore_output.
    
    Args:
        data_dict: Dictionary to filter (may contain nested dictionaries)
        ignore_patterns: List of patterns to exclude (e.g., ["self", "_*", "password"])
        recursive: Whether to apply patterns recursively to nested dictionaries
        max_depth: Maximum recursion depth to prevent infinite loops (default: 100)
        current_depth: Current recursion depth (internal use)
    
    Returns:
        Filtered dictionary with matching keys removed
    """
    if not isinstance(data_dict, dict):
        return data_dict
    
    # Safety check: prevent infinite loops from extremely deep nesting
    if current_depth >= max_depth:
        logger.warning(
            f"_apply_ignore_patterns: max depth {max_depth} reached, "
            f"stopping recursion to prevent infinite loop"
        )
        return data_dict
    
    # If no patterns, return copy (no filtering needed, even if recursive=True)
    if not ignore_patterns:
        return data_dict.copy()
    
    result = {}
    for key, value in data_dict.items():
        # Skip keys that match ignore patterns
        if _matches_ignore_pattern(key, ignore_patterns):
            continue
        
        # Recursively process nested dictionaries if recursive=True
        if recursive and isinstance(value, dict):
            result[key] = _apply_ignore_patterns(
                value, ignore_patterns, recursive, max_depth, current_depth + 1
            )
        else:
            result[key] = value
    
    return result


def _merge_with_default_ignore_patterns(
    ignore_patterns: Optional[List[str]], 
    client: Optional[Any] = None
) -> List[str]:
    """
    Merge user-provided ignore patterns with client's default ignore patterns.
    
    Args:
        ignore_patterns: Optional list of user-provided patterns
        client: Optional client instance (to avoid repeated get_aiqa_client() calls)
    
    Returns:
        List of patterns including client's default ignore patterns
    """
    if client is None:
        client = get_aiqa_client()
    default_patterns = client.default_ignore_patterns
    
    if ignore_patterns is None:
        return default_patterns.copy() if default_patterns else []
    
    # Merge patterns, avoiding duplicates
    merged = list(default_patterns)
    for pattern in ignore_patterns:
        if pattern not in merged:
            merged.append(pattern)
    return merged


def _prepare_and_filter_input(
    args: tuple,
    kwargs: dict,
    filter_input: Optional[Callable[[Any], Any]],
    ignore_input: Optional[List[str]],
    sig: Optional[inspect.Signature] = None,
) -> Any:
    """
    Prepare and filter input for span attributes.
    
    Process flow:
    1. Apply filter_input to args, kwargs (receives same inputs as decorated function, including self)
    2. Convert into dict ready for span.attributes.input
    3. Apply ignore_input to the dict (supports string, wildcard, and list patterns)
       Client's default ignore patterns are automatically merged with ignore_input.
    
    Args:
        args: Positional arguments (including self for bound methods)
        kwargs: Keyword arguments
        filter_input: Optional function to filter/transform args and kwargs before conversion.
            Receives *args, **kwargs with the same signature as the function being decorated,
            including `self` for bound methods. This allows extracting properties from any object.
        ignore_input: Optional list of keys/patterns to exclude from the final dict.
            If "self" is in ignore_input, it will be removed from the final dict but filter_input
            still receives it. Client's default ignore patterns are automatically merged.
        sig: Optional function signature for proper arg name resolution
    
    Returns:
        Prepared input data (dict, list, or other) ready for span.attributes.input
    """
    # Step 1: Apply filter_input to args, kwargs (same inputs as decorated function, including self)
    if filter_input:
        # filter_input receives the exact same args/kwargs as the decorated function
        # This allows it to access self and extract properties from any object
        try:
            filtered_result = filter_input(*args, **kwargs)
        except TypeError:
            # Fallback: backward compatibility - convert to dict first
            temp_dict = _prepare_input(args, kwargs, sig)
            filtered_result = filter_input(temp_dict)
        
        # Step 2: Convert filter_input result into dict ready for span.attributes.input
        if isinstance(filtered_result, dict):
            input_data = filtered_result
        else:
            # Convert filter_input result to dict using signature
            # Use original sig (not filtered) since filter_input received all args including self
            input_data = _prepare_input(
                (filtered_result,) if not isinstance(filtered_result, tuple) else filtered_result,
                {},
                sig
            )
    else:
        # Step 2: Convert into dict ready for span.attributes.input
        input_data = _prepare_input(args, kwargs, sig)
    
    # Step 3: Apply ignore_input to the dict (removes "self" from final dict if specified)
    # Merge with client's default ignore patterns
    client = get_aiqa_client()
    merged_ignore_input = _merge_with_default_ignore_patterns(ignore_input, client)
    should_ignore_self = "self" in merged_ignore_input
    
    if isinstance(input_data, dict):
        input_data = _apply_ignore_patterns(
            input_data, 
            merged_ignore_input, 
            recursive=client.ignore_recursive
        )
        # Handle case where we removed self and there are no remaining args/kwargs
        if should_ignore_self and not input_data:
            return None
    elif merged_ignore_input:
        # Warn if ignore patterns are set but input_data is not a dict
        logger.warning(f"_prepare_and_filter_input: skip: ignore patterns are set but input_data is not a dict: {type(input_data)}")
    
    return input_data


def _filter_and_serialize_output(
    result: Any,
    filter_output: Optional[Callable[[Any], Any]],
    ignore_output: Optional[List[str]],
) -> Any:
    """
    Filter and serialize output for span attributes.
    Client's default ignore patterns are automatically merged with ignore_output.
    """
    output_data = result
    if filter_output:
        if isinstance(output_data, dict):
            output_data = output_data.copy() # copy to provide shallow protection against the user accidentally mutating the output with filter_output
        output_data = filter_output(output_data)
    
    # Apply ignore_output patterns (supports key, wildcard, and list patterns)
    # Merge with client's default ignore patterns
    client = get_aiqa_client()
    merged_ignore_output = _merge_with_default_ignore_patterns(ignore_output, client)
    
    if isinstance(output_data, dict):
        output_data = _apply_ignore_patterns(
            output_data, 
            merged_ignore_output,
            recursive=client.ignore_recursive
        )
    elif merged_ignore_output:
        # Warn if ignore patterns are set but output_data is not a dict
        logger.warning(f"_filter_and_serialize_output: skip: ignore patterns are set but output_data is not a dict: {type(output_data)}")
    
    # Serialize immediately to create immutable result (removes mutable structures)
    if isinstance(output_data, (dict, list, tuple)):
        return safe_json_dumps(output_data, strip_private_keys=False)
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
    
    # Prepare, filter, and serialize output (serialization happens in _filter_and_serialize_output)
    output_data = _filter_and_serialize_output(output_data, filter_output, ignore_output)
    if output_data is not None:
        # output_data is already serialized (immutable) from _filter_and_serialize_output
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
    root: bool = False,
):
    """
    Decorator to automatically create spans for function calls.
    Records input/output as span attributes. Spans are automatically linked via OpenTelemetry context.
    
    Works with synchronous functions, asynchronous functions, generator functions, and async generator functions.
    
    Args:
        func: The function to trace (when used as @WithTracing)
        name: Optional custom name for the span (defaults to function name)
        ignore_input: List of keys to exclude from input data when recording span attributes.
            self is handled as "self"
            Supports simple wildcards (e.g., "_*" 
            matches "_apple", "_fruit"). The pattern "_*" is applied by default
            to filter properties starting with '_' in nested objects. For example, use 
            ["password", "api_key"] to exclude additional sensitive fields from being traced.
        ignore_output: List of keys to exclude from output data when recording span attributes.
            Only applies when output is a dictionary. Supports simple wildcards (e.g., "_*" 
            matches "_apple", "_fruit"). The pattern "_*" is applied by default
            to filter properties starting with '_' in nested objects. Useful for excluding 
            large or sensitive fields from traces.
        filter_input: Function to filter/transform input before recording.
            Receives the same arguments as the decorated function (*args, **kwargs),
            including `self` for bound methods. This allows you to extract specific
            properties from any kind of object. For example, to extract `dataset_id`
            from `self` in a method: `filter_input=lambda self, x: {"dataset": self.dataset_id, "x": x}`.
            Returns a dict or any value (will be converted to dict). Applied before ignore_input.
        filter_output: Function to filter/transform output before recording.
            Receives the output value and returns a filtered/transformed version.
        root: Whether this is a root span. If True, the span will not be linked to any parent spans.
    
    Example:
        @WithTracing
        def my_function(x, y):
            return x + y
        
        @WithTracing(name="custom_name")
        def another_function():
            pass
        
        # Extract properties from self in a method
        class MyClass:
            def __init__(self, dataset_id):
                self.dataset_id = dataset_id
            
            @WithTracing(
                filter_input=lambda self, x: {"dataset": self.dataset_id, "x": x}
            )
            def process(self, x):
                return x * 2
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
                if isinstance(input_data, (dict, list, tuple)):
                    span.set_attribute("input", safe_json_dumps(input_data, strip_private_keys=False))
                else:
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
            span_kw = {"context": otel_context.Context()} if root else {}
            with tracer.start_as_current_span(fn_name, **span_kw) as span:
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
            span_kw = {"context": otel_context.Context()} if root else {}
            with tracer.start_as_current_span(fn_name, **span_kw) as span:
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
            span_kw = {"context": otel_context.Context()} if root else {}
            span = tracer.start_span(fn_name, **span_kw)
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
                # executor() returns an async generator object, not a coroutine, so don't await it
                return executor()
            
            # Get tracer after initialization (lazy)
            tracer = get_aiqa_tracer()
            # Create span but don't use 'with' - span will be closed by TracedAsyncGenerator
            span_kw = {"context": otel_context.Context()} if root else {}
            span = tracer.start_span(fn_name, **span_kw)
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


