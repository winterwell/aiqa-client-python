# Functions for extracting and setting LLM-specific attributes on a span.
import logging
from .constants import LOG_TAG
from opentelemetry import trace
from typing import Any

logger = logging.getLogger(LOG_TAG)


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
    
    Looks for usage dict or object with prompt_tokens, completion_tokens, and total_tokens.
    Sets gen_ai.usage.input_tokens, gen_ai.usage.output_tokens, and gen_ai.usage.total_tokens.
    Only sets attributes that are not already set.
    
    This function detects token usage from OpenAI API response patterns:
    - OpenAI Chat Completions API: The 'usage' object (dict or Usage object) contains 'prompt_tokens', 'completion_tokens', and 'total_tokens'.
      See https://platform.openai.com/docs/api-reference/chat/object (usage field)
    - OpenAI Completions API: The 'usage' object contains 'prompt_tokens', 'completion_tokens', and 'total_tokens'.
      See https://platform.openai.com/docs/api-reference/completions/object (usage field)
    
    Handles both dict and object cases (e.g., OpenAI SDK Usage objects).
    
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
        
        # Extract token usage if found (handle both dict and object cases)
        if usage is not None:
            try:
                # Support both OpenAI format (prompt_tokens/completion_tokens) and Bedrock format (input_tokens/output_tokens)
                # Handle dict case
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens") or usage.get("PromptTokens")
                    completion_tokens = usage.get("completion_tokens") or usage.get("CompletionTokens")
                    input_tokens = usage.get("input_tokens") or usage.get("InputTokens")
                    output_tokens = usage.get("output_tokens") or usage.get("OutputTokens")
                    total_tokens = usage.get("total_tokens") or usage.get("TotalTokens")
                # Handle object case (e.g., OpenAI Usage object)
                else:
                    prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "PromptTokens", None)
                    completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "CompletionTokens", None)
                    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "InputTokens", None)
                    output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "OutputTokens", None)
                    total_tokens = getattr(usage, "total_tokens", None) or getattr(usage, "TotalTokens", None)
                
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
