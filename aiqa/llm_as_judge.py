"""
LLM-as-judge scoring functionality for evaluating outputs.
"""

import os
import json
import asyncio
import logging
import requests
from typing import Any, Dict, Optional, Callable, Awaitable
from .constants import LOG_TAG
from .types import MetricResult, Metric, Example, CallLLMType

logger = logging.getLogger(LOG_TAG)

def parse_llm_response(content: str) -> Optional[MetricResult]:
    """
    Parse LLM response content string into score and message.
    
    Args:
        content: Raw content string from LLM response
        
    Returns:
        MetricResult object with score:[0,1], message (optional), and error (optional)
    """
    try:
        result = json.loads(content)
        score = result.get("score")
        if not score:
            return None
        message = result.get("message")
        # Ensure score is in [0, 1] range
        score = max(0.0, min(1.0, float(score)))
        return {"score": score, "message": message}
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Failed to parse JSON response: %s", e)
        raise Exception(f"Could not parse LLM response: {content}")


async def get_model_from_server(
    model_id: str, server_url: str, headers: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Fetch a model from the server with API key included.
    
    Args:
        model_id: ID of the model to fetch
        server_url: URL of the AIQA server
        headers: HTTP headers for authentication
            
    Returns:
        Model dictionary with api_key if available, None if not found or error
    """
    try:
        def _do_request():
            return requests.get(
                f"{server_url}/model/{model_id}?fields=apiKey",  # Server uses camelCase 'apiKey' (also accepts 'api_key')
                headers=headers,
            )
        
        response = await asyncio.to_thread(_do_request)
        if response.ok:
            model = response.json()
            # Server returns 'apiKey' (camelCase)
            if model.get("apiKey"):
                return model
        return None
    except Exception as e:
        logger.warning("Could not fetch model from server: %s", e)
        return None


async def call_openai(
    system_prompt: str, user_message: str, api_key: str, output_schema: str = None
) -> str:
    """Call OpenAI API for LLM-as-judge scoring. Returns raw content string."""
    def _do_request():
        # TODO send output_schema if provided
        return requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "gpt-4o-mini",  # Default model, can be made configurable
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )

    response = await asyncio.to_thread(_do_request)

    if not response.ok:
        error_text = response.text
        raise Exception(
            f"OpenAI API error: {response.status_code} {response.reason} - {error_text}"
        )

    data = response.json()
    choices = data.get("choices", [])
    if not choices or not isinstance(choices, list):
        raise Exception("OpenAI API did not return choices")
    content = choices[0].get("message", {}).get("content")
    if not content:
        raise Exception("OpenAI API did not return content")

    logger.debug("OpenAI raw response: %s...", content[:500])
    return content


async def call_anthropic(
    system_prompt: str, user_message: str, api_key: str
) -> str:
    """Call Anthropic (Claude) API for LLM-as-judge scoring. Returns raw content string."""
    def _do_request():
        return requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-3-5-sonnet-20241022",  # Default model
                "max_tokens": 1024,
                "temperature": 0,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
        )

    response = await asyncio.to_thread(_do_request)

    if not response.ok:
        error_text = response.text
        raise Exception(
            f"Anthropic API error: {response.status_code} {response.reason} - {error_text}"
        )

    data = response.json()
    content_list = data.get("content", [])
    if not content_list or not isinstance(content_list, list):
        raise Exception("Anthropic API did not return content")
    content = content_list[0].get("text", "")
    if not content:
        raise Exception("Anthropic API did not return content")

    logger.debug("Anthropic raw response: %s...", content[:500])
    return content


async def call_llm_fallback(
    system_prompt: str,
    user_message: str,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Fallback LLM call function that checks for API key parameter or environment variables.

    Args:
        system_prompt: System prompt for the LLM
        user_message: User message containing the output to score
        api_key: Optional API key to use (takes precedence over env vars)
        provider: Optional provider name ('openai', 'anthropic', etc.) to determine which API to call

    Returns:
        Dictionary with "score" (float 0-1) and "message" (str)
    """
    # If API key provided, use it with the specified provider
    if api_key:
        if provider == 'openai' or provider is None:
            content = await call_openai(system_prompt, user_message, api_key)
        elif provider == 'anthropic':
            content = await call_anthropic(system_prompt, user_message, api_key)
        else:
            # Try OpenAI first, then Anthropic for unknown providers
            try:
                content = await call_openai(system_prompt, user_message, api_key)
            except Exception:
                content = await call_anthropic(system_prompt, user_message, api_key)
    else:
        # Fallback to environment variables
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if openai_key:
            logger.debug("Using OpenAI API (from OPENAI_API_KEY env var)")
            content = await call_openai(system_prompt, user_message, openai_key)
        elif anthropic_key:
            logger.debug("Using Anthropic API (from ANTHROPIC_API_KEY env var)")
            content = await call_anthropic(system_prompt, user_message, anthropic_key)
        else:
            raise Exception(
                "No LLM API key found. Either:\n"
                "  - Specify a model in metric.parameters.model (to use server-stored API key), or\n"
                "  - Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable (for local API key)"
            )
    return content


async def score_llm_metric_local(
    input_data: Any,
    output: Any,
    example: Example,
    metric: Metric,
    llm_call_fn: Optional[CallLLMType] = None,
) -> MetricResult:
    """
    Score an LLM-as-judge metric.

    Args:
        input_data: The input data to score
        output: The output to score
        example: The example object
        metric: The metric definition
        llm_call_fn: Optional async function that takes (system_prompt, user_message) and returns
                    raw content string (typically JSON). If not provided, will use fallback.

    Returns:
        MetricResult object with score:[0,1], message (optional), and error (optional)
    """
    # Build system prompt from metric description or prompt field
    metric_prompt = metric.get("prompt")
    if metric_prompt:
        system_prompt = metric_prompt
    else:
        metric_text = metric.get("description", "") or metric.get("name", "")
        system_prompt = f"""You are a judge evaluating AI assistant OUTPUTs for this metric:
{metric_text}

You MUST respond with ONLY a valid JSON object (no other text before or after) containing:
- "score": a number between 0 and 1 (where 1 is best)
- "message": a brief explanation of your scoring

Example response format:
{{"score": 0.75, "message": "The response was helpful but could be more concise"}}

Be strict but fair in your evaluation."""

    if isinstance(input_data, dict):
        input_text = json.dumps(input_data)
    else:
        input_text = str(input_data)

    if isinstance(output, dict):
        output_text = json.dumps(output)
    else:
        output_text = str(output)

    # Build user message with the input and output to score 
    user_message = f"""<INPUT>:
{input_text}
</INPUT>
<OUTPUT>
{output_text}
</OUTPUT>
Evaluate this OUTPUT according to the metric and return ONLY a valid JSON object {{"score": number, "message": string}}."""

    logger.debug(
        "Sending to LLM scorer: system_prompt=%s..., user_message=%s..., input_sample=%s",
        system_prompt[:200], user_message[:500], repr(input_text[:100]),
    )
    logger.debug("Calling LLM to score metric '%s'", metric.get("name"))
    if llm_call_fn:
        result = await llm_call_fn(system_prompt, user_message)
    else:
        # Note: api_key and provider should be handled by the caller if model is specified
        # This fallback uses environment variables
        result = await call_llm_fallback(system_prompt, user_message)
    
    score_result = parse_llm_response(result)
    if not score_result:
        raise Exception(f"Failed to parse LLM response for metric {metric.get('id', 'unknown')}: {result}")
    return score_result

