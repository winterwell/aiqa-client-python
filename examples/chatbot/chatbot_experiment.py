"""
Chatbot experiment runner using ExperimentRunner.
Runs experiments on datasets using the chatbot engine.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import requests

# Add parent directory to Python path so we can import aiqa
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables from parent directory .env file BEFORE importing chatbot
# (chatbot.py initializes OpenAI client on import, which requires OPENAI_API_KEY)
env_path = parent_dir / ".env"
load_dotenv(dotenv_path=env_path)

# Add current directory to path for chatbot imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from aiqa import ExperimentRunner, WithTracing, get_aiqa_client
from aiqa.http_utils import get_server_url, get_api_key, build_headers, format_http_error

# Import chatbot functions (after env is loaded)
from chatbot import process_user_input, get_tools

# Initialize AIQA client for tracing
get_aiqa_client()

# ============================================================================
# CONFIGURATION - Set these parameters as needed
# ============================================================================

EXPERIMENT_NAME = "Chatbot Experiment"
MODEL = "gpt-4o-mini"  # Options: "gpt-4o-mini", "gpt-4o", "gpt-4", etc.
SYSTEM_PROMPT = "You are a helpful assistant. You can search the web when users ask questions that require current information or facts you're unsure about. Use the web_search tool when appropriate."

# ============================================================================

def list_datasets(organisation_id: Optional[str] = None, server_url: Optional[str] = None, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all datasets for the given organisation.
    
    Args:
        organisation_id: Optional organisation ID. Note: Currently required as a query parameter
                         even for API key authentication, though the API key implicitly sets the organisation.
        server_url: Optional server URL (defaults to env var)
        api_key: Optional API key (defaults to env var)
    
    Returns:
        List of dataset objects
    
    Raises:
        Exception: If the request fails or organisation_id is missing
    """
    url = get_server_url(server_url)
    key = get_api_key(api_key)
    headers = build_headers(key)
    
    # The server currently requires organisation as a query parameter
    # even though API key auth implicitly sets it
    if not organisation_id:
        # Try without organisation param - might work if server is updated
        params = {}
    else:
        params = {"organisation": organisation_id}
    
    response = requests.get(
        f"{url}/dataset",
        params=params if params else None,
        headers=headers,
    )
    
    if not response.ok:
        if not organisation_id and response.status_code == 400:
            error_msg = (
                "Organisation ID is required to list datasets. "
                "Although the API key implicitly sets the organisation, "
                "the /dataset endpoint currently requires it as a query parameter. "
                "Please set AIQA_ORGANISATION_ID environment variable."
            )
            raise Exception(f"{error_msg} Server error: {format_http_error(response, 'list datasets')}")
        raise Exception(format_http_error(response, "list datasets"))
    
    data = response.json()
    # Handle both array and object with hits field
    if isinstance(data, list):
        return data
    return data.get("hits", [])


def get_first_dataset(organisation_id: Optional[str] = None, server_url: Optional[str] = None, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the first dataset from the list for the given organisation.
    
    Args:
        organisation_id: Optional organisation ID. If not provided, the server will use the organisation
                         from the API key (for API key authentication).
        server_url: Optional server URL (defaults to env var)
        api_key: Optional API key (defaults to env var)
    
    Returns:
        First dataset object or None if no datasets found
    """
    datasets = list_datasets(organisation_id, server_url, api_key)
    if not datasets:
        return None
    return datasets[0]


@WithTracing
def chatbot_engine(input_data: Any, parameters: Dict[str, Any]) -> str:
    """
    Engine function for the experiment runner.
    Takes input_data (which should be a user message string) and parameters,
    and returns the chatbot's response.
    
    Args:
        input_data: The input from the example (should be a string user message)
        parameters: Parameters dict that may contain 'model' and 'system_prompt'
    
    Returns:
        The chatbot's response as a string
    """
    # Extract model and system_prompt from parameters, with defaults
    model = parameters.get("model", MODEL)
    system_prompt = parameters.get("system_prompt", SYSTEM_PROMPT)
    
    # Handle input_data - it might be a string or a dict
    if isinstance(input_data, dict):
        # If it's a dict, try to extract the user message
        user_input = input_data.get("input") or input_data.get("message") or str(input_data)
    else:
        user_input = str(input_data)
    
    # Initialize messages with system prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    
    tools = get_tools()
    
    # Process the user input and get response
    response_content = process_user_input(user_input, messages, tools, model)
    
    return response_content or ""




async def main():
    """Main function to run the chatbot experiment."""
    # Get organisation ID from environment (optional - can be derived from dataset)
    organisation_id = os.getenv("AIQA_ORGANISATION_ID")
    
    # Note: The organisation ID is usually not needed as the API key implicitly sets it.
    # However, listing datasets currently requires it as a query parameter.
    # If not provided, we'll try to get it from the first dataset we fetch.
    
    # Get the first dataset
    # Note: The /dataset endpoint currently requires organisation as a query parameter
    # even though API key auth implicitly sets it. This is a server limitation.
    print("Fetching datasets...")
    first_dataset = get_first_dataset(organisation_id)
    
    if not first_dataset:
        print("ERROR: No datasets found for this organisation.", file=sys.stderr)
        sys.exit(1)
    
    dataset_id = first_dataset.get("id")
    dataset_name = first_dataset.get("name", "Unnamed Dataset")
    
    # If organisation_id wasn't provided, get it from the dataset
    if not organisation_id:
        organisation_id = first_dataset.get("organisation")
        if organisation_id:
            print(f"Using organisation ID from dataset: {organisation_id}")
        else:
            print("ERROR: Could not determine organisation ID from dataset.", file=sys.stderr)
            print(f"Please set AIQA_ORGANISATION_ID in {env_path}", file=sys.stderr)
            sys.exit(1)
    
    print(f"Using dataset: {dataset_name} (ID: {dataset_id})")
    
    # Create experiment runner (organisation_id can be None - it will be derived from dataset)
    experiment_runner = ExperimentRunner(
        dataset_id=dataset_id,
        organisation_id=organisation_id,
    )
    
    # Get dataset info
    dataset = experiment_runner.get_dataset()
    metrics = dataset.get("metrics", [])
    print(f"Found {len(metrics)} metrics in dataset: {[m.get('name') for m in metrics]}")
    
    # Create experiment with custom setup
    experiment = experiment_runner.create_experiment(
        {
            "name": EXPERIMENT_NAME,
            "parameters": {
                "model": MODEL,
                "system_prompt": SYSTEM_PROMPT,
            },
        }
    )
    
    print(f"Created experiment: {experiment.get('id')} - {EXPERIMENT_NAME}")
    print(f"Model: {MODEL}")
    print(f"System prompt: {SYSTEM_PROMPT[:50]}...")
    print()
    
    # Get example inputs
    examples = experiment_runner.get_example_inputs()
    print(f"Processing {len(examples)} examples")
    print()
    
    # Run experiments on each example
    # No scorer provided - will use default scorer which handles LLM metrics locally
    # using API keys from .env file (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    for example in examples:
        try:
            result = await experiment_runner.run_example(example, chatbot_engine, None)
            if result and len(result) > 0:
                print(f"Completed example {example.get('id', 'unknown')}: {result}")
            else:
                print(f"No results for example {example.get('id', 'unknown')}")
        except Exception as e:
            print(f"Error processing example {example.get('id', 'unknown')}: {e}")
            # Continue with next example instead of failing entire run
    
    # Get summaries
    print("\nFetching summary results...")
    summaries = experiment_runner.get_summaries()
    print(f"Summaries: {summaries}")


if __name__ == "__main__":
    asyncio.run(main())

