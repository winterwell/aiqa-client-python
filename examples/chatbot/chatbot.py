"""
Simple command-line chatbot that uses OpenAI and can perform web searches.
Loads environment variables from aiqa-client-python/.env
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json
from typing import List, Dict, Any
from ddgs import DDGS

# Add parent directory to Python path so we can import aiqa
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from aiqa import WithTracing, get_aiqa_client

# Load environment variables from parent directory .env file
env_path = parent_dir / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize AIQA client for tracing
get_aiqa_client()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("ERROR: OPENAI_API_KEY not found in environment variables.", file=sys.stderr)
    print(f"Please set it in {env_path}", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=openai_api_key)


@WithTracing
def web_search(query: str) -> str:
    """
    Web search function using DuckDuckGo search.
    Returns a formatted string with search results.
    """
    try:
        # Use DuckDuckGo search library for reliable results
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        
        if not results:
            return f"Search query: {query}\n(No search results found. Try rephrasing your question.)"
        
        result_parts = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "").strip()
            body = result.get("body", "").strip()
            href = result.get("href", "").strip()
            
            if title:
                result_parts.append(f"{i}. {title}")
                if body:
                    result_parts.append(f"   {body}")
                if href:
                    result_parts.append(f"   URL: {href}")
                result_parts.append("")
        
        return "\n".join(result_parts).strip()
            
    except Exception as e:
        return f"Search query: {query}\n(Error performing search: {str(e)})"


def get_tools() -> List[Dict[str, Any]]:
    """Get the list of tools available for OpenAI function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information, facts, or answers to questions. Use this when you need up-to-date information or when you're unsure about something.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]


@WithTracing
def call_openai(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> Any:
    """Call OpenAI API with the given messages and tools."""
    print(f"    ...calling OpenAI...")
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Let the model decide when to use tools
    )


@WithTracing
def handle_tool_call(tool_call: Any, messages: List[Dict[str, Any]]) -> str:
    """Handle a single tool call and return the result."""
    function_name = tool_call.function.name
    print(f"    ...handling tool call: {function_name}...")
    function_args = json.loads(tool_call.function.arguments)
    
    if function_name == "web_search":
        query = function_args.get("query", "")
        return web_search(query)
    else:
        return f"Unknown function: {function_name}"


@WithTracing
def process_tool_calls(
    tool_calls: List[Any], messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str = "gpt-4o-mini"
) -> Any:
    """Process all tool calls and get the final assistant response."""
    # Add tool results to messages
    for tool_call in tool_calls:
        function_result = handle_tool_call(tool_call, messages)
        
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": function_result,
            }
        )
    
    # Get the next response from OpenAI
    response = call_openai(messages, tools, model)
    assistant_message = response.choices[0].message
    messages.append(assistant_message)
    
    return assistant_message


@WithTracing
def process_user_input(
    user_input: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str = "gpt-4o-mini"
) -> str:
    """Process a single user input and return the assistant's response."""
    # Add user message to conversation
    messages.append({"role": "user", "content": user_input})
    
    # Call OpenAI with function calling
    response = call_openai(messages, tools, model)
    assistant_message = response.choices[0].message
    messages.append(assistant_message)
    
    # Handle tool calls iteratively
    while assistant_message.tool_calls:
        assistant_message = process_tool_calls(
            assistant_message.tool_calls, messages, tools, model
        )
    
    # Return the final assistant response content
    return assistant_message.content or ""


def main(model: str = "gpt-4o-mini", system_prompt: str = "You are a helpful assistant. You can search the web when users ask questions that require current information or facts you're unsure about. Use the web_search tool when appropriate."):
    """Main chatbot loop."""
    print("Chatbot ready! Type 'exit' or 'quit' to stop.\n", flush=True)
    
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    
    tools = get_tools()
    
    while True:
        try:
            # Read user input from stdin
            print("You: ", end="", flush=True)
            user_input = sys.stdin.readline()
            
            if not user_input:
                # EOF
                break
            
            user_input = user_input.strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!", flush=True)
                break
            
            # Process user input
            response_content = process_user_input(user_input, messages, tools, model)
            
            # Print the assistant's response
            if response_content:
                print(f"Bot: {response_content}\n", flush=True)
            
        except KeyboardInterrupt:
            print("\nGoodbye!", flush=True)
            break
        except Exception as e:
            print(f"Error: {e}\n", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

