"""
Example usage of the WithTracing decorator.
"""

import asyncio
import os
from dotenv import load_dotenv
import logging
from aiqa import get_aiqa_client, WithTracing, flush_tracing, set_span_name

# Load environment variables from .env file
load_dotenv()

# Initialize client (must be called before using WithTracing)
# This loads environment variables and initializes the tracing system
client = get_aiqa_client()

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("aiqa").setLevel(logging.DEBUG)

# Print to verify environment variables are loaded
print(os.getenv("AIQA_SERVER_URL"))
print(os.getenv("AIQA_API_KEY"))
print(f"Component tag: {os.getenv('AIQA_COMPONENT_TAG', 'not set')}")

@WithTracing
def sync_function(x: int, y: int) -> int:
    """A simple synchronous function."""
    return x + y


@WithTracing
async def async_function(x: int, y: int) -> int:
    """A simple asynchronous function."""
    await asyncio.sleep(0.1)  # Simulate async work
    return x * y


@WithTracing(name="custom_name")
def named_function():
    """A function with a custom span name."""
    return "hello"


@WithTracing
def fn_with_dynamic_trace_name(name:str) -> str:
    set_span_name(f"dynamic_trace_name_{name}")
    return f"hello {name}"

@WithTracing(filter_input=lambda x: {"filtered": str(x)})
def filtered_input_function(data: dict):
    """A function with input filtering."""
    return {"result": "processed"}


async def main():
    # Call synchronous traced function
    result1 = sync_function(5, 3)
    print(f"sync_function result: {result1}")

    # Call asynchronous traced function
    result2 = await async_function(4, 6)
    print(f"async_function result: {result2}")

    # Call named function
    result3 = named_function()
    print(f"named_function result: {result3}")

    # Call filtered function
    result4 = filtered_input_function({"key": "value"})
    print(f"filtered_input_function result: {result4}")

    # Flush spans before exiting (optional - auto-flush happens every 5 seconds)
    # await flush_tracing()
    
    # Call function with dynamic trace name
    result5 = fn_with_dynamic_trace_name("world")
    print(f"fn_with_dynamic_trace_name result: {result5}")
    result6 = named_function()
    print(f"named_function result: {result6}")
    await asyncio.sleep(5)
    print(f"5 seconds passed")
    await asyncio.sleep(60)
    # auto-flush should have done the job


if __name__ == "__main__":
    asyncio.run(main())

