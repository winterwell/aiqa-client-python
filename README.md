# A Python client for the AIQA server

OpenTelemetry-based client for tracing Python functions and sending traces to the AIQA server.

## Installation

### From PyPI (recommended)

```bash
pip install aiqa-client
```

### From source

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Development Setup

For development, install with dev dependencies to run tests:

```bash
pip install -e ".[dev]"
```

Then run the unit tests:

```bash
pytest
```

See [TESTING.md](TESTING.md) for detailed testing instructions.

## Setup

Set the following environment variables:

```bash
export AIQA_SERVER_URL="http://localhost:3000"
export AIQA_API_KEY="your-api-key"
```

**Note:** If `AIQA_SERVER_URL` or `AIQA_API_KEY` are not set, tracing will be automatically disabled. You'll see one warning message at the start, and your application will continue to run without tracing.
You can check if tracing is enabled via `get_aiqa_client().enabled`.

## Usage

### Basic Usage

```python
from dotenv import load_dotenv
from aiqa import get_aiqa_client, WithTracing

# Load environment variables from .env file (if using one)
load_dotenv()

# Initialize client (must be called before using WithTracing)
# This loads environment variables and initializes the tracing system
get_aiqa_client()

@WithTracing
def my_function(x, y):
    return x + y

@WithTracing
async def my_async_function(x, y):
    await asyncio.sleep(0.1)
    return x * y
```

### Custom Span Name

```python
@WithTracing(name="custom_span_name")
def my_function():
    pass
```

### Input/Output Filtering

```python
@WithTracing(
    filter_input=lambda x: {"filtered": str(x)},
    filter_output=lambda x: {"result": x}
)
def my_function(data):
    return {"processed": data}
```

### Flushing Spans

Spans are automatically flushed every 5 seconds. To flush immediately:

```python
from aiqa import flush_tracing
import asyncio

async def main():
    # Your code here
    await flush_tracing()

asyncio.run(main())
```

### Shutting Down

To ensure all spans are sent before process exit:

```python
from aiqa import flush_tracing
import asyncio

async def main():
    # Your code here
    await flush_tracing()

asyncio.run(main())
```

### Enabling/Disabling Tracing

You can programmatically enable or disable tracing:

```python
from aiqa import get_aiqa_client

client = get_aiqa_client()

# Disable tracing (spans won't be created or exported)
client.enabled = False

# Re-enable tracing
client.enabled = True

# Check if tracing is enabled
if client.enabled:
    print("Tracing is enabled")
```

When tracing is disabled:
- Spans are not created (functions execute normally without tracing overhead)
- Spans are not exported to the server

### Setting Span Attributes and Names

```python
from aiqa import set_span_attribute, set_span_name

def my_function():
    set_span_attribute("custom.attribute", "value")
    set_span_name("custom_span_name")
    # ... rest of function
```

### Grouping Traces by Conversation

To group multiple traces together that are part of the same conversation or session:

```python
from aiqa import WithTracing, set_conversation_id

@WithTracing
def handle_user_request(user_id: str, session_id: str):
    # Set conversation ID to group all traces for this user session
    set_conversation_id(f"user_{user_id}_session_{session_id}")
    # All spans created in this function and its children will have this gen_ai.conversation.id
    # ... rest of function
```

The `gen_ai.conversation.id` attribute allows you to filter and group traces in the AIQA server by conversation, making it easier to analyze multi-step interactions or user sessions. See the [OpenTelemetry GenAI Events specification](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/) for more details.

### Trace ID Propagation Across Services/Agents

To link traces across different services or agents, you can extract and propagate trace IDs:

#### Getting Current Trace ID

```python
from aiqa import get_active_trace_id, get_span_id

# Get the current trace ID and span ID
trace_id = get_active_trace_id()  # Returns hex string (32 chars) or None
span_id = get_span_id()    # Returns hex string (16 chars) or None

# Pass these to another service (e.g., in HTTP headers, message queue, etc.)
```

#### Continuing a Trace in Another Service

```python
from aiqa import create_span_from_trace_id

# Continue a trace from another service/agent
# trace_id and parent_span_id come from the other service
with create_span_from_trace_id(
    trace_id="abc123...", 
    parent_span_id="def456...",
    span_name="service_b_operation"
):
    # Your code here - this span will be linked to the original trace
    pass
```

#### Using OpenTelemetry Context Propagation (Recommended)

For HTTP requests, use the built-in context propagation:

```python
from aiqa import inject_trace_context, extract_trace_context
import requests
from opentelemetry.trace import use_span

# In the sending service:
headers = {}
inject_trace_context(headers)  # Adds trace context to headers
response = requests.get("http://other-service/api", headers=headers)

# In the receiving service:
# Extract context from incoming request headers
ctx = extract_trace_context(request.headers)

# Use the context to create a span
from opentelemetry.trace import use_span
with use_span(ctx):
    # Your code here
    pass

# Or create a span with the context
from opentelemetry import trace
tracer = trace.get_tracer("aiqa-tracer")
with tracer.start_as_current_span("operation", context=ctx):
    # Your code here
    pass
```

## Features

- Automatic tracing of function calls (sync and async)
- Records function inputs and outputs as span attributes
- Automatic error tracking and exception recording
- Thread-safe span buffering and auto-flushing
- OpenTelemetry context propagation for nested spans
- Trace ID propagation utilities for distributed tracing

## Example

See `example.py` for a complete working example.