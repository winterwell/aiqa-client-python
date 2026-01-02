# Testing the Package Locally

## Prerequisites

1. Ensure you have Python 3.8+ installed
2. Install build tools:
```bash
pip install build wheel
```

## Running Unit Tests

The package includes unit tests using `pytest`. To run them:

### Setup for Testing

First, install the package with development dependencies:

```bash
cd client-python

# Create and activate virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies including dev dependencies (pytest, pytest-asyncio, etc.)
pip install -r requirements.txt
pip install -e ".[dev]"   # defined in pyproject.toml
```

### Running All Tests

```bash
# Run all tests
pytest

# Or using Python module syntax
python -m pytest
```

### Running Specific Test Files

```bash
# Run a specific test file
pytest aiqa/test_tracing.py
pytest aiqa/test_experiment_runner.py
pytest aiqa/test_startup_reliability.py
```

### Running Specific Tests

```bash
# Run a specific test class
pytest aiqa/test_tracing.py::TestGetSpan

# Run a specific test method
pytest aiqa/test_tracing.py::TestGetSpan::test_get_span_success_with_span_id
```

### Test Output Options

```bash
# Verbose output (shows each test name)
pytest -v

# Very verbose output (shows print statements)
pytest -vv -s

# Show test coverage (requires pytest-cov)
pytest --cov=aiqa --cov-report=html
```

## Method 1: Install in Development Mode (Recommended)

This allows you to make changes and test them immediately without rebuilding:

```bash
cd client-python

# Create and activate virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable/development mode
pip install -e .
```

Now you can test imports:
```bash
python -c "from aiqa import WithTracing; print('Import successful')"
```

Run the example:
```bash
pip install -r requirements.examples.txt
python example.py
```

## Method 2: Build and Install from Wheel

This tests the actual distribution that will be uploaded to PyPI:

```bash
cd client-python

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

# Install from the built wheel
pip install dist/aiqa_client-*.whl

# Or install from source distribution
pip install dist/aiqa-client-*.tar.gz
```

Test imports:
```bash
python -c "from aiqa import WithTracing, flush_tracing, shutdown_tracing; print('All imports successful!')"
```

## Method 3: Test with a Test Script

Create a test script to verify functionality:

```python
# test_aiqa.py
import asyncio
import os
from aiqa import WithTracing, flush_tracing, shutdown_tracing

# Set environment variables (or use .env file)
os.environ["AIQA_SERVER_URL"] = "http://localhost:3000"
os.environ["AIQA_API_KEY"] = "test-key"

@WithTracing
def sync_function(x: int, y: int) -> int:
    """A synchronous traced function."""
    return x + y

@WithTracing
async def async_function(x: int, y: int) -> int:
    """An asynchronous traced function."""
    await asyncio.sleep(0.1)
    return x * y

@WithTracing(name="custom_name")
def named_function():
    """A function with custom span name."""
    return "hello"

async def main():
    # Test sync function
    result1 = sync_function(5, 3)
    print(f"sync_function result: {result1}")

    # Test async function
    result2 = await async_function(4, 6)
    print(f"async_function result: {result2}")

    # Test named function
    result3 = named_function()
    print(f"named_function result: {result3}")

    # Flush spans (optional - auto-flush happens every 5 seconds)
    await flush_tracing()
    
    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())
```

Run the test:
```bash
python test_aiqa.py
```

## Verify Package Structure

Check that all files are included correctly:

```bash
# List package contents
python -c "import aiqa; import os; print(os.path.dirname(aiqa.__file__))"
ls -la $(python -c "import aiqa; import os; print(os.path.dirname(aiqa.__file__))")
```

Expected files:
- `__init__.py`
- `tracing.py`
- `aiqa_exporter.py`
- `py.typed`

## Check Package Metadata

Verify package information:

```bash
pip show aiqa-client
```

Should show:
- Name: aiqa-client
- Version: 1.0.0
- Location: (path to installed package)

## Test Without Server

The package will work even without a server configured - spans will be buffered but not sent. You'll see warnings in the logs:

```bash
# Without AIQA_SERVER_URL set
python example.py
# Should see: "Skipping flush: AIQA_SERVER_URL is not set..."
```

## Test with Mock Server (Optional)

To test the full flow, you can use a simple HTTP server:

```bash
# Terminal 1: Start a mock server
python -m http.server 3000

# Terminal 2: Run your test with server URL set
export AIQA_SERVER_URL="http://localhost:3000"
python example.py
```

## Clean Up

To uninstall and start fresh:

```bash
pip uninstall aiqa-client -y
rm -rf dist/ build/ *.egg-info
```

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated
- Check that package is installed: `pip list | grep aiqa-client`
- Verify Python path: `python -c "import sys; print(sys.path)"`

### Build Errors
- Ensure `build` and `wheel` are installed: `pip install build wheel`
- Check `pyproject.toml` syntax is valid
- Verify all dependencies in `requirements.txt` are installed

### Runtime Errors
- Check OpenTelemetry SDK version compatibility
- Verify environment variables are set correctly
- Check server URL is accessible (if testing with server)

