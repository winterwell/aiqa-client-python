# Publishing to PyPI

## Quick Update

```bash
rm -rf dist/ build/ *.egg-info
source .venv/bin/activate
python -m build
python -m twine upload dist/*
```

## Prerequisites

1. Install build tools:
```bash
pip install build twine
```

2. Create accounts:
   - PyPI account at https://pypi.org/account/register/
   - TestPyPI account at https://test.pypi.org/account/register/ (separate account!)

3. Create API tokens:
   - For PyPI: https://pypi.org/manage/account/token/
   - For TestPyPI: https://test.pypi.org/manage/account/token/ (separate token!)
   
   **Important**: TestPyPI requires a separate account and token from regular PyPI.

## Building the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source and wheel distributions
python -m build
```

This creates:
- `dist/aiqa-client-1.0.0.tar.gz` (source distribution)
- `dist/aiqa_client-1.0.0-py3-none-any.whl` (wheel)

## Testing the Build

Test the build locally before uploading:

```bash
# Install from the built wheel
pip install dist/aiqa_client-*.whl

# Test imports
python -c "from aiqa import WithTracing; print('OK')"
```

## Uploading to PyPI

### Test PyPI (for testing)

**Important**: TestPyPI requires a separate account and API token from regular PyPI.

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# When prompted:
# Username: __token__
# Password: Your TestPyPI API token (from https://test.pypi.org/manage/account/token/)

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ aiqa-client
```

**Troubleshooting 403 Forbidden errors:**
- Ensure you're using a TestPyPI token (not a regular PyPI token)
- Verify the token hasn't expired
- Check that the package name isn't already taken by another user
- Try creating a new token with "Entire account" scope
- If using environment variables, ensure `TWINE_PASSWORD` is set to your TestPyPI token

### Production PyPI

```bash
# Upload to production PyPI
python -m twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token

## Version Updates

To publish a new version:

1. Update version in `pyproject.toml`:
```toml
version = "1.0.1"
```

2. Update version in `aiqa/__init__.py`:
```python
__version__ = "1.0.1"
```

3. Build and upload:
```bash
python -m build
python -m twine upload dist/*
```

## Automated Publishing

For CI/CD, set environment variables:

**For TestPyPI:**
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-testpypi-api-token>
python -m twine upload --repository testpypi dist/*
```

**For Production PyPI:**
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-pypi-api-token>
python -m twine upload dist/*
```

Alternatively, create a `~/.pypirc` file:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <your-pypi-api-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <your-testpypi-api-token>
```

Once you've created and saved your `.pypirc` file (see above) in your home directory, `twine` will automatically use the credentials stored in this file when uploading packages. This allows you to avoid specifying usernames and passwords each time.

**Notes:**
- Ensure that your `.pypirc` file has secure permissions (e.g., `chmod 600 ~/.pypirc`) to protect your tokens.
- You can define additional repositories in your `.pypirc` as needed by following the same format.

