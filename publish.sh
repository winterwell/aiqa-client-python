# Assumes setup has been done
echo "Publishing to PyPI"
echo "Assumes setup has been done. That is: python -m venv .venv, source .venv/bin/activate, pip install build twine, .pypirc file created"

echo "Removing old builds: rm -rf dist/ build/ *.egg-info"
rm -rf dist/ build/ *.egg-info

echo "Activating virtual environment"
source .venv/bin/activate

echo "Building package: python -m build"
python -m build

echo "Uploading package to PyPI: python -m twine upload dist/*"
python -m twine upload dist/*
echo "Done"