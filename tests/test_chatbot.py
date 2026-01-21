"""
Unit tests for chatbot.py web_search function.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import sys
from pathlib import Path

# Mock dependencies before importing chatbot module
# These patches will remain active for the duration of the test module
def mock_getenv(key, default=None):
    """Mock os.getenv that returns appropriate values for different env vars."""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        # Don't set AIQA_API_KEY to disable tracing during tests
        # AIQA_SERVER_URL: not needed since tracing is disabled
    }
    return env_vars.get(key, default)

_mock_get_aiqa_client = patch("aiqa.get_aiqa_client", return_value=Mock())
_mock_load_dotenv = patch("dotenv.load_dotenv")
_mock_getenv = patch("os.getenv", side_effect=mock_getenv)
_mock_openai = patch("openai.OpenAI", return_value=Mock())
_mock_ddgs = patch("ddgs.DDGS")

_mock_get_aiqa_client.start()
_mock_load_dotenv.start()
_mock_getenv.start()
_mock_openai.start()
_mock_ddgs.start()

# Add parent directory to Python path so we can import chatbot
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "examples" / "chatbot"))

# Now import chatbot - the mocks are already in place
from chatbot import web_search


class TestWebSearch:
    """Tests for web_search function."""

    @patch("chatbot.DDGS")
    def test_web_search_success(self, mock_ddgs_class):
        """Test successful search returns results."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_context = MagicMock()
        mock_ddgs_context.__enter__.return_value = mock_ddgs_instance
        mock_ddgs_context.__exit__.return_value = None
        mock_ddgs_class.return_value = mock_ddgs_context
        
        mock_ddgs_instance.text.return_value = [
            {
                "title": "The Answer to Life",
                "body": "The answer to life, the universe, and everything is 42",
                "href": "https://example.com/answer"
            },
            {
                "title": "Hitchhiker's Guide",
                "body": "A comedic science fiction series",
                "href": "https://example.com/hitchhikers"
            }
        ]

        result = web_search("what is the answer to life")

        assert "The Answer to Life" in result
        assert "answer to life, the universe, and everything is 42" in result
        assert "Hitchhiker's Guide" in result
        mock_ddgs_instance.text.assert_called_once_with("what is the answer to life", max_results=5)

    @patch("chatbot.DDGS")
    def test_web_search_no_results(self, mock_ddgs_class):
        """Test search with no results."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_context = MagicMock()
        mock_ddgs_context.__enter__.return_value = mock_ddgs_instance
        mock_ddgs_context.__exit__.return_value = None
        mock_ddgs_class.return_value = mock_ddgs_context
        mock_ddgs_instance.text.return_value = []

        result = web_search("test query")

        assert "No search results found" in result
        assert "test query" in result

