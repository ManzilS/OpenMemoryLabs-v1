"""
Tests for LMStudioLLM and OpenRouterLLM clients.

All network calls are mocked — these tests run entirely offline.
They verify: factory routing, response parsing, error handling,
and the key logic branches (output-block format, reasoning stripping,
legacy shape, choices shape, plain-text fallback).
"""

import json
import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from oml.llm.base import BaseLLM
from oml.llm.factory import get_llm_client
from oml.llm.lmstudio import LMStudioLLM
from oml.llm.openrouter import OpenRouterLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(json_body, status_code=200):
    """Build a fake requests.Response with the given JSON body."""
    resp = Mock()
    resp.ok = (status_code < 400)
    resp.status_code = status_code
    resp.text = json.dumps(json_body)
    resp.json.return_value = json_body
    return resp


def _make_error_response(status_code=500, text="Internal Server Error"):
    resp = Mock()
    resp.ok = False
    resp.status_code = status_code
    resp.text = text
    resp.json.side_effect = ValueError("not json")
    return resp


# ---------------------------------------------------------------------------
# TestLMStudioLLM
# ---------------------------------------------------------------------------

class TestLMStudioLLM:

    # ── factory routing ──────────────────────────────────────────────────────

    def test_factory_routing(self):
        """get_llm_client('lmstudio:...') returns an LMStudioLLM."""
        client = get_llm_client("lmstudio:qwen/qwen3-30b-a3b")
        assert isinstance(client, LMStudioLLM)
        assert isinstance(client, BaseLLM)

    def test_factory_preserves_model_name(self):
        client = get_llm_client("lmstudio:my-model/v2")
        assert client.model_name == "my-model/v2"

    # ── output-block format (LM Studio >= 0.3) ───────────────────────────────

    def test_output_block_message_extracted(self):
        """Primary format: output list with type='message' block."""
        llm = LMStudioLLM("test-model")
        body = {
            "output": [
                {"type": "reasoning", "content": "Let me think..."},
                {"type": "message", "content": "The answer is 42."},
            ]
        }
        with patch("requests.post", return_value=_make_response(body)):
            result = llm.generate("What is the answer?")
        assert result == "The answer is 42."

    def test_reasoning_tokens_stripped(self):
        """Reasoning blocks must NOT appear in the returned string."""
        llm = LMStudioLLM("test-model")
        body = {
            "output": [
                {"type": "reasoning", "content": "SECRET THINKING BLOCK"},
                {"type": "message", "content": "Clean answer."},
            ]
        }
        with patch("requests.post", return_value=_make_response(body)):
            result = llm.generate("question")
        assert "SECRET THINKING BLOCK" not in result
        assert result == "Clean answer."

    def test_output_block_no_message_type_uses_last(self):
        """When no type='message' block exists, fall back to the last block."""
        llm = LMStudioLLM("test-model")
        body = {
            "output": [
                {"type": "other", "content": "Only block."},
            ]
        }
        with patch("requests.post", return_value=_make_response(body)):
            result = llm.generate("q")
        assert result == "Only block."

    def test_multiple_message_blocks_uses_last(self):
        """When multiple message blocks exist, the LAST one is returned."""
        llm = LMStudioLLM("test-model")
        body = {
            "output": [
                {"type": "message", "content": "First message."},
                {"type": "message", "content": "Final message."},
            ]
        }
        with patch("requests.post", return_value=_make_response(body)):
            result = llm.generate("q")
        assert result == "Final message."

    # ── legacy / fallback response shapes ────────────────────────────────────

    def test_legacy_response_key_parsed(self):
        """Older LM Studio format: {"response": "..."}"""
        llm = LMStudioLLM("test-model")
        body = {"response": "Legacy answer."}
        with patch("requests.post", return_value=_make_response(body)):
            result = llm.generate("q")
        assert result == "Legacy answer."

    def test_choices_fallback_parsed(self):
        """OpenAI-compat shape: {"choices": [{"message": {"content": "..."}}]}"""
        llm = LMStudioLLM("test-model")
        body = {
            "choices": [
                {"message": {"content": "OpenAI-compat answer."}}
            ]
        }
        with patch("requests.post", return_value=_make_response(body)):
            result = llm.generate("q")
        assert result == "OpenAI-compat answer."

    def test_plain_text_fallback(self):
        """If the response is not JSON, return raw text."""
        resp = Mock()
        resp.ok = True
        resp.status_code = 200
        resp.text = "Plain text response."
        resp.json.side_effect = ValueError("not json")

        llm = LMStudioLLM("test-model")
        with patch("requests.post", return_value=resp):
            result = llm.generate("q")
        assert result == "Plain text response."

    # ── error handling ────────────────────────────────────────────────────────

    def test_connection_error_returns_error_string(self):
        """ConnectionError → returns descriptive error string, does not raise."""
        import requests as req
        llm = LMStudioLLM("test-model")
        with patch("requests.post", side_effect=req.exceptions.ConnectionError("refused")):
            result = llm.generate("q")
        assert result.startswith("Error:")
        assert "connect" in result.lower() or "lm studio" in result.lower()

    def test_timeout_returns_error_string(self):
        """Timeout → returns descriptive error string, does not raise."""
        import requests as req
        llm = LMStudioLLM("test-model")
        with patch("requests.post", side_effect=req.exceptions.Timeout()):
            result = llm.generate("q")
        assert "Error" in result

    def test_http_error_status_returns_error_string(self):
        """HTTP 500 → returns error string, does not raise."""
        llm = LMStudioLLM("test-model")
        with patch("requests.post", return_value=_make_error_response(500)):
            result = llm.generate("q")
        assert "Error" in result
        assert "500" in result

    # ── ping helper ───────────────────────────────────────────────────────────

    def test_ping_uses_models_endpoint(self):
        """ping() hits /api/v0/models and returns True on 200."""
        resp = Mock()
        resp.ok = True
        llm = LMStudioLLM("test-model")
        with patch("requests.get", return_value=resp) as mock_get:
            assert llm.ping() is True
            url_called = mock_get.call_args[0][0]
        assert "api/v0/models" in url_called

    def test_ping_returns_false_on_connection_error(self):
        """ping() returns False if the server is not reachable."""
        import requests as req
        llm = LMStudioLLM("test-model")
        with patch("requests.get", side_effect=req.exceptions.ConnectionError()):
            assert llm.ping() is False


# ---------------------------------------------------------------------------
# TestOpenRouterLLM
# ---------------------------------------------------------------------------

class TestOpenRouterLLM:

    # ── factory routing ──────────────────────────────────────────────────────

    def test_factory_routing(self):
        """get_llm_client('openrouter:...') returns an OpenRouterLLM."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            client = get_llm_client("openrouter:openai/gpt-4o-mini")
        assert isinstance(client, OpenRouterLLM)
        assert isinstance(client, BaseLLM)

    def test_factory_preserves_model_name(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            client = get_llm_client("openrouter:anthropic/claude-3-haiku")
        assert client.model_name == "anthropic/claude-3-haiku"

    # ── generate() ───────────────────────────────────────────────────────────

    def test_generate_calls_chat_completions(self):
        """generate() calls the OpenAI SDK's chat.completions.create."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "OpenRouter answer."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            llm = OpenRouterLLM("openai/gpt-4o-mini")
            llm._client = mock_client
            result = llm.generate("What is 2+2?")

        assert result == "OpenRouter answer."
        mock_client.chat.completions.create.assert_called_once()

    def test_generate_returns_string(self):
        """generate() always returns a str."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "42"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            llm = OpenRouterLLM("openai/gpt-4o-mini")
            llm._client = mock_client
            result = llm.generate("question")
        assert isinstance(result, str)

    def test_no_api_key_does_not_crash_on_init(self):
        """Missing OPENROUTER_API_KEY should log a warning, not raise."""
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            # Should not raise
            llm = OpenRouterLLM("openai/gpt-4o-mini", api_key="")
        assert isinstance(llm, OpenRouterLLM)

    def test_generate_error_returns_error_string(self):
        """If the SDK raises, generate() returns an error string (no crash)."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            llm = OpenRouterLLM("openai/gpt-4o-mini")
            llm._client = mock_client
            result = llm.generate("q")

        assert "Error" in result

    def test_extra_headers_set_when_env_vars_present(self):
        """OPENROUTER_SITE_URL and OPENROUTER_APP_NAME are sent as headers."""
        env = {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_SITE_URL": "https://mysite.example",
            "OPENROUTER_APP_NAME": "TestApp",
        }
        captured_headers = {}

        def capture_init(api_key, base_url, default_headers=None):
            captured_headers.update(default_headers or {})
            return MagicMock()

        with patch.dict(os.environ, env):
            with patch("openai.OpenAI", side_effect=capture_init):
                llm = OpenRouterLLM("openai/gpt-4o-mini")

        assert "HTTP-Referer" in captured_headers
        assert captured_headers["HTTP-Referer"] == "https://mysite.example"
        assert captured_headers.get("X-Title") == "TestApp"

    def test_repr_contains_model_name(self):
        """repr() should include the model name for easy debugging."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            llm = OpenRouterLLM("anthropic/claude-3-haiku")
        assert "anthropic/claude-3-haiku" in repr(llm)

    def test_list_models_returns_list(self):
        """list_models() returns a list (empty on error, populated on success)."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            llm = OpenRouterLLM("openai/gpt-4o-mini")

        mock_resp = Mock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"data": [{"id": "openai/gpt-4o"}, {"id": "anthropic/claude-3-opus"}]}

        with patch("requests.get", return_value=mock_resp):
            models = llm.list_models()

        assert isinstance(models, list)
        assert "openai/gpt-4o" in models

    def test_list_models_returns_empty_list_on_error(self):
        """list_models() returns [] when the network call fails."""
        import requests as req
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            llm = OpenRouterLLM("openai/gpt-4o-mini")
        with patch("requests.get", side_effect=req.exceptions.ConnectionError()):
            models = llm.list_models()
        assert models == []
