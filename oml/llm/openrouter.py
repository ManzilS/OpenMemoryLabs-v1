"""
OpenRouter LLM Client
======================
Connects to `OpenRouter <https://openrouter.ai>`_ — a unified API gateway
that routes requests to hundreds of models (GPT-4o, Claude, Llama, Mistral,
Gemini, …) through a single OpenAI-compatible endpoint.

OpenRouter is **fully OpenAI-SDK-compatible**, so this client reuses the
official ``openai`` Python package with a custom ``base_url``.

Usage
-----
Register as ``openrouter:<model_name>`` in the factory.  Examples:

    oml teeg-ingest "some fact" --model openrouter:openai/gpt-4o-mini
    oml teeg-ingest "some fact" --model openrouter:anthropic/claude-3-haiku
    oml teeg-ingest "some fact" --model openrouter:meta-llama/llama-3-8b-instruct

Environment variables
---------------------
``OPENROUTER_API_KEY``  Required.  Your OpenRouter API key.
                        Get one at https://openrouter.ai/keys
``OPENROUTER_BASE_URL`` Override the API base URL (default:
                        https://openrouter.ai/api/v1).
``OPENROUTER_SITE_URL`` Optional.  Passed as ``HTTP-Referer`` header (helps
                        OpenRouter attribute usage to your project).
``OPENROUTER_APP_NAME`` Optional.  Passed as ``X-Title`` header.

Model strings
-------------
OpenRouter uses ``<provider>/<model>`` slugs, e.g.:

    openai/gpt-4o-mini
    anthropic/claude-3-haiku
    meta-llama/llama-3-8b-instruct
    mistralai/mistral-7b-instruct
    google/gemini-flash-1.5

Full list: https://openrouter.ai/models
"""

import logging
import os

from oml.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLM(BaseLLM):
    """LLM client that routes requests through OpenRouter.

    Requires the ``openai`` Python package (``pip install openai``).
    Set the ``OPENROUTER_API_KEY`` environment variable before use.

    Args:
        model_name:  OpenRouter model slug, e.g. ``"openai/gpt-4o-mini"``.
        api_key:     Override the API key (falls back to ``OPENROUTER_API_KEY``
                     env var).
        base_url:    Override the API endpoint (rarely needed).
        temperature: Sampling temperature (0 = deterministic).
        max_tokens:  Maximum tokens in the completion.
        site_url:    Your site URL for OpenRouter attribution.  Falls back to
                     ``OPENROUTER_SITE_URL`` env var.
        app_name:    Your app name for OpenRouter attribution.  Falls back to
                     ``OPENROUTER_APP_NAME`` env var.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0,
        max_tokens: int = 2048,
        site_url: str = None,
        app_name: str = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenRouter. "
                "Install it with: pip install openai"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not resolved_key:
            logger.warning(
                "OPENROUTER_API_KEY is not set. Calls will fail with 401. "
                "Get a key at https://openrouter.ai/keys"
            )

        resolved_base = (
            base_url
            or os.environ.get("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL)
        )

        # Build extra headers for OpenRouter attribution
        extra_headers = {}
        _site = site_url or os.environ.get("OPENROUTER_SITE_URL", "")
        _app = app_name or os.environ.get("OPENROUTER_APP_NAME", "OpenMemoryLab")
        if _site:
            extra_headers["HTTP-Referer"] = _site
        if _app:
            extra_headers["X-Title"] = _app

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
            default_headers=extra_headers if extra_headers else None,
        )

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send *prompt* to OpenRouter and return the generated text.

        Raises:
            openai.AuthenticationError: If ``OPENROUTER_API_KEY`` is invalid.
            openai.RateLimitError:      If you exceed your rate limits.
        """
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            # Catch broad so pipeline doesn't crash; log the real error
            logger.error("OpenRouter call failed for model %s: %s", self.model_name, exc)
            return f"Error calling OpenRouter ({self.model_name}): {exc}"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def list_models(self) -> list:
        """Return available model slugs from the OpenRouter models endpoint.

        Requires a valid API key.  Returns an empty list on any error.
        """
        try:
            import requests  # lightweight: avoids openai SDK overhead
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self._client.api_key}"},
                timeout=15,
            )
            if resp.ok:
                return [m["id"] for m in resp.json().get("data", [])]
        except Exception as exc:
            logger.warning("Could not list OpenRouter models: %s", exc)
        return []

    def __repr__(self) -> str:
        return f"OpenRouterLLM(model={self.model_name!r})"
