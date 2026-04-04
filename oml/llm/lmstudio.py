"""
LM Studio LLM Client
=====================
Connects to a locally running LM Studio server.

LM Studio exposes two APIs:
  - Native:           POST http://localhost:1234/api/v1/chat
  - OpenAI-compat:    POST http://localhost:1234/v1/chat/completions

This client uses the **native** endpoint (``/api/v1/chat``) — the same one
tested by the user with curl.  The native format is::

    {
        "model":         "<model_name>",
        "system_prompt": "<optional system instructions>",
        "input":         "<user message>"
    }

Usage
-----
Register as ``lmstudio:<model_name>`` in the factory.  Examples:

    oml teeg-ingest "some fact" --model lmstudio:qwen/qwen3-30b-a3b
    oml teeg-query  "what is X?" --model lmstudio:qwen/qwen3-30b-a3b

Environment variables
---------------------
``LMSTUDIO_HOST``   Override the server URL (default: http://localhost:1234).
``LMSTUDIO_TIMEOUT`` Request timeout in seconds (default: 120).
"""

import logging
import os

import requests

from oml.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://localhost:1234"
_DEFAULT_TIMEOUT = 120


class LMStudioLLM(BaseLLM):
    """LLM client for a locally running LM Studio server.

    Uses the native ``/api/v1/chat`` endpoint.  Falls back to inspecting
    ``choices[0].message.content`` (OpenAI-compat shape) when the native
    ``response`` key is absent — LM Studio versions vary slightly.

    Args:
        model_name:    The model identifier as shown in LM Studio (e.g.
                       ``"qwen/qwen3-30b-a3b"``).
        base_url:      Override the server URL.  Falls back to
                       ``LMSTUDIO_HOST`` env var then ``http://localhost:1234``.
        system_prompt: Optional system prompt sent on every request.
        timeout:       HTTP timeout in seconds.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = None,
        system_prompt: str = None,
        timeout: int = None,
    ):
        self.model_name = model_name
        self.base_url = (
            base_url
            or os.environ.get("LMSTUDIO_HOST", _DEFAULT_HOST)
        ).rstrip("/")
        # system_prompt priority: constructor arg > LMSTUDIO_SYSTEM_PROMPT env var > default
        # Tip: for Qwen3 / other thinking models, set LMSTUDIO_SYSTEM_PROMPT="/nothink"
        # to disable chain-of-thought and reduce latency by ~10-30x.
        self.system_prompt = (
            system_prompt
            or os.environ.get("LMSTUDIO_SYSTEM_PROMPT", "You are a helpful assistant.")
        )
        self.timeout = timeout or int(
            os.environ.get("LMSTUDIO_TIMEOUT", str(_DEFAULT_TIMEOUT))
        )

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send *prompt* to LM Studio and return the generated text.

        Tries the native ``/api/v1/chat`` endpoint first.  Parses the
        response robustly to handle slight API version differences.
        """
        url = f"{self.base_url}/api/v1/chat"
        payload = {
            "model": self.model_name,
            "system_prompt": self.system_prompt,
            "input": prompt,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            msg = (
                f"Cannot connect to LM Studio at {self.base_url}. "
                "Make sure the server is running (Settings → Developer → "
                "Enable local server in LM Studio)."
            )
            logger.error(msg)
            return f"Error: {msg}"
        except requests.exceptions.Timeout:
            msg = f"LM Studio request timed out after {self.timeout}s."
            logger.error(msg)
            return f"Error: {msg}"
        except requests.RequestException as exc:
            logger.error("LM Studio request failed: %s", exc)
            return f"Error calling LM Studio: {exc}"

        if not resp.ok:
            logger.error(
                "LM Studio returned HTTP %s: %s", resp.status_code, resp.text[:300]
            )
            return f"Error calling LM Studio ({resp.status_code}): {resp.text[:300]}"

        try:
            data = resp.json()
        except ValueError:
            # Plain-text fallback (some LM Studio builds return raw text)
            return resp.text.strip()

        # --- LM Studio native /api/v1/chat shape (LM Studio ≥ 0.3) ---
        # {"output": [{"type": "reasoning", "content": "..."},
        #              {"type": "message",   "content": "..."}], ...}
        output_blocks = data.get("output")
        if output_blocks and isinstance(output_blocks, list):
            # Prefer the last block with type="message"; fall back to last block
            message_blocks = [
                b for b in output_blocks
                if isinstance(b, dict) and b.get("type") == "message"
            ]
            chosen = message_blocks[-1] if message_blocks else output_blocks[-1]
            return str(chosen.get("content", "")).strip()

        # Legacy native format: {"response": "..."}
        if "response" in data:
            return str(data["response"]).strip()

        # OpenAI-compat shape: {"choices": [{"message": {"content": "..."}}]}
        choices = data.get("choices")
        if choices and isinstance(choices, list) and choices[0]:
            msg = choices[0].get("message") or choices[0].get("text", "")
            if isinstance(msg, dict):
                return str(msg.get("content", "")).strip()
            return str(msg).strip()

        # Last resort: return whole JSON so the caller can see what arrived
        logger.warning("Unexpected LM Studio response shape: %s", list(data.keys()))
        return str(data)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if LM Studio is reachable, False otherwise."""
        try:
            resp = requests.get(f"{self.base_url}/api/v0/models", timeout=5)
            return resp.ok
        except requests.RequestException:
            return False

    def list_models(self) -> list:
        """Return the list of loaded models from LM Studio."""
        try:
            resp = requests.get(f"{self.base_url}/api/v0/models", timeout=10)
            if resp.ok:
                data = resp.json()
                return data if isinstance(data, list) else data.get("data", [])
        except requests.RequestException as exc:
            logger.warning("Could not list LM Studio models: %s", exc)
        return []

    def __repr__(self) -> str:
        return f"LMStudioLLM(model={self.model_name!r}, url={self.base_url!r})"
