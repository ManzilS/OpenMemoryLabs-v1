"""oml/llm/factory.py — LLM client factory.

Routes a model-string prefix to the correct provider client.  All clients
implement the ``BaseLLM`` protocol (a single ``generate(prompt) -> str``
method), so the rest of the pipeline is provider-agnostic.

Supported prefixes
------------------
``mock``                  Deterministic mock — returns a fixed response.  No API key,
                          no network.  Used in all unit/integration tests.
``smart-mock``            Prompt-aware mock — returns structurally correct TOON blocks,
                          verdicts, or QA answers.  Zero API calls, microsecond latency.
                          Ideal for dev iteration and CI smoke tests.
``ollama:<name>``         Local Ollama instance (``OLLAMA_HOST`` env var, default
                          http://localhost:11434).  Example: ``ollama:qwen3:4b``.
``openai:<name>``         OpenAI API.  Requires ``OPENAI_API_KEY`` env var.
                          Example: ``openai:gpt-4o-mini``.
``gemini:<name>``         Google Gemini API.  Requires ``GEMINI_API_KEY`` env var.
                          Example: ``gemini:gemini-1.5-flash``.
``lmstudio:<name>``       Local LM Studio server (``LMSTUDIO_HOST`` env var,
                          default http://localhost:1234).  Uses native
                          ``/api/v1/chat`` endpoint.
                          Example: ``lmstudio:qwen/qwen3-30b-a3b``.
``openrouter:<name>``     OpenRouter API gateway.  Requires ``OPENROUTER_API_KEY``
                          env var.  Supports 200+ models via OpenAI-compatible API.
                          Example: ``openrouter:openai/gpt-4o-mini``.
<bare model name>   Treated as an Ollama model name (backward-compatible
                    shorthand).  Example: ``qwen3.5:cloud``.

Cache wrapping
--------------
If the environment variable ``OML_CACHE_MODE`` is set to any value other than
``"off"``, the returned client is automatically wrapped in a
``CachedLLMClient``.  This means every ``generate()`` call checks the
disk-persisted cache first; only cache misses reach the real provider.

  * ``OML_CACHE_MODE=auto``    (default when env var is set) — hit→cache, miss→API+store
  * ``OML_CACHE_MODE=record``  — always call API, always overwrite cache
  * ``OML_CACHE_MODE=replay``  — strict replay; raises ``CacheMissError`` on miss
  * ``OML_CACHE_MODE=off``     — no caching (same as env var not set)

Cache file location: ``artifacts/llm_cache.json`` (``OML_CACHE_DIR`` overrides
the directory).  Mock/smart-mock clients are never wrapped (they make no API
calls, so caching them is pointless).
"""

import os

from oml.llm.base import BaseLLM

# Models that make no API calls — skip cache wrapping for these
_NO_CACHE_MODELS = frozenset({"mock", "dummy", "smart-mock"})


def get_llm_client(model_str: str) -> BaseLLM:
    """Return a ``BaseLLM`` instance for the given model string.

    If ``OML_CACHE_MODE`` is set (and not ``"off"``), wraps the returned client
    in a ``CachedLLMClient`` transparently.

    Args:
        model_str: A provider-prefixed model identifier.  See module docstring
            for the full list of supported formats.

    Returns:
        A concrete ``BaseLLM`` subclass ready to call ``.generate()``.

    Raises:
        ImportError: If the provider's optional dependency is not installed
            (e.g. ``google-genai`` for Gemini).
    """
    inner = _build_inner(model_str)

    # Wrap with cache if OML_CACHE_MODE is set (and model isn't a mock)
    cache_mode = os.environ.get("OML_CACHE_MODE", "off")
    if cache_mode != "off" and model_str not in _NO_CACHE_MODELS:
        from oml.llm.cache import CachedLLMClient, LLMCache
        cache_dir = os.environ.get("OML_CACHE_DIR", "artifacts")
        cache = LLMCache(cache_path=cache_dir, mode=cache_mode)
        return CachedLLMClient(inner=inner, cache=cache, model_name=model_str)

    return inner


def _build_inner(model_str: str) -> BaseLLM:
    """Construct the raw (un-cached) LLM client."""
    if model_str in ("mock", "dummy"):
        from oml.llm.mock import MockLLM
        return MockLLM()

    if model_str == "smart-mock":
        from oml.llm.smart_mock import SmartMockLLM
        return SmartMockLLM()

    if model_str.startswith("openai:"):
        from oml.llm.openai import OpenAILLM
        _, name = model_str.split(":", 1)
        return OpenAILLM(name)

    if model_str.startswith("ollama:"):
        from oml.llm.ollama import OllamaLLM
        _, name = model_str.split(":", 1)
        return OllamaLLM(name)

    if model_str.startswith("gemini:"):
        from oml.llm.gemini import GeminiLLM
        _, name = model_str.split(":", 1)
        return GeminiLLM(name)

    if model_str.startswith("lmstudio:"):
        from oml.llm.lmstudio import LMStudioLLM
        _, name = model_str.split(":", 1)
        return LMStudioLLM(name)

    if model_str.startswith("openrouter:"):
        from oml.llm.openrouter import OpenRouterLLM
        _, name = model_str.split(":", 1)
        return OpenRouterLLM(name)

    # Unknown provider prefix should fail fast to avoid silent misconfiguration.
    if ":" in model_str:
        raise ValueError(
            "Unknown LLM provider prefix in "
            f"'{model_str}'. Use one of: mock, smart-mock, ollama:, openai:, "
            "gemini:, lmstudio:, openrouter:."
        )

    # Backward-compatible shorthand: treat bare model names as Ollama models.
    from oml.llm.ollama import OllamaLLM
    return OllamaLLM(model_str)
