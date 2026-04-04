from __future__ import annotations
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a response for a given prompt."""
        pass

    def generate_many(self, prompts: List[str], max_workers: int = 8) -> List[str]:
        """Run *N* prompts concurrently using a thread pool.

        Each call to ``generate()`` runs in its own thread so that network I/O
        (OpenAI, Gemini, Ollama, ...) overlaps.  Ordering is preserved -- the
        returned list is index-aligned with *prompts*.

        Parameters
        ----------
        prompts:
            List of prompt strings to send to the LLM.
        max_workers:
            Maximum concurrent threads.  Capped at ``len(prompts)`` automatically.

        Returns
        -------
        List[str]
            Responses in the same order as *prompts*.
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0])]
        workers = min(max_workers, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.generate, prompts))
