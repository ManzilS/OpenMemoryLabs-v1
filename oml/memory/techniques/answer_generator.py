"""
AnswerGenerator — LLM-based answer generation from TEEG context.
=================================================================

Encapsulates the query-prompt construction and LLM generation call
that was previously inline in :meth:`TEEGPipeline.query`.
"""

from __future__ import annotations


class AnswerGenerator:
    """
    Generate an answer to a question given a TEEG context block.

    Parameters
    ----------
    llm_client:
        Any object exposing a ``generate(prompt: str) -> str`` method.
    """

    def __init__(self, llm_client):
        self._llm = llm_client

    def generate(self, question: str, context_str: str) -> str:
        """Build the query prompt and return the LLM's answer."""
        prompt = self._build_query_prompt(question, context_str)
        return self._llm.generate(prompt)

    @staticmethod
    def _build_query_prompt(question: str, context_str: str) -> str:
        return f"""You are a knowledgeable assistant with access to a structured memory system.
Use ONLY the information in the TEEG MEMORY block below to answer the question.
If the memory does not contain enough information, say so clearly.

{context_str}

QUESTION: {question}

ANSWER:"""
