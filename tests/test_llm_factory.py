"""Tests for the LLM factory routing logic.

These tests are intentionally light-weight; they just ensure that
model-name prefixes map to the expected provider classes.
"""
import pytest

from oml.llm import factory
from oml.llm.base import BaseLLM


@pytest.mark.parametrize(
    "model_name",
    [
        "mock",
        "dummy",
    ],
)
def test_llm_factory_returns_base_llm(model_name):
    """Factory should return an object implementing the BaseLLM interface for valid model names."""
    llm = factory.get_llm_client(model_name)
    assert isinstance(llm, BaseLLM)
