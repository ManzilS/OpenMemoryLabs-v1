from oml.eval.base import ModelInterface
from oml.llm import get_llm_client

# Adapters to maintain ModelInterface compatibility if needed (BaseLLM has generate(prompt), ModelInterface has generate(prompt))
# They are identical signatures, so we can just alias or wrap.

def get_model(model_str: str) -> ModelInterface:
    """
    Factory function to get a model instance (Backward compatibility wrapper).
    """
    return get_llm_client(model_str)
