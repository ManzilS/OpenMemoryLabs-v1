import re
from oml.llm.base import BaseLLM

class MockLLM(BaseLLM):
    """A mock LLM for testing without API keys."""
    def generate(self, prompt: str) -> str:
        # Lost-in-the-middle needle detection
        if "The secret code is" in prompt:
            match = re.search(r"The secret code is (\d+)", prompt)
            if match:
                return str(match.group(1))
        if "Blaxland" in prompt and "1813" in prompt:
            return "1813"
        return "This is a test response."
