from oml.llm.base import BaseLLM

class OpenAILLM(BaseLLM):
    """Wrapper for OpenAI API."""
    def __init__(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.client = OpenAI() # Uses OPENAI_API_KEY env var by default
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content or ""
