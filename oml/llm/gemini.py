import os
try:
    from google import genai
except ImportError:
    genai = None
from oml.llm.base import BaseLLM

class GeminiLLM(BaseLLM):
    """Wrapper for Google Gemini API."""
    def __init__(self, model_name: str):
        if genai is None:
            raise ImportError("Please install google-genai: pip install google-genai")
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            if hasattr(response, 'text'):
                return response.text
            return "Error: No text in response."
        except Exception as e:
            return f"Error calling Gemini: {e}"
