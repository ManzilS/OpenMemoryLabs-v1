import os
import requests
from oml.llm.base import BaseLLM

class OllamaLLM(BaseLLM):
    """Wrapper for Ollama API."""
    def __init__(self, model_name: str, base_url: str = None):
        self.model_name = model_name
        # Fallback to env var or default
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            if not response.ok:
                return f"Error calling Ollama ({response.status_code}): {response.text}"
            return response.json().get("response", "")
        except requests.RequestException as e:
            return f"Error calling Ollama: {e}"
