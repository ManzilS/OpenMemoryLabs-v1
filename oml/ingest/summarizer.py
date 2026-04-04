from oml.models.schema import Document
from oml.llm import get_llm_client

class Summarizer:
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        # Lazy load client or load immediately
        try:
            self.client = get_llm_client(model_name)
        except Exception as e:
             # Fallback to mock if model not found/configured?
             # Or raise error
             print(f"Warning: Failed to load LLM '{model_name}': {e}. Falling back to mock.")
             self.client = get_llm_client("mock")
        
    def summarize_document(self, doc: Document) -> str:
        """
        Summarizes the document text using the configured LLM.
        """
        text = doc.clean_text
        if not text:
            return ""
            
        # Simple truncation for now to avoid blowing up context on large files
        # A better approach would be to check token count or use a "map-reduce" style for large docs
        max_chars = 10000 
        if len(text) > max_chars:
            text = text[:max_chars] + "...[TRUNCATED]"

        prompt = f"Please provide a concise summary of the following document:\n\n{text}"
        
        try:
            summary = self.client.generate(prompt)
            return summary.strip()
        except Exception as e:
            # Fallback or log error
            print(f"Summarization failed for {doc.doc_id} with model {self.model_name}: {e}")
            return ""
