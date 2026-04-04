import json
import logging
from typing import List, Tuple
from oml.llm.factory import get_llm_client

logger = logging.getLogger(__name__)

def extract_triples(text: str, model_name: str) -> List[Tuple[str, str, str]]:
    """
    Extracts knowledge graph triples (Subject, Predicate, Object) from the given text.
    """
    prompt = f"""Extract all factual entity relationships from the following text. 
Return ONLY a valid JSON list of lists, where each inner list has exactly three strings: [Subject, Predicate, Object].
Do not include markdown blocks, explanations, or any other text. If no facts are found, return [].

Example format:
[
  ["Victor Frankenstein", "created", "the monster"],
  ["The monster", "demanded", "a female companion"]
]

Text:
{text}
"""
    try:
        model = get_llm_client(model_name)
        response = model.generate(prompt)
        
        # Clean response string to ensure it parses as JSON
        clean_resp = response.strip()
        if clean_resp.startswith("```json"):
            clean_resp = clean_resp[7:]
        if clean_resp.startswith("```"):
            clean_resp = clean_resp[3:]
        if clean_resp.endswith("```"):
            clean_resp = clean_resp[:-3]
        clean_resp = clean_resp.strip()
            
        triples = json.loads(clean_resp)
        valid_triples = [tuple(t) for t in triples if isinstance(t, list) and len(t) == 3]
        return valid_triples
    except Exception as e:
        logger.warning(f"[GraphExtractor] Failed to extract triples: {e}")
        return []
