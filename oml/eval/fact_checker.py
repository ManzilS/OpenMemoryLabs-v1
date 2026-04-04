from rdflib import Graph, URIRef, Literal, Namespace
import logging
from typing import List, Tuple
from oml.llm.factory import get_llm_client
import json

logger = logging.getLogger(__name__)

OML = Namespace("http://openmemorylab.org/ontology/")

class SemanticFactChecker:
    """
    Evaluator that builds an RDF graph of absolute ground-truth facts, 
    and checks generated responses against that semantic graph via SPARQL.
    """
    def __init__(self, use_llm: bool = True):
        self.graph = Graph()
        self.graph.bind("oml", OML)
        self.use_llm = use_llm
        
    def add_facts(self, facts: List[Tuple[str, str, str]]):
        """Adds explicit ground-truth facts to the RDF graph."""
        for subj, pred, obj in facts:
            s_fmt = subj.replace(' ', '_').replace('"', '').lower()
            p_fmt = pred.replace(' ', '_').replace('"', '').lower()
            s = URIRef(f"http://openmemorylab.org/entity/{s_fmt}")
            p = URIRef(f"http://openmemorylab.org/relation/{p_fmt}")
            o = Literal(obj)
            self.graph.add((s, p, o))
            
    def _extract_claims(self, text: str, model_name: str) -> List[Tuple[str, str, str]]:
        """Uses LLM to extract factual claims from generated text."""
        prompt = f"""Extract all factual claims from the following text into (Subject, Predicate, Object) triples.
Return ONLY a valid JSON list of lists of 3 strings. Do not include markdown or explanations.

Text: {text}
"""
        model = get_llm_client(model_name)
        try:
            resp = model.generate(prompt).strip()
            if resp.startswith("```json"): resp = resp[7:]
            if resp.startswith("```"): resp = resp[3:]
            if resp.endswith("```"): resp = resp[:-3]
            claims = json.loads(resp.strip())
            return [tuple(c) for c in claims if isinstance(c, list) and len(c) == 3]
        except Exception as e:
            logger.warning(f"Failed to extract claims: {e}")
            return []
            
    def verify(self, text: str, model_name: str) -> dict:
        """
        Extracts claims from the generated text and checks if they exist in the RDF graph.
        Returns a dictionary of verified/unverified/contradicted claims.
        """
        if not self.use_llm:
            return {"status": "error", "message": "LLM required for claim extraction"}
            
        claims = self._extract_claims(text, model_name)
        if not claims:
            return {"status": "no_claims_found", "score": 1.0, "total_claims": 0}
            
        results = {"verified": [], "unverified": []}
        
        for subj, pred, obj in claims:
            # Construct SPARQL query to see if subj + pred exists, and if obj matches
            s_fmt = subj.replace(' ', '_').replace('"', '').lower()
            p_fmt = pred.replace(' ', '_').replace('"', '').lower()
            s_uri = f"http://openmemorylab.org/entity/{s_fmt}"
            p_uri = f"http://openmemorylab.org/relation/{p_fmt}"
            
            query = f"""
            SELECT ?o
            WHERE {{
                <{s_uri}> <{p_uri}> ?o .
            }}
            """
            try:
                q_res = self.graph.query(query)
                found_objects = [str(row[0]).lower() for row in q_res]
                
                if not found_objects:
                    results["unverified"].append({"claim": (subj, pred, obj), "reason": "Subject or Predicate completely absent from Ground Truth RDF Graph"})
                else:
                    # Naive semantic match: if the object string is in the found objects or vice versa
                    o_lower = obj.lower()
                    matched = False
                    for f_o in found_objects:
                        if o_lower in f_o or f_o in o_lower:
                            matched = True
                            break
                    
                    if matched:
                        results["verified"].append({"claim": (subj, pred, obj)})
                    else:
                        results["unverified"].append({"claim": (subj, pred, obj), "reason": f"Contradicts RDF Graph. Graph says: {found_objects}"})
            except Exception as e:
                logger.warning(f"SPARQL error: {e}")
                results["unverified"].append({"claim": (subj, pred, obj), "reason": "SPARQL parse error"})
                    
        # Calculate boolean score
        total = len(results["verified"]) + len(results["unverified"])
        score = len(results["verified"]) / total if total > 0 else 0.0
        
        return {
            "status": "success",
            "score": score,
            "total_claims": total,
            "verified_claims": [c["claim"] for c in results["verified"]],
            "unverified_claims": results["unverified"]
        }
