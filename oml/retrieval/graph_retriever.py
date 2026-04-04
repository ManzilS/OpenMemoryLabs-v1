import networkx as nx
import pickle
import json
from pathlib import Path
from typing import List, Tuple
from oml.llm.factory import get_llm_client
import logging

logger = logging.getLogger(__name__)

class GraphRetriever:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.graph_path = artifacts_dir / "knowledge_graph.pkl"
        self.graph = nx.DiGraph()
        self._loaded = False

    def load(self) -> bool:
        if self.graph_path.exists():
            try:
                with open(self.graph_path, "rb") as f:
                    self.graph = pickle.load(f)
                self._loaded = True
                return True
            except Exception as e:
                logger.error(f"[GraphRetriever] Failed to load graph: {e}")
                return False
        return False

    def save(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)
            
    def get_all_nodes(self) -> List[str]:
        return list(self.graph.nodes)

    def add_triples(self, triples: List[Tuple[str, str, str]]):
        """Adds triples to the NetworkX directed graph."""
        for subj, pred, obj in triples:
            # Simple normalization
            s = subj.strip().lower()
            o = obj.strip().lower()
            if not s or not o: continue
            
            # Add nodes if they don't exist
            if not self.graph.has_node(s):
                self.graph.add_node(s, label=subj.strip())
            if not self.graph.has_node(o):
                self.graph.add_node(o, label=obj.strip())
                
            # Add edge
            self.graph.add_edge(s, o, relation=pred.strip())

    def _extract_entities_from_query(self, query: str, model_name: str) -> List[str]:
        prompt = f"""Extract the key entities (people, places, concepts) from the following query.
Return ONLY a valid JSON list of strings. Do not include markdown or explanations.

Query: {query}
"""
        try:
            model = get_llm_client(model_name)
            response = model.generate(prompt)
            clean_resp = response.strip()
            if clean_resp.startswith("```json"): clean_resp = clean_resp[7:]
            if clean_resp.startswith("```"): clean_resp = clean_resp[3:]
            if clean_resp.endswith("```"): clean_resp = clean_resp[:-3]
            entities = json.loads(clean_resp.strip())
            return [str(e).lower().strip() for e in entities if isinstance(e, str)]
        except Exception as e:
            logger.warning(f"[GraphRetriever] Entity extraction failed: {e}")
            # Fallback: exact substring match
            words = [w.lower().strip(",.?!") for w in query.split()]
            return [w for w in words if len(w) > 3]

    def search_graph(self, query: str, model_name: str, hop_depth: int = 1) -> str:
        """
        Extracts entities from query, finds them in the graph, and returns their neighborhood context.
        """
        if not self._loaded:
            self.load()
            
        if self.graph.number_of_nodes() == 0:
            return ""

        entities = self._extract_entities_from_query(query, model_name)
        found_nodes = [node for node in entities if self.graph.has_node(node)]
        
        if not found_nodes:
            # Try partial matches if strict fails
            for node in self.graph.nodes:
                for e in entities:
                    if e in node or node in e:
                        found_nodes.append(node)
                        break

        if not found_nodes:
            return ""
            
        neighborhood_facts = set()
        for node in set(found_nodes):
            # Outgoing edges
            for target in self.graph.successors(node):
                edge_data = self.graph.get_edge_data(node, target)
                rel = edge_data.get('relation', 'related to')
                n_label = self.graph.nodes[node].get('label', node)
                t_label = self.graph.nodes[target].get('label', target)
                neighborhood_facts.add(f"{n_label} --[{rel}]--> {t_label}")
                
            # Incoming edges
            for source in self.graph.predecessors(node):
                edge_data = self.graph.get_edge_data(source, node)
                rel = edge_data.get('relation', 'related to')
                s_label = self.graph.nodes[source].get('label', source)
                n_label = self.graph.nodes[node].get('label', node)
                neighborhood_facts.add(f"{s_label} --[{rel}]--> {n_label}")

        if not neighborhood_facts:
            return ""

        return "[KNOWLEDGE GRAPH CONTEXT]\n- " + "\n- ".join(neighborhood_facts)
