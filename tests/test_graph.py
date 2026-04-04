import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from oml.retrieval.graph_retriever import GraphRetriever
from oml.ingest.graph_extractor import extract_triples

@pytest.fixture
def temp_graph_dir(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    return artifacts_dir

def test_extract_triples_valid():
    # Mock LLM generation to return a valid JSON array of triples
    with patch('oml.ingest.graph_extractor.get_llm_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate.return_value = '```json\n[["Frankenstein", "created", "The Monster"], ["Walton", "met", "Frankenstein"]]\n```'
        mock_get_client.return_value = mock_client
        
        triples = extract_triples("Some text here", "fake_model")
        
        assert len(triples) == 2
        assert triples[0] == ("Frankenstein", "created", "The Monster")
        assert triples[1] == ("Walton", "met", "Frankenstein")

def test_extract_triples_invalid_json():
    # If LLM hallucinates non-JSON, we should safely return an empty list
    with patch('oml.ingest.graph_extractor.get_llm_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate.return_value = "This is not JSON at all."
        mock_get_client.return_value = mock_client
        
        triples = extract_triples("Some text here", "fake_model")
        
        assert isinstance(triples, list)
        assert len(triples) == 0

def test_graph_retriever_add_and_search(temp_graph_dir):
    retriever = GraphRetriever(temp_graph_dir)
    triples = [
        ("Victor", "built", "Creature"),
        ("Victor", "lives in", "Geneva"),
        ("Creature", "fled to", "Mountains")
    ]
    retriever.add_triples(triples)
    
    # Query mentioning Victor should pull his 1-hop neighborhood
    with patch('oml.retrieval.graph_retriever.get_llm_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate.return_value = '["Victor"]' # Mock LLM extracting entity from query
        mock_get_client.return_value = mock_client
        
        context = retriever.search_graph("Tell me about Victor", "fake_model")
        
        assert "- Victor --[built]--> Creature" in context
        assert "- Victor --[lives in]--> Geneva" in context
        assert "Creature fled to Mountains" not in context # 1-hop only from Victor
        
def test_graph_retriever_persistence(temp_graph_dir):
    retriever1 = GraphRetriever(temp_graph_dir)
    retriever1.add_triples([("A", "knows", "B")])
    retriever1.save()
    
    retriever2 = GraphRetriever(temp_graph_dir)
    success = retriever2.load()
    
    assert success is True
    assert retriever2.graph.has_edge("a", "b")
