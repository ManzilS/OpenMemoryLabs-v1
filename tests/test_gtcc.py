import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from oml.retrieval.gtcc import GTCCRetriever

def test_gtcc_initialization(tmp_path):
    """Test that GTCC initializes correctly with a temporary directory"""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    
    # We just need to ensure the class loads
    with patch("oml.retrieval.gtcc.ProvenanceIndex") as mock_pi:
        retriever = GTCCRetriever(artifacts_dir)
        
        # Should have initialized both underlying indices
        mock_pi.assert_called_once_with(artifacts_dir)


def test_gtcc_expand_results(tmp_path):
    """Test the bridge-discovery algorithm logically"""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    
    with patch("oml.retrieval.gtcc.ProvenanceIndex") as mock_pi:
            
            # Set up mock Provenance Index
            mock_pi_instance = mock_pi.return_value
            
            # Mock find_bridge_chunks to return a fake bridge
            # The signature returns a list of (chunk_id, bridge_score) tuples
            mock_pi_instance.find_bridge_chunks.return_value = [("chunk_BRIDGE", 5.0)]
            
            retriever = GTCCRetriever(artifacts_dir)
            
            # Seed the expansion with chunk_A
            results = retriever.expand_results(["chunk_A"], max_bridges=2)
            
            # Expected output is a list of tuples: (chunk_id, Score, Source)
            # The seed chunk should always be mapped as 'retrieved' with score 1.0
            seed_result = next((r for r in results if r[0] == "chunk_A"), None)
            assert seed_result is not None
            assert seed_result[2] == "retrieved"
            
            # We expect "chunk_BRIDGE" to be discovered because it shares "Rare Item"
            bridge_result = next((r for r in results if r[0] == "chunk_BRIDGE"), None)
            assert bridge_result is not None
            assert bridge_result[2] == "bridge"
            
            # We expect bridge_result score to be > 0
            assert bridge_result[1] > 0.0
