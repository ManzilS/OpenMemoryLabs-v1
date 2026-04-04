import pytest
from oml.ingest.rebel_extractor import extract_triples_rebel, _parse_rebel_output

def test_parse_rebel_output_basic():
    """Test the parser can extract simple tuples from raw REBEL special tokens."""
    # This is a simulation of Babelscape/rebel-large raw token output
    raw_text = "<triplet> Robert Walton <subj> sister <obj> Margaret Saville"
    triples = _parse_rebel_output(raw_text)
    
    assert len(triples) == 1
    assert triples[0][0] == "Robert Walton"
    assert triples[0][1] == "sister"
    assert triples[0][2] == "Margaret Saville"

def test_parse_rebel_output_multiple():
    """Test the parser can handle multiple sequential triplets."""
    raw_text = (
        "<triplet> Enron <subj> location <obj> Houston "
        "<triplet> Enron <subj> industry <obj> Energy"
    )
    triples = _parse_rebel_output(raw_text)
    
    assert len(triples) == 2
    assert triples[0] == ("Enron", "location", "Houston")
    assert triples[1] == ("Enron", "industry", "Energy")

def test_extract_triples_rebel_mocked():
    """Test the main extraction wrapper with a mocked transformers model."""
    from unittest.mock import patch, MagicMock
    
    with patch("oml.ingest.rebel_extractor._load_model") as mock_load:
        # Setup mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # When model.generate is called, return fake token IDs
        mock_model.generate.return_value = [[1, 2, 3]]
        
        # When tokenizer.batch_decode is called, return the fake special phrase
        mock_tokenizer.batch_decode.return_value = ["<triplet> Apple <subj> founder <obj> Steve Jobs"]
        
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        triples = extract_triples_rebel("Apple was founded by Steve Jobs.")
        
        assert len(triples) == 1
        # The parser returns tuples, not dicts
        assert triples[0] == ("Apple", "founder", "Steve Jobs")
