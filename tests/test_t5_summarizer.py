import pytest
from unittest.mock import patch, MagicMock
from oml.ingest.t5_summarizer import T5Summarizer
from oml.models.schema import Document

def test_t5_summarizer_initialization():
    """Test that the summarizer initializes without error"""
    with patch("oml.ingest.t5_summarizer._load_model") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())
        summarizer = T5Summarizer()
        mock_load.assert_called_once()
        assert summarizer is not None

def test_t5_summarize_document():
    """Test that summarize_document correctly uses the mock model."""
    with patch("oml.ingest.t5_summarizer._load_model") as mock_load:
        # Setup mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Simulating generation output
        mock_model.generate.return_value = [[1, 2, 3]]
        
        # Simulating decoding output
        mock_tokenizer.decode.return_value = "This is a brief summary of a longer text."
        
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        summarizer = T5Summarizer()
        
        doc = Document(
            doc_id="test_doc_1",
            source="test.txt",
            clean_text="This is a very long text. " * 30
        )
        
        summary = summarizer.summarize_document(doc)
        
        # Verify the mock model was used
        assert mock_model.generate.called
        assert mock_tokenizer.decode.called
        
        assert summary == "This is a brief summary of a longer text."
