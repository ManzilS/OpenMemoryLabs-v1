import pytest
from oml.models.schema import Document
from oml.ingest.chunkers.text import RecursiveStrategy
from oml.ingest.chunkers.code import CodeStrategy
from oml.ingest.chunkers import get_strategy, segment_document

def test_get_strategy():
    assert isinstance(get_strategy("text"), RecursiveStrategy)
    assert isinstance(get_strategy("code"), CodeStrategy)

def test_recursive_strategy_short_text():
    doc = Document(doc_id="test1", clean_text="Short text here.", source="test")
    strategy = RecursiveStrategy()
    chunks = strategy.segment(doc, min_size=10, max_size=100)
    
    assert len(chunks) == 1
    assert chunks[0].chunk_text == "Short text here."
    assert chunks[0].chunk_id == "test1_c0"

def test_recursive_strategy_long_text():
    # Construct a long text with clear paragraph breaks
    paragraphs = [
        "This is the first paragraph. It has some reasonable length but not too long.",
        "This is the second paragraph. We want to test if it gets split correctly.",
        "Here is the third paragraph. It should be in a separate chunk."
    ]
    long_text = "\n\n".join(paragraphs)
    doc = Document(doc_id="test2", clean_text=long_text, source="test")
    strategy = RecursiveStrategy()
    
    # Very small max size forces a split at every paragraph
    chunks = strategy.segment(doc, min_size=10, max_size=100)
    
    assert len(chunks) == 3
    assert chunks[0].chunk_text.strip() == paragraphs[0]
    assert chunks[1].chunk_text.strip() == paragraphs[1]
    assert chunks[2].chunk_text.strip() == paragraphs[2]

def test_recursive_strategy_extremely_long_word():
    # If there are no spaces or newlines, it must forcibly split
    doc = Document(doc_id="test3", clean_text="A" * 250, source="test")
    strategy = RecursiveStrategy()
    
    chunks = strategy.segment(doc, min_size=10, max_size=100)
    
    assert len(chunks) == 3
    assert len(chunks[0].chunk_text) == 100
    assert len(chunks[1].chunk_text) == 100
    assert len(chunks[2].chunk_text) == 50

def test_code_strategy_basic():
    code_text = (
        "def func1():\n"
        "    print('hello')\n"
        "\n"
        "def func2():\n"
        "    print('world')\n"
        "    print('!')\n"
    )
    doc = Document(doc_id="code1", clean_text=code_text, source="test.py", doc_type="code")
    strategy = CodeStrategy()
    
    chunks = strategy.segment(doc, min_size=10, max_size=100)
    
    # We expect 2 chunks, one for each function based on top-level indent split
    assert len(chunks) == 2
    assert "def func1():" in chunks[0].chunk_text
    assert "def func2():" in chunks[1].chunk_text

def test_segment_document_wrapper():
    doc = Document(doc_id="test3", clean_text="Just some text.", source="test")
    chunks = segment_document(doc, min_size=5, max_size=50)
    assert len(chunks) == 1
    assert chunks[0].chunk_text == "Just some text."
