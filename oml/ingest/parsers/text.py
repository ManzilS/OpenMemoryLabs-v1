from pathlib import Path
from datetime import datetime
from oml.models.schema import Document

def parse_text_file(file_path: Path) -> Document:
    """Parses a generic text file into a Document object."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback
        text = file_path.read_text(encoding="latin-1", errors="replace")
        
    doc_type = "text"
    if file_path.suffix.lower() in [
        '.py', '.js', '.ts', '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.php', '.html', '.css'
    ]:
        doc_type = "code"

    return Document(
        source=file_path.name,
        author="Unknown",
        recipients=[],
        subject=file_path.stem, # Use filename as subject
        timestamp=datetime.fromtimestamp(file_path.stat().st_mtime), # Use file mtime
        raw_text=text,
        clean_text=text, # No cleaning for raw text
        doc_type=doc_type
    )
