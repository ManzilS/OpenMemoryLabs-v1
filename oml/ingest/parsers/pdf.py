"""
PDF parser for OpenMemoryLab.

Uses ``pypdf`` (pure-Python, no system dependencies) when available.
Falls back gracefully with a clear error message if it is not installed.

Install:  pip install pypdf
"""
from pathlib import Path
from datetime import datetime

from oml.models.schema import Document


def parse_pdf_file(file_path: Path) -> Document:
    """
    Parses a PDF file into a Document object by extracting its text content.

    Each page's text is separated by a newline for readability.

    Raises:
        ImportError: if ``pypdf`` is not installed.
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a readable PDF.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to parse PDF files. "
            "Install it with:  pip install pypdf"
        ) from exc

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise ValueError(f"Could not read PDF '{file_path.name}': {exc}") from exc

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
            pages_text.append(text)
        except Exception:
            # Skip unreadable pages rather than failing the whole document
            pages_text.append("")

    full_text = "\n".join(pages_text).strip()

    # Attempt to extract a title from PDF metadata; fall back to filename stem
    meta = reader.metadata or {}
    title = (
        meta.get("/Title", "") or meta.get("title", "") or file_path.stem
    )
    author = meta.get("/Author", "") or meta.get("author", "") or "Unknown"

    return Document(
        source=file_path.name,
        author=author,
        recipients=[],
        subject=title,
        timestamp=datetime.fromtimestamp(file_path.stat().st_mtime),
        raw_text=full_text,
        clean_text=full_text,
        doc_type="pdf",
    )
