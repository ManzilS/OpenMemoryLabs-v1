from pathlib import Path
from oml.models.schema import Document
from oml.ingest.parsers.email import parse_email_file
from oml.ingest.parsers.text import parse_text_file
from oml.ingest.parsers.pdf import parse_pdf_file

# Extension-to-parser registry.  Register new parsers here.
_PARSER_REGISTRY: dict[str, object] = {
    ".eml": parse_email_file,
    ".pdf": parse_pdf_file,
}


def get_parser_for(file_path: Path):
    """
    Returns the appropriate parser callable for a given file path.

    Falls back to the generic text parser when no specific parser is registered
    for the file's extension.
    """
    ext = file_path.suffix.lower()
    return _PARSER_REGISTRY.get(ext, parse_text_file)


def general_parse(file_path: Path) -> Document:
    """Dispatches to the correct parser based on file extension."""
    parser = get_parser_for(file_path)
    return parser(file_path)


__all__ = [
    "parse_email_file",
    "parse_text_file",
    "parse_pdf_file",
    "get_parser_for",
    "general_parse",
]
