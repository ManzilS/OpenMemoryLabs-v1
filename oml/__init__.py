"""OpenMemoryLab (oml) package."""

__all__ = ["main"]
__version__ = "1.0.0"

# Provide a convenient import for the CLI entrypoint
from .cli import main  # noqa: E402, F401
