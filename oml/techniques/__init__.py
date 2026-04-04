"""Technique protocol definitions and registry for OpenMemoryLab.

This package defines the pluggable technique interfaces that memory pipelines
compose at runtime.  Each protocol represents one stage of the memory
lifecycle (ingest, retrieval, evolution, compression, seeding, walking).

Concrete implementations live in their natural home packages:

- ``oml.memory.techniques.*``    -- ingest, evolution, compression
- ``oml.retrieval.techniques.*`` -- retrieval, seeding, walking

The :class:`TechniqueRegistry` in ``oml.techniques.registry`` maps
human-readable names to implementations and ships a handful of built-in
presets (``"teeg"``, ``"prism"``, ``"lightweight"``).
"""

from oml.techniques.protocols import (
    CompressionTechnique,
    EvolutionTechnique,
    IngestTechnique,
    RetrievalTechnique,
    SeedingTechnique,
    WalkingTechnique,
)
from oml.techniques.registry import PRESETS, TechniqueRegistry

__all__ = [
    "CompressionTechnique",
    "EvolutionTechnique",
    "IngestTechnique",
    "RetrievalTechnique",
    "SeedingTechnique",
    "WalkingTechnique",
    "TechniqueRegistry",
    "PRESETS",
]
