"""
Technique modules for the TEEG memory pipeline.
================================================

Each module encapsulates one stage of the pipeline:

- ``stage1_prescreen``      -- Fast 3B-model pre-screen (YES / SCOPE? / NO)
- ``stage2_judge``          -- Full LLM judge (CONTRADICTS / EXTENDS / SUPPORTS / UNRELATED)
- ``confidence_engine``     -- Bayesian confidence decay and boost application
- ``belief_propagation``    -- Single-hop directed belief propagation
- ``llm_distiller``         -- LLM-based raw-text → AtomicNote distillation
- ``heuristic_distiller``   -- Rule-based fallback distillation (no LLM)
- ``answer_generator``      -- LLM answer generation from TEEG context
"""

from oml.memory.techniques.stage1_prescreen import Stage1PreScreen
from oml.memory.techniques.stage2_judge import Stage2Judge
from oml.memory.techniques.confidence_engine import ConfidenceEngine
from oml.memory.techniques.belief_propagation import BeliefPropagator
from oml.memory.techniques.llm_distiller import LLMDistiller
from oml.memory.techniques.heuristic_distiller import HeuristicDistiller
from oml.memory.techniques.answer_generator import AnswerGenerator

__all__ = [
    "Stage1PreScreen",
    "Stage2Judge",
    "ConfidenceEngine",
    "BeliefPropagator",
    "LLMDistiller",
    "HeuristicDistiller",
    "AnswerGenerator",
]
