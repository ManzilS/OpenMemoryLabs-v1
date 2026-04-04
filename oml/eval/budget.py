"""oml/eval/budget.py — Pre-flight experiment cost estimation.

``ExperimentBudgetPlanner`` estimates how many LLM calls an experiment will
make, how many can be served from cache, and roughly what the remaining calls
will cost in USD.

Knows about the following operations (call counts per operation):

  ``"teeg-ingest"``        1 distil + up to 5 judge calls per text  → 6/text
  ``"prism-ingest"``       same as teeg-ingest (no batching in single mode)
  ``"prism-batch"``        2 calls per batch of N texts (1 distil + 1 evolve)
  ``"teeg-query"``         1 call (generation after Scout retrieval)
  ``"prism-query"``        1 call (delegates to TEEGPipeline)
  ``"eval-faithfulness"``  3 calls (3 synthetic Q&A judgments)
  ``"eval-lost-in-middle"``3 calls (needle at 3 positions)
  ``"eval-ablations"``     3 × n_queries calls (default n_queries=5 → 15)
  ``"eval-oml-vs-rag"``    2 × n_queries calls (default n_queries=5 → 10)
  ``"eval-retrieval-precision"`` 0 calls (keyword-based, no LLM)

Usage::

    from oml.eval.budget import ExperimentBudgetPlanner

    planner = ExperimentBudgetPlanner()
    est = planner.estimate("teeg-ingest", n_texts=20, model="openai:gpt-4o-mini")
    print(est)
    planner.pre_flight(est)   # prompts [y/N]; raises RunAborted on decline
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Pricing (same table as oml/llm/cache.py — kept in sync manually)
# ---------------------------------------------------------------------------
_PRICE_PER_1K_IN: dict[str, float] = {
    "gpt-4o-mini":       0.000150,
    "gpt-4o":            0.005000,
    "gpt-4-turbo":       0.010000,
    "gemini-1.5-flash":  0.000075,
    "gemini-1.5-pro":    0.001250,
    "gemini-2.0-flash":  0.000100,
    "claude-3-haiku":    0.000250,
    "claude-3-sonnet":   0.003000,
    "claude-3-opus":     0.015000,
}

_AVG_PROMPT_TOKENS = 400  # conservative average across distil + judge calls


def _price_per_call(model: str) -> float:
    for prefix, price in _PRICE_PER_1K_IN.items():
        if prefix in model:
            return price * _AVG_PROMPT_TOKENS / 1000
    return 0.0  # local/unknown → free


# ---------------------------------------------------------------------------
# Call-count formulas
# ---------------------------------------------------------------------------

def _calls_naive(operation: str, n_texts: int, n_queries: int) -> int:
    """Worst-case (un-optimised) call count."""
    op = operation.lower().replace("_", "-")
    if op in ("teeg-ingest", "prism-ingest"):
        return 6 * n_texts          # 1 distil + 5 judge per text
    if op == "prism-batch":
        return 2 * n_texts          # 1 distil + 1 evolve per text (naive)
    if op in ("teeg-query", "prism-query"):
        return 1 * n_queries
    if op == "eval-faithfulness":
        return 3
    if op == "eval-lost-in-middle":
        return 3
    if op == "eval-ablations":
        return 3 * n_queries
    if op == "eval-oml-vs-rag":
        return 2 * n_queries
    if op in ("eval-retrieval-precision", "eval-retrieval_precision"):
        return 0
    # Generic: treat as 1 call per text
    return max(1, n_texts)


def _calls_optimised(operation: str, n_texts: int, n_queries: int) -> int:
    """Call count with PRISM batching applied."""
    op = operation.lower().replace("_", "-")
    if op == "prism-batch":
        # 2 calls per batch of n_texts (1 distil batch + 1 evolve batch)
        return 2
    if op == "teeg-ingest":
        # No batching in TEEG; same as naive
        return 6 * n_texts
    if op == "prism-ingest":
        # Single mode: still 1 distil + up to 5 judge, but SketchGate may skip
        return max(1, 6 * n_texts)
    # Everything else unchanged
    return _calls_naive(operation, n_texts, n_queries)


# ---------------------------------------------------------------------------
# BudgetEstimate
# ---------------------------------------------------------------------------

@dataclass
class BudgetEstimate:
    """Result of ``ExperimentBudgetPlanner.estimate()``."""

    operation: str
    n_texts: int
    model: str
    total_calls_naive: int
    total_calls_optimized: int
    cached_calls: int          # 0 unless cache_warm=True
    api_calls_needed: int      # total_calls_optimized - cached_calls
    cost_estimate_usd: float   # api_calls_needed × price_per_call

    def savings_pct(self) -> float:
        """Percentage saved vs naive (including cache + PRISM)."""
        if self.total_calls_naive == 0:
            return 100.0
        actual = self.api_calls_needed
        return max(0.0, 100.0 * (1 - actual / self.total_calls_naive))

    def __str__(self) -> str:
        model_label = self.model if self.model else "unknown"
        is_free = self.cost_estimate_usd == 0.0
        cost_str = "$0.00 (free/local)" if is_free else f"~${self.cost_estimate_usd:.4f}"
        lines = [
            "╔═══════════════════════════════════════════════════════╗",
            "║           Experiment Budget Pre-Flight Check          ║",
            "╠═══════════════════════════════════════════════════════╣",
            f"║  Operation : {self.operation:<40}║",
            f"║  Model     : {model_label:<40}║",
            f"║  N texts   : {str(self.n_texts):<40}║",
            "╠═══════════════════════════════════════════════════════╣",
            f"║  Calls (naive, no PRISM) : {str(self.total_calls_naive):<28}║",
            f"║  Calls (optimised)       : {str(self.total_calls_optimized):<28}║",
            f"║  Served from cache       : {str(self.cached_calls):<28}║",
            f"║  API calls needed        : {str(self.api_calls_needed):<28}║",
            f"║  Estimated cost          : {cost_str:<28}║",
            f"║  Total savings           : {f'{self.savings_pct():.1f}%':<28}║",
            "╚═══════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RunAborted
# ---------------------------------------------------------------------------

class RunAborted(RuntimeError):
    """Raised by ``pre_flight()`` when the user declines to proceed."""


# ---------------------------------------------------------------------------
# ExperimentBudgetPlanner
# ---------------------------------------------------------------------------

class ExperimentBudgetPlanner:
    """Pre-flight cost estimator for OpenMemoryLab experiments.

    Example::

        planner = ExperimentBudgetPlanner()
        est = planner.estimate("prism-batch", n_texts=32, model="openai:gpt-4o-mini")
        print(est)
        planner.pre_flight(est)   # [y/N] prompt
    """

    def estimate(
        self,
        operation: str,
        n_texts: int = 1,
        model: str = "mock",
        cache_warm: bool = False,
        n_queries: int = 5,
    ) -> BudgetEstimate:
        """Estimate the cost of running an experiment.

        Args:
            operation:   One of the recognised operation strings (see module docstring).
            n_texts:     Number of texts/documents to ingest (for ingest operations).
            model:       Model string (e.g. ``"openai:gpt-4o-mini"``).
            cache_warm:  Set ``True`` if the cache already contains responses for this
                         exact experiment (all optimised calls become cache hits → $0).
            n_queries:   Number of queries for query/eval operations.

        Returns:
            ``BudgetEstimate`` dataclass with all cost fields populated.
        """
        naive = _calls_naive(operation, n_texts, n_queries)
        optimised = _calls_optimised(operation, n_texts, n_queries)
        cached = optimised if cache_warm else 0
        needed = max(0, optimised - cached)
        price = _price_per_call(model)
        cost = needed * price

        return BudgetEstimate(
            operation=operation,
            n_texts=n_texts,
            model=model,
            total_calls_naive=naive,
            total_calls_optimized=optimised,
            cached_calls=cached,
            api_calls_needed=needed,
            cost_estimate_usd=cost,
        )

    def pre_flight(
        self,
        estimate: BudgetEstimate,
        auto_confirm: bool = False,
    ) -> None:
        """Print the estimate and ask for confirmation.

        Args:
            estimate:      A ``BudgetEstimate`` from ``estimate()``.
            auto_confirm:  Skip the prompt and proceed automatically (``--yes`` flag).

        Raises:
            RunAborted: If the user enters anything other than ``y``/``yes``.
        """
        print(str(estimate))

        if auto_confirm or estimate.api_calls_needed == 0:
            print("✓ Auto-confirmed (no API calls needed)." if estimate.api_calls_needed == 0
                  else "✓ Auto-confirmed (--yes flag).")
            return

        try:
            answer = input("\nProceed with this experiment? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer not in ("y", "yes"):
            raise RunAborted(
                f"Experiment aborted by user (would have used {estimate.api_calls_needed} API calls)."
            )
        print("✓ Confirmed. Running experiment…")
