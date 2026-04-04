"""
Retrieval Precision@K evaluation task.

Uses a small, self-contained labeled dataset (query -> relevant chunk keywords)
so it runs without any external data, but the dataset is structured to match
content ingested from Frankenstein or the Enron demo corpus.

Metric: Precision@K  =  |relevant ∩ retrieved_top_k| / K
"""
from pathlib import Path
from typing import Any

from oml.eval.base import EvalTask, EvalResult, ModelInterface
from oml.eval.run import register_task
from oml.retrieval.hybrid import HybridRetriever
from oml.storage.factory import get_storage


# Each entry has:
#   q          – the natural-language query
#   keywords   – substrings that MUST appear in a relevant chunk's text
#   min_hits   – how many of the top-K results must contain at least one keyword
#                to count as a "hit" (we set min_hits = 1 for a binary hit/miss per result)
LABELED_DATASET = [
    {
        # Expects chunks from ingested Alan Turing / Bletchley Park articles
        "q": "Who worked at Bletchley Park to break the Enigma cipher during World War II?",
        "keywords": ["turing", "bletchley", "enigma", "cipher", "codebreak"],
        "description": "Turing Bletchley Park query",
    },
    {
        # Expects chunks from ingested Blue Mountains exploration article
        "q": "Which explorers first crossed the Blue Mountains in Australia and in what year?",
        "keywords": ["blaxland", "lawson", "wentworth", "mountains", "1813"],
        "description": "Blue Mountains crossing query",
    },
    {
        # Expects chunks from ingested Birmingham Campaign article
        "q": "What tactics were used during the 1963 Birmingham civil rights campaign?",
        "keywords": ["birmingham", "civil", "rights", "segregation", "demonstrators", "nonviolent"],
        "description": "Birmingham Campaign query",
    },
    {
        # Expects chunks from ingested Islamic calendar article
        "q": "How many days are in the Islamic lunar calendar and how many months does it have?",
        "keywords": ["islamic", "lunar", "calendar", "months", "354", "hijri"],
        "description": "Islamic calendar query",
    },
    {
        # Expects chunks from ingested Ovambo people article
        "q": "What percentage of Namibia's population are the Ovambo people?",
        "keywords": ["ovambo", "namibia", "population", "ethnic", "percent"],
        "description": "Ovambo Namibia population query",
    },
]


@register_task("retrieval_precision")
class RetrievalPrecisionTask(EvalTask):
    """
    Measures Precision@K of the hybrid retriever against a small labeled set.

    A retrieved chunk is counted as a hit if its text contains at least one of
    the expected keywords (case-insensitive).  We report both the mean P@K across
    all queries and per-query breakdown in ``details``.
    """

    name = "retrieval_precision"

    def run(self, model: ModelInterface, config: dict[str, Any]) -> EvalResult:
        top_k: int = config.get("top_k", 5)
        alpha: float = config.get("alpha", 0.5)

        # ── setup retriever ──────────────────────────────────────────────────
        try:
            artifacts_dir = Path("artifacts")
            if not artifacts_dir.exists():
                return EvalResult(
                    self.name,
                    0.0,
                    {"error": "No artifacts/ directory found. Run `oml ingest` first."},
                )

            retriever = HybridRetriever(artifacts_dir)
            storage = get_storage("sqlite")

        except Exception as exc:
            return EvalResult(self.name, 0.0, {"error": f"Setup failed: {exc}"})

        # ── evaluate ─────────────────────────────────────────────────────────
        per_query: dict[str, Any] = {}
        precision_scores: list[float] = []

        for item in LABELED_DATASET:
            query = item["q"]
            keywords = [kw.lower() for kw in item["keywords"]]

            try:
                results = retriever.search(query, top_k=top_k, alpha=alpha)
            except Exception as exc:
                per_query[query] = {"error": str(exc), "precision": 0.0}
                precision_scores.append(0.0)
                continue

            if not results:
                per_query[query] = {"hits": 0, "retrieved": 0, "precision": 0.0}
                precision_scores.append(0.0)
                continue

            # Fetch chunk texts to inspect content
            chunk_ids = [r.chunk_id for r in results]
            try:
                chunks = storage.get_chunks_by_ids(chunk_ids)
            except Exception:
                chunks = []

            chunk_text_map = {c.chunk_id: c.chunk_text.lower() for c in chunks if c}

            hits = 0
            result_details: list[dict] = []
            for r in results:
                text = chunk_text_map.get(r.chunk_id, "")
                is_hit = any(kw in text for kw in keywords)
                if is_hit:
                    hits += 1
                result_details.append(
                    {
                        "chunk_id": r.chunk_id,
                        "score": round(r.score, 4),
                        "hit": is_hit,
                    }
                )

            k = len(results)
            precision = hits / k if k > 0 else 0.0
            precision_scores.append(precision)

            per_query[query] = {
                "description": item["description"],
                "hits": hits,
                "retrieved": k,
                "precision_at_k": round(precision, 4),
                "results": result_details,
            }

        mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

        details = {
            "mean_precision_at_k": round(mean_precision, 4),
            "top_k": top_k,
            "alpha": alpha,
            "per_query": per_query,
        }

        return EvalResult(task_name=self.name, score=mean_precision, details=details)
