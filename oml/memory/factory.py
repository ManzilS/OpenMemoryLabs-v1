from __future__ import annotations

from typing import Literal

from oml.config import DEFAULT_MODEL
from oml.memory.protocol import MemoryPipeline


def get_memory_pipeline(
    pipeline_type: Literal["teeg", "prism"],
    *,
    artifacts_dir: str = "teeg_store",
    model: str = DEFAULT_MODEL,
    token_budget: int = 3000,
    scout_top_k: int = 8,
    scout_max_hops: int = 2,
    dedup_threshold: float = 0.75,
    batch_size: int = 8,
) -> MemoryPipeline:
    """Factory for interchangeable memory pipelines used in experiments."""
    if pipeline_type == "teeg":
        from oml.memory.teeg_pipeline import TEEGPipeline

        return TEEGPipeline(
            artifacts_dir=artifacts_dir,
            model=model,
            token_budget=token_budget,
            scout_top_k=scout_top_k,
            scout_max_hops=scout_max_hops,
        )

    if pipeline_type == "prism":
        from oml.memory.prism_pipeline import PRISMPipeline

        return PRISMPipeline(
            artifacts_dir=artifacts_dir,
            model=model,
            token_budget=token_budget,
            scout_top_k=scout_top_k,
            scout_max_hops=scout_max_hops,
            dedup_threshold=dedup_threshold,
            batch_size=batch_size,
        )

    raise ValueError(f"Unknown memory pipeline type: {pipeline_type}")
