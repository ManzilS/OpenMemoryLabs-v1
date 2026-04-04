import gc
import json
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print(f"Warning: Failed to import vector dependencies: {e}")
    # Define dummy class/functions if import fails so code doesn't crash immediately.
    SentenceTransformer = None
    faiss = None
    np = None

from oml.retrieval.base import BaseIndex
from oml.utils.device import resolve_device

logger = logging.getLogger(__name__)


class MockModel:
    def __init__(self, device=None):
        self.device = type("obj", (object,), {"type": "cpu"})

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        # Return random vectors of dimension 384 (all-MiniLM-L6-v2 size).
        return np.random.rand(len(texts), 384).astype("float32")


class MockIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = None

    def add(self, vectors):
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query_vec, top_k):
        # Brute force cosine similarity for mock.
        if self.vectors is None:
            return np.array([[]]), np.array([[]])

        # Normalize query if not already.
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores = np.dot(self.vectors, query_vec.T).flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        distances = scores[indices]
        return np.array([distances]), np.array([indices])


def write_mock_index(index, path):
    import pickle

    with open(path, "wb") as f:
        pickle.dump(index, f)


def read_mock_index(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


class VectorIndex(BaseIndex):
    """
    FAISS-based vector index helper.
    Automatically uses the GPU when available (device="auto").
    """

    def __init__(
        self,
        index_path: Path,
        mapping_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
    ):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.model_name = model_name
        # Resolve "auto" -> "cuda" or "cpu" once at construction time.
        self.device = resolve_device(device)
        self._model = None
        self.index = None
        # Maps internal FAISS row id -> chunk_id.
        self.chunk_ids = []

    @property
    def model(self):
        """Lazy load the model onto the resolved device."""
        if self._model is None:
            if SentenceTransformer is None:
                logger.warning("SentenceTransformer not available. Using MockModel.")
                global np
                if np is None:
                    import numpy as np
                self._model = MockModel()
            else:
                logger.info(
                    "Loading embedding model: %s on %s...",
                    self.model_name,
                    self.device,
                )
                try:
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                except Exception as e:
                    logger.error(
                        "Failed to load SentenceTransformer: %s. Using MockModel.",
                        e,
                    )
                    self._model = MockModel()
        return self._model

    def build(
        self,
        chunk_ids: List[str],
        texts: List[str],
        batch_size: int = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Embeds texts and incrementally builds the vector index to limit peak memory."""
        if not texts:
            return
        if len(chunk_ids) != len(texts):
            raise ValueError(
                f"chunk_ids/texts length mismatch: {len(chunk_ids)} vs {len(texts)}"
            )

        on_gpu = self.device == "cuda" or (
            hasattr(self.model, "device")
            and getattr(self.model.device, "type", "") == "cuda"
        )

        # Larger batches on GPU to saturate throughput.
        if batch_size is None:
            batch_size = 512 if on_gpu else 32

        total = len(texts)
        logger.info(
            "Embedding %s chunks (batch=%s, device=%s)...",
            total,
            batch_size,
            self.device,
        )
        use_text_progress = progress_callback is None
        if use_text_progress:
            print(
                f"Embedding {total} chunks (batch={batch_size}, device={self.device})..."
            )
        started_at = time.monotonic()
        progress_step = max(batch_size * 100, 10_000)
        next_progress = progress_step

        self.index = None
        kept_chunk_ids: List[str] = []

        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = chunk_ids[i : i + batch_size]

            batch_emb = self.model.encode(
                batch_texts, convert_to_numpy=True, show_progress_bar=False
            )
            batch_emb = np.asarray(batch_emb, dtype="float32")
            if batch_emb.ndim == 1:
                batch_emb = batch_emb.reshape(1, -1)

            if batch_emb.shape[0] != len(batch_ids):
                raise RuntimeError(
                    "Embedding model returned an unexpected batch size "
                    f"({batch_emb.shape[0]} != {len(batch_ids)})."
                )

            # Normalize for cosine similarity.
            if faiss:
                faiss.normalize_L2(batch_emb)
            else:
                norm = np.linalg.norm(batch_emb, axis=1, keepdims=True)
                batch_emb = batch_emb / (norm + 1e-10)

            if self.index is None:
                dimension = batch_emb.shape[1]
                if faiss:
                    # Keep a CPU index for stable persistence on all environments.
                    self.index = faiss.IndexFlatIP(dimension)
                else:
                    logger.warning("FAISS not available. Using MockIndex.")
                    self.index = MockIndex(dimension)

            self.index.add(batch_emb)
            kept_chunk_ids.extend(batch_ids)

            processed = min(i + len(batch_texts), total)
            if progress_callback is not None:
                progress_callback(processed, total)
            if use_text_progress and (processed == total or processed >= next_progress):
                elapsed = time.monotonic() - started_at
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = max(total - processed, 0)
                eta_s = remaining / rate if rate > 0 else 0.0
                pct = (processed / total) * 100.0 if total else 100.0
                print(
                    "Embedding progress: "
                    f"{processed}/{total} ({pct:.1f}%) | "
                    f"{rate:.0f} chunks/s | ETA {eta_s/60:.1f}m"
                )
                while processed >= next_progress:
                    next_progress += progress_step
                logger.info("Embedded %s/%s chunks", processed, total)

            del batch_emb
            if processed % (batch_size * 200) == 0:
                gc.collect()

        self.chunk_ids = kept_chunk_ids

    def save(self):
        """Saves FAISS index and ID mapping (GPU index is converted to CPU first)."""
        if self.index is None:
            raise ValueError("Index not built.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if faiss and isinstance(self.index, faiss.Index):
            # GPU indices cannot be written directly; convert to CPU index first.
            try:
                save_index = faiss.index_gpu_to_cpu(self.index)
            except Exception:
                save_index = self.index  # already CPU or conversion failed
            faiss.write_index(save_index, str(self.index_path))
        else:
            write_mock_index(self.index, str(self.index_path))

        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f)

        logger.info("Vector index saved to %s", self.index_path)

    def load(self) -> bool:
        """Loads index and mapping."""
        if not self.index_path.exists() or not self.mapping_path.exists():
            return False

        try:
            if faiss:
                try:
                    self.index = faiss.read_index(str(self.index_path))
                except Exception:
                    # Fallback if we saved a mock index but now have faiss (unlikely) or vice versa.
                    self.index = read_mock_index(str(self.index_path))
            else:
                self.index = read_mock_index(str(self.index_path))

            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.chunk_ids = json.load(f)
            return True
        except Exception as e:
            print(f"Failed to load Vector index: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns list of (chunk_id, score).
        Score is cosine similarity (-1 to 1).
        """
        if self.index is None:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        query_vec = np.asarray(query_vec, dtype="float32")

        if faiss:
            faiss.normalize_L2(query_vec)
            distances, indices = self.index.search(query_vec, top_k)
        else:
            distances, indices = self.index.search(query_vec[0], top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                results.append((chunk_id, float(score)))

        return results
