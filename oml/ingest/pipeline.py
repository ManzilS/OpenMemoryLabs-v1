from pathlib import Path
from typing import Optional
import gc
import io
import logging
import sys

# Rich progress bars use Unicode glyphs that fail on Windows charmap codec.
# Force UTF-8 on stdout/stderr if needed (safe no-op on Linux/macOS).
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Rich progress bars use Unicode glyphs that crash on Windows charmap.
# When running inside Streamlit (no real terminal), route Rich output to
# a StringIO sink so it never hits the broken codec.
def _safe_console() -> Console:
    """Return a Rich Console that won't crash on Windows charmap encoding."""
    if sys.platform != "win32":
        return Console(stderr=True)
    # On Windows, check if we're in a real terminal or piped/Streamlit
    try:
        if sys.stderr.isatty():
            return Console(stderr=True)
    except Exception:
        pass
    # Sink: discard output safely
    return Console(file=io.StringIO(), force_terminal=False)

_console = _safe_console()

logger = logging.getLogger(__name__)

from oml.config import DEFAULT_MODEL, DEFAULT_SUMMARIZER, DEFAULT_GRAPH_MODEL
from oml.storage.factory import get_storage
from oml.ingest.parsers import general_parse
from oml.ingest.chunkers import segment_document
from oml.ingest.summarizer import Summarizer
from oml.retrieval.bm25 import BM25Index
from oml.retrieval.vector import VectorIndex


class IngestionPipeline:
    """
    Orchestrates the ingestion process: Parse -> Summarize (Optional) -> Segment -> Store -> Index.
    Designed to be run programmatically, independently of the CLI.
    """
    
    def __init__(
        self,
        storage_type: str,
        data_dir: str = "data",
        artifacts_dir: str = "artifacts",
        device: str = "auto",
        storage_config: Optional[dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.device = device
        
        self.data_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        self.storage_type = storage_type
        self.storage = get_storage(self.storage_type, config=storage_config or {})
        self.storage.init_db()
        
    def _setup_pipeline_components(
        self,
        summarize: bool = False,
        summarizer_type: str = DEFAULT_SUMMARIZER,
        build_graph: bool = False,
        graph_model: str = DEFAULT_GRAPH_MODEL,
        model: str = DEFAULT_MODEL
    ):
        summarizer = None
        if summarize:
            if summarizer_type == "t5":
                from oml.ingest.t5_summarizer import T5Summarizer
                summarizer = T5Summarizer(device=self.device)
                logger.info(f"Summarization enabled (using T5-Small, device={self.device}).")
            else:
                summarizer = Summarizer(model_name=model)
                logger.info(f"Summarization enabled (using {model}).")

        graph_retriever = None
        provenance_index = None
        use_rebel = graph_model == "rebel"
        if build_graph:
            from oml.retrieval.graph_retriever import GraphRetriever
            from oml.retrieval.provenance_index import ProvenanceIndex
            graph_retriever = GraphRetriever(self.artifacts_dir)
            graph_retriever.load()
            provenance_index = ProvenanceIndex(self.artifacts_dir)
            if use_rebel:
                logger.info("Knowledge Graph extraction enabled (using REBEL-Large, local).")
            else:
                logger.info(f"Knowledge Graph extraction enabled (using {model}).")
                
        return summarizer, graph_retriever, provenance_index, use_rebel
        

    def run(
        self,
        path: Optional[str] = None,
        limit: Optional[int] = None,
        limit_chunks: Optional[int] = None,
        rebuild_indices: bool = True,
        only_index: bool = False,
        summarize: bool = False,
        summarizer_type: str = DEFAULT_SUMMARIZER,
        build_graph: bool = False,
        graph_model: str = DEFAULT_GRAPH_MODEL,
        model: str = DEFAULT_MODEL,
        demo: bool = False
    ):
        logger.info(f"Initialized {self.storage_type} storage.")
        
        summarizer, graph_retriever, provenance_index, use_rebel = self._setup_pipeline_components(
            summarize, summarizer_type, build_graph, graph_model, model
        )

        if not only_index:
            target_dir = None
            
            # 2. Source Selection
            if demo:
                logger.info("Using built-in demo dataset...")
                demo_dir = self.data_dir / "demo_dataset"
                demo_dir.mkdir(parents=True, exist_ok=True)
                (demo_dir / "cats.txt").write_text("Cats are wonderful pets. They sleep a lot and purr when happy.")
                (demo_dir / "dogs.txt").write_text("Dogs are loyal animals. They love to play fetch and go for walks.")
                (demo_dir / "finance.txt").write_text("The stock market is volatile today. Tech stocks are rising while energy falls.")
                target_dir = demo_dir
            elif path:
                target_dir = Path(path)
                if not target_dir.exists():
                    raise FileNotFoundError(f"Path '{path}' does not exist!")
                logger.info(f"Using provided source: {target_dir}")
            else:
                raise ValueError(
                    "No data source specified. Use --demo for built-in samples "
                    "or --path to supply your own files."
                )
        
            # 3. Processing Loop
            logger.info(f"Scanning files in {target_dir}...")
            
            if target_dir.is_file():
                all_files = [target_dir]
            else:
                all_files = [p for p in target_dir.rglob("*") if p.is_file()]
        
            if not all_files:
                raise FileNotFoundError(f"No files found in {target_dir}!")
                
            if limit:
                all_files = all_files[:limit]
                
            logger.info(f"Processing {len(all_files)} files...")
            
            batch_size = 100
            current_docs = []
            current_chunks = []
            
            with Progress(console=_console) as progress:
                file_task = progress.add_task("[green]Processing files...", total=len(all_files))
                
                for file_path in all_files:
                    try:
                        doc = general_parse(file_path)
                        
                        if summarizer:
                            doc.summary = summarizer.summarize_document(doc)
        
                        chunks = segment_document(doc)
                        if limit_chunks is not None:
                            chunks = chunks[:limit_chunks]
                        
                        if graph_retriever:
                            chunk_task = progress.add_task(f"[cyan]Extracting Graph Triples from {file_path.name}...", total=len(chunks))

                            if use_rebel:
                                from oml.ingest.rebel_extractor import extract_triples_rebel
                                _extract_fn = lambda chunk: (chunk, extract_triples_rebel(chunk.chunk_text, device=self.device))
                            else:
                                from oml.ingest.graph_extractor import extract_triples
                                _extract_fn = lambda chunk: (chunk, extract_triples(chunk.chunk_text, model))

                            from concurrent.futures import ThreadPoolExecutor, as_completed
                            with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as _pool:
                                futures = {_pool.submit(_extract_fn, chunk): chunk for chunk in chunks}
                                for future in as_completed(futures):
                                    try:
                                        chunk, triples = future.result()
                                    except Exception as _exc:
                                        chunk = futures[future]
                                        triples = []
                                        import logging as _logging
                                        _logging.getLogger(__name__).warning(
                                            "Triple extraction failed for chunk %s: %s", chunk.chunk_id, _exc
                                        )
                                    if triples:
                                        graph_retriever.add_triples(triples)
                                        if provenance_index:
                                            provenance_index.add_triples(chunk.chunk_id, triples)
                                    progress.advance(chunk_task)
                            progress.remove_task(chunk_task)
                        
                        current_docs.append(doc)
                        current_chunks.extend(chunks)
                        
                        if len(current_docs) >= batch_size:
                            self.storage.upsert_documents(current_docs)
                            self.storage.upsert_chunks(current_chunks)
                            current_docs = []
                            current_chunks = []
                            
                    except Exception as e:
                        logger.info(f"Failed to process {file_path}: {e}")
                    
                    progress.advance(file_task)
                
            if current_docs:
                self.storage.upsert_documents(current_docs)
                self.storage.upsert_chunks(current_chunks)
                
            if graph_retriever:
                logger.info("Saving Knowledge Graph...")
                graph_retriever.save()
            if provenance_index:
                logger.info("Saving Provenance Index...")
                provenance_index.save()
                stats = provenance_index.stats()
                logger.info(f"  Chunks indexed: {stats['chunks_indexed']}, Entities: {stats['unique_entities']}, Avg entities/chunk: {stats['avg_entities_per_chunk']:.1f}")
        
            logger.info("Ingestion complete.")
            self._print_saved_locations()
            
        if rebuild_indices or only_index:
            self._build_indices()
            
    def _build_indices(self):
        if self.storage_type == "lancedb":
            logger.info("LanceDB handles indexing automatically. Skipping manual index build.")
            return

        logger.info("Building Indices...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_console,
        ) as progress:
            build_task = progress.add_task("[green]Building Indices...", total=4)

            all_chunks = self.storage.get_all_chunks()
            chunk_ids = [c.chunk_id for c in all_chunks]
            texts = [c.chunk_text for c in all_chunks]
            progress.advance(build_task)

            if not chunk_ids:
                logger.info("No chunks found to index!")
                return

            bm25_path = self.artifacts_dir / "bm25.pkl"
            bm25 = BM25Index(bm25_path)
            bm25.build(chunk_ids, texts)
            bm25.save()
            progress.advance(build_task)

            # Release large BM25 structures before vector embedding on big corpora.
            del bm25
            gc.collect()

            index_task = progress.add_task(
                f"[cyan]Indexing {len(chunk_ids)} chunks...",
                total=len(chunk_ids),
            )
            last_processed = 0

            def _on_vector_progress(processed: int, total: int) -> None:
                nonlocal last_processed
                if processed <= last_processed:
                    return
                progress.update(index_task, total=total, completed=processed)
                last_processed = processed

            vector_path = self.artifacts_dir / "vector.index"
            map_path = self.artifacts_dir / "vector_map.json"
            vector = VectorIndex(vector_path, map_path, device=self.device)
            vector.build(chunk_ids, texts, progress_callback=_on_vector_progress)
            progress.update(index_task, completed=len(chunk_ids))
            progress.advance(build_task)

            vector.save()
            progress.advance(build_task)

        logger.info(f"Vector index saved to {vector_path}")
        logger.info("Indexing complete.")

    def _print_saved_locations(self) -> None:
        """Print where ingested data is persisted for the active storage backend."""
        locations: list[tuple[str, Path]] = []

        if self.storage_type == "sqlite":
            conn_str = getattr(self.storage, "conn_str", "")
            if isinstance(conn_str, str) and conn_str.startswith("sqlite:///"):
                db_path = Path(conn_str.replace("sqlite:///", "", 1)).resolve()
                locations.append(("SQLite DB", db_path))
        elif self.storage_type == "lancedb":
            persist_path = getattr(self.storage, "persist_path", "")
            if persist_path:
                locations.append(("LanceDB", Path(persist_path).resolve()))

        if not locations:
            if self.storage_type == "memory":
                logger.info("Saved outputs: none (memory storage is in-process only).")
            return

        logger.info("Saved outputs:")
        for label, path in locations:
            logger.info(f"  - {label}: {path}")
