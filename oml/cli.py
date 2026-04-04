import typer
from typing import Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from oml.eval.run import run_task
# Import tasks to ensure they are registered
from oml import __version__
from oml.config import DEFAULT_MODEL, DEFAULT_STORAGE, DEFAULT_SUMMARIZER, DEFAULT_GRAPH_MODEL, DEFAULT_SQLITE_PATH

app = typer.Typer(
    name="oml",
    help="OpenMemoryLab CLI. Automatically reads defaults from 'oml.yaml' if present.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """OpenMemoryLab command-line interface.

    Common commands:
      - oml ingest             Ingest data and build indices
      - oml query              Run a one-off hybrid search over your data
      - oml chat               Start an interactive RAG chat session
      - oml consolidate        Build higher-level MemoryNotes from threads
      - oml teeg-ingest        Ingest text into TEEG evolving graph memory
      - oml teeg-query         Query TEEG memory with graph traversal
      - oml teeg-consolidate   Compress TEEG note clusters into summaries
      - oml prism-ingest       Ingest with write-time dedup (SketchGate)
      - oml prism-batch        Batch ingest with N-to-1 LLM call coalescing
      - oml prism-query        Query PRISM memory (Scout + TieredContextPacker)
      - oml prism-consolidate  Compress PRISM note clusters into summaries
      - oml prism-stats        Show PRISM efficiency statistics
      - oml eval               Run evaluation tasks/ablations

    Run 'oml <command> --help' for details on each command.
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())

@app.command()
def version():
    """Show package version"""
    print(__version__)

@app.command()
def eval(
    task: str = typer.Argument(..., help="Task to run: 'lost-in-middle', 'faithfulness', 'ablations', etc."),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="Model to use"),
    limit: int = typer.Option(10, help="Limit items (for ablations)"),
):
    """Run evaluation tasks"""
    try:
        # Pass limit to config
        config = {"limit": limit}
        
        result = run_task(task, model_name=model, config=config)
        typer.echo(f"Task '{task}' finished with score: {result.score:.2f}")
    except Exception as e:
        typer.secho(f"Error running eval: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable hot-reload (development only)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of uvicorn worker processes"),
):
    """
    Start the OpenMemoryLab REST API server.

    Serves /query, /chat, /teeg/ingest, /teeg/query and /health endpoints
    with auto-generated Swagger docs at http://<host>:<port>/docs.

    Examples:

      oml api                           # 0.0.0.0:8000

      oml api --port 9000 --reload      # dev mode with hot-reload

      oml api --host 127.0.0.1          # localhost only
    """
    try:
        import uvicorn
    except ImportError:
        typer.secho(
            "uvicorn is required to run the API server. "
            "Install it with: pip install uvicorn[standard]",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    typer.secho(
        f"Starting OpenMemoryLab API on http://{host}:{port} ...",
        fg=typer.colors.GREEN,
    )
    typer.secho(f"  Swagger UI -> http://{host}:{port}/docs", fg=typer.colors.CYAN)
    typer.secho(f"  ReDoc      -> http://{host}:{port}/redoc", fg=typer.colors.CYAN)

    uvicorn.run(
        "oml.api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
    )


@app.command()
def ui():
    """
    Launch the OpenMemoryLab Streamlit Web UI.
    """
    import subprocess
    from pathlib import Path
    import oml

    # Resolve the path to oml/app/ui.py
    oml_dir = Path(oml.__file__).parent
    ui_script = oml_dir / "app" / "ui.py"

    try:
        host = "127.0.0.1"
        port = 8501
        local_url = f"http://{host}:{port}"

        typer.secho("Starting OpenMemoryLab UI...", fg=typer.colors.GREEN)
        typer.secho(f"Local URL: {local_url}", fg=typer.colors.CYAN)

        subprocess.run(
            [
                "streamlit",
                "run",
                str(ui_script),
                "--server.address",
                host,
                "--server.port",
                str(port),
                "--server.headless",
                "true",
            ]
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        typer.secho(f"Failed to start UI: {e}", fg=typer.colors.RED)

@app.command()
def chat(
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="Model to use (mock, ollama:<name>, gemini:<name>)"),
    top_k: int = typer.Option(5, help="Number of retrieved chunks"),
    budget: int = typer.Option(4000, help="Max token budget"),
    show_prompt: bool = typer.Option(False, "--show", "-s", help="Show the prompt sent to the LLM"),
    show_tokens: bool = typer.Option(False, help="Show approximate token usage per turn"),
    storage_type: str = typer.Option(DEFAULT_STORAGE, help="Storage backend to query (matches ingest)"),
):
    """
    Start an interactive RAG chat session.
    """
    from oml.app.chat import chat_loop
    # Check enviroment/defaults logic could go here, but kept simple for now
    chat_loop(
        model=model,
        top_k=top_k,
        budget=budget,
        show_prompt=show_prompt,
        show_tokens=show_tokens,
        storage_type=storage_type,
    )


@app.command()
def consolidate(
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="LLM to use for summarization"),
    limit: int = typer.Option(10, help="Max threads to process"),
    db_path: str = typer.Option(DEFAULT_SQLITE_PATH, help="SQLite path for thread-memory consolidation"),
    artifacts_dir: str = typer.Option("artifacts", "--artifacts-dir", help="Directory for note vector index artifacts"),
):
    """
    [Phase 9] Consolidate threads into high-level MemoryNotes.
    """
    from oml.memory.consolidate import consolidate_threads
    
    consolidate_threads(
        db_path=db_path,
        model_name=model,
        limit=limit,
        artifacts_dir=artifacts_dir,
    )

@app.command()
def query(
    text: str = typer.Argument(..., help="Query text"),
    top_k: int = typer.Option(5, help="Number of results"),
    alpha: float = typer.Option(0.5, help="Hybrid weight (0.0=BM25 only, 1.0=Vector only)"),
    budget: int = typer.Option(None, help="Max token budget for context"),
    show_prompt: bool = typer.Option(False, help="Show the final packed prompt"),
    show_tokens: bool = typer.Option(False, help="Show approximate token usage for this query"),
    full: bool = typer.Option(False, "--full", "-f", help="Show full chunk text"),
    use_rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Use Cross-Encoder reranking"),
    use_hyde: bool = typer.Option(False, "--hyde", help="Use Hypothetical Document Embeddings for dense search"),
    use_graph: bool = typer.Option(False, "--graph", help="Use Knowledge Graph 1-hop traversal for context"),
    storage_type: str = typer.Option(DEFAULT_STORAGE, help="Storage backend to query (matches ingest)"),
):
    """Search the index (Hybrid BM25 + Vector)"""
    from oml.retrieval.pipeline import QueryPipeline
    import os
    
    pipeline = QueryPipeline(storage_type)
    model_name = os.getenv("OML_MODEL", DEFAULT_MODEL)
    
    context_chunks, prompt, approx_tokens = pipeline.run(
        query=text,
        top_k=top_k,
        alpha=alpha,
        budget=budget or 4000,
        use_rerank=use_rerank,
        use_hyde=use_hyde,
        use_graph=use_graph,
        model_name=model_name
    )
    
    # Printing logic remains in the CLI layer
    if not context_chunks:
        typer.secho("No results found.", fg=typer.colors.YELLOW)
        return
        
    typer.echo(f"Found {len([c for c in context_chunks if not c.chunk_id.startswith('summary_') and not c.chunk_id.startswith('note_')])} contextual results for query: '{text}'")

    for c in context_chunks:
        # We only print actual chunks in detailed view, summaries/notes are silently injected into prompt
        if c.chunk_id.startswith('summary_') or c.chunk_id == 'knowledge_graph_context':
             continue
        if c.text.startswith('[MEMORY NOTE'):
             continue
             
        typer.echo(f"[{c.chunk_id}] Score: {c.score:.4f}")
        if full:
            typer.echo(f"    Content:\n{c.text}")
            typer.echo("    " + "-"*40)
        else:
            typer.echo(f"    Sample: {c.text[:100]}...")

    if budget or show_prompt or show_tokens:
        if show_prompt:
            typer.secho("\n=== Packed Prompt ===", fg=typer.colors.GREEN)
            typer.echo(prompt)
            typer.secho("=====================\n", fg=typer.colors.GREEN)

        if show_tokens:
            typer.secho(
                f"[ContextBudget] Approx tokens used: {approx_tokens}/{budget or 4000}",
                fg=typer.colors.CYAN,
            )

@app.command()
def ingest(
    path: Optional[str] = typer.Argument(None, help="Path to the extracted dataset directory"),
    limit: int = typer.Option(None, help="Limit number of files to process (for testing)"),
    limit_chunks: int = typer.Option(None, help="Limit total number of chunks to ingest (for fast testing)"),
    rebuild_indices: bool = typer.Option(True, help="Build indices after ingestion"),
    only_index: bool = typer.Option(False, "--only-index", help="Skip ingestion and only rebuild indices"),
    device: str = typer.Option("auto", "--device", help="Device for local models: auto (default) | cpu | cuda"),
    storage_type: str = typer.Option(DEFAULT_STORAGE, help="Storage backend: sqlite, memory, etc. (Default from oml.yaml)"),
    summarize: bool = typer.Option(False, help="Enable summarization of documents"),
    summarizer_type: str = typer.Option(DEFAULT_SUMMARIZER, "--summarizer", help="Summarizer: 't5' (fast local T5-Small) or 'llm'"),
    build_graph: bool = typer.Option(False, "--graph", help="Extract entities and build knowledge graph"),
    graph_model: str = typer.Option(DEFAULT_GRAPH_MODEL, "--graph-model", help="Graph extractor: 'rebel' (fast local Babelscape/rebel-large) or 'llm'"),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="Model to use for LLM-based tasks (if enabled)"),
    demo: bool = typer.Option(False, "--demo", help="Ingest a tiny built-in dataset for instant testing"),
):
    """
    Ingest data: Parse -> Summarize (Optional) -> Segment -> Store -> Index.
    """
    from oml.ingest.pipeline import IngestionPipeline
    
    pipeline = IngestionPipeline(
        storage_type=storage_type,
        device=device
    )
    
    try:
        pipeline.run(
            path=path,
            limit=limit,
            limit_chunks=limit_chunks,
            rebuild_indices=rebuild_indices,
            only_index=only_index,
            summarize=summarize,
            summarizer_type=summarizer_type,
            build_graph=build_graph,
            graph_model=graph_model,
            model=model,
            demo=demo
        )
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)

@app.command("teeg-ingest")
def teeg_ingest(
    text: Optional[str] = typer.Argument(None, help="Raw text to ingest (omit to read from --file)"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Path to a text file to ingest"),
    context_hint: str = typer.Option("", "--context", "-c", help="Free-text context hint (source, chapter, date...)"),
    source_id: str = typer.Option("", "--source-id", help="Optional ID linking back to originating document"),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="LLM for note distillation + evolution"),
    artifacts_dir: str = typer.Option("teeg_store", "--dir", help="TEEG storage directory"),
    token_budget: int = typer.Option(3000, "--budget", help="Max tokens for TEEG context in queries"),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist store to disk after ingestion"),
    show_note: bool = typer.Option(False, "--show-note", help="Print the resulting AtomicNote in TOON format"),
    batch_file: Optional[str] = typer.Option(None, "--batch", help="Path to a file with one text per line (batch ingest)"),
):
    """
    [TEEG] Ingest raw text into the evolving graph memory system.

    An LLM distils the text into a structured AtomicNote, then MemoryEvolver
    checks for contradictions with existing memory and links the note into the
    relation graph.  Use --model mock for instant offline testing.

    Examples:

      oml teeg-ingest "Victor Frankenstein created the creature in 1797."

      oml teeg-ingest --file chapter5.txt --context "Frankenstein Ch.5" --model ollama:qwen3:4b

      oml teeg-ingest --batch facts.txt --save
    """
    from oml.memory.teeg_pipeline import TEEGPipeline
    from pathlib import Path as _Path

    pipeline = TEEGPipeline(
        artifacts_dir=artifacts_dir,
        model=model,
        token_budget=token_budget,
    )

    # ── batch mode ─────────────────────────────────────────────────────────
    if batch_file:
        batch_path = _Path(batch_file)
        if not batch_path.exists():
            typer.secho(f"Batch file not found: {batch_file}", fg=typer.colors.RED)
            raise typer.Exit(1)
        lines = [l.strip() for l in batch_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        typer.echo(f"[TEEG] Ingesting {len(lines)} texts from {batch_file}...")
        notes = pipeline.ingest_batch(lines, context_hint=context_hint)
        typer.secho(f"[TEEG] OK Ingested {len(notes)} notes", fg=typer.colors.GREEN)
        if show_note:
            for note in notes:
                typer.echo(note.to_toon())
                typer.echo("---")
        if save:
            pipeline.save()
        return

    # ── single note mode ───────────────────────────────────────────────────
    if file:
        file_path = _Path(file)
        if not file_path.exists():
            typer.secho(f"File not found: {file}", fg=typer.colors.RED)
            raise typer.Exit(1)
        raw = file_path.read_text(encoding="utf-8")
        source_id = source_id or file_path.name
    elif text:
        raw = text
    else:
        typer.secho("Provide text as argument, --file, or --batch.", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"[TEEG] Ingesting note via model '{model}'...")
    note = pipeline.ingest(raw, context_hint=context_hint, source_id=source_id)
    typer.secho(f"[TEEG] OK Note stored: {note.note_id}", fg=typer.colors.GREEN)

    if show_note:
        typer.echo("\n" + note.to_toon())

    stats = pipeline.stats()
    typer.echo(
        f"[TEEG] Store: {stats['active_notes']} active / "
        f"{stats['total_notes']} total notes, "
        f"{stats['graph_edges']} graph edges"
    )

    if save:
        pipeline.save()


@app.command("teeg-query")
def teeg_query(
    question: str = typer.Argument(..., help="Question to answer from TEEG memory"),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="LLM for answer generation"),
    artifacts_dir: str = typer.Option("teeg_store", "--dir", help="TEEG storage directory"),
    top_k: int = typer.Option(8, "--top-k", help="Max notes to retrieve"),
    max_hops: int = typer.Option(2, "--hops", help="Graph traversal depth"),
    token_budget: int = typer.Option(3000, "--budget", help="Max tokens for TEEG context"),
    show_context: bool = typer.Option(False, "--show-context", help="Print the TOON context block sent to the LLM"),
    explain: bool = typer.Option(False, "--explain", help="Show traversal explanation instead of generating an answer"),
    search_only: bool = typer.Option(False, "--search", help="Show matching notes without generating an answer"),
):
    """
    [TEEG] Query the evolving graph memory and get an LLM-generated answer.

    ScoutRetriever walks the relation graph from seed notes to build a
    TOON-encoded context block, which is then sent to the LLM for grounded
    answer generation.

    Examples:

      oml teeg-query "Who created the creature?"

      oml teeg-query "What happened at 1am?" --explain

      oml teeg-query "Frankenstein's motivation" --search --top-k 5

      oml teeg-query "Who built the monster?" --model ollama:qwen3:4b --show-context
    """
    from oml.memory.teeg_pipeline import TEEGPipeline

    pipeline = TEEGPipeline(
        artifacts_dir=artifacts_dir,
        model=model,
        token_budget=token_budget,
        scout_top_k=top_k,
        scout_max_hops=max_hops,
    )

    stats = pipeline.stats()
    if stats["active_notes"] == 0:
        typer.secho(
            "[TEEG] No notes found. Run 'oml teeg-ingest' first.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(0)

    # ── explain mode ───────────────────────────────────────────────────────
    if explain:
        typer.secho("[TEEG] Traversal explanation:", fg=typer.colors.CYAN)
        typer.echo(pipeline.explain_query(question, top_k=top_k))
        return

    # ── search-only mode ───────────────────────────────────────────────────
    if search_only:
        results = pipeline.search(question, top_k=top_k)
        if not results:
            typer.secho("[TEEG] No matching notes found.", fg=typer.colors.YELLOW)
            return
        typer.secho(f"[TEEG] Top {len(results)} notes for: {question!r}", fg=typer.colors.CYAN)
        for i, (note, score, hops) in enumerate(results, 1):
            label = "seed" if hops == 0 else f"hop-{hops}"
            typer.echo(f"\n  {i}. [{label}]  score={score:.3f}  id={note.note_id}")
            typer.echo(f"     {note.content[:120]}")
            if note.tags:
                typer.echo(f"     tags: {', '.join(note.tags)}")
        return

    # ── full query mode ────────────────────────────────────────────────────
    typer.echo(f"[TEEG] Querying with model '{model}'...")
    answer, context_str = pipeline.query(question, top_k=top_k, return_context=True)

    if show_context:
        typer.secho("\n=== TEEG MEMORY CONTEXT ===", fg=typer.colors.CYAN)
        typer.echo(context_str)
        typer.secho("===========================\n", fg=typer.colors.CYAN)

    typer.secho("\nAnswer:", fg=typer.colors.GREEN)
    typer.echo(answer)
    typer.echo(
        f"\n[TEEG] {stats['active_notes']} active notes  |  "
        f"{stats['graph_edges']} graph edges  |  "
        f"model: {model}"
    )


@app.command("teeg-consolidate")
def teeg_consolidate(
    artifacts_dir: str = typer.Option("teeg_store", "--dir", help="TEEG storage directory"),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="LLM for summary generation"),
    min_cluster: int = typer.Option(3, "--min-cluster", help="Minimum notes per cluster to consolidate"),
    max_clusters: int = typer.Option(10, "--max-clusters", help="Maximum clusters to consolidate per run"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show projected savings without modifying the store"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use heuristic summaries instead of LLM (faster, free)"),
):
    """
    [TEEG] Consolidate clusters of related notes into compressed summaries.

    Detects groups of notes that share keywords or are directly connected in
    the relation graph, then compresses each cluster into one summary note.
    Original notes are archived (not deleted) and remain reachable via the graph.

    Token savings: each archived note saves ~87 LLM context tokens per query.

    Examples:

      oml teeg-consolidate                         # consolidate with LLM summaries
      oml teeg-consolidate --dry-run               # show projected savings only
      oml teeg-consolidate --no-llm --min-cluster 2
    """
    from oml.storage.teeg_store import TEEGStore as _TEEGStore
    from oml.memory.consolidator import MemoryConsolidator as _Consolidator

    store = _TEEGStore(artifacts_dir=artifacts_dir)
    consolidator = _Consolidator(
        store,
        model_name=model,
        min_cluster_size=min_cluster,
        use_llm_summary=not no_llm,
    )

    stats_before = store.stats()
    typer.echo(
        f"[TEEG] Store: {stats_before['active_notes']} active / "
        f"{stats_before['total_notes']} total notes"
    )

    if dry_run:
        result = consolidator.dry_run()
        typer.secho("\n[TEEG] Dry-run projection:", fg=typer.colors.CYAN)
    else:
        result = consolidator.consolidate(max_clusters=max_clusters)
        typer.secho("\n[TEEG] Consolidation complete:", fg=typer.colors.GREEN)

    typer.echo(f"  Clusters found      : {result.clusters_found}")
    typer.echo(f"  Notes archived      : {result.notes_archived}")
    typer.echo(f"  Summary notes       : {result.summaries_created}")
    typer.echo(f"  Token savings (est) : ~{result.token_savings_est} per query")
    typer.echo(f"  Small clusters skip : {result.skipped_small_clusters}")

    if not dry_run:
        stats_after = store.stats()
        typer.echo(
            f"\n[TEEG] Store after: {stats_after['active_notes']} active / "
            f"{stats_after['total_notes']} total notes"
        )


@app.command("prism-ingest")
def prism_ingest(
    text: Optional[str] = typer.Argument(None, help="Text to ingest"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Read text from file"),
    context: str = typer.Option("", "--context", help="Optional context hint"),
    source_id: str = typer.Option("", "--source-id", help="Optional ID linking back to originating document"),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="LLM model"),
    artifacts_dir: str = typer.Option("teeg_store", "--dir", help="PRISM storage directory"),
    dedup_threshold: float = typer.Option(0.75, "--dedup", help="MinHash Jaccard dedup threshold"),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist store to disk after ingestion"),
    show_note: bool = typer.Option(False, "--show-note", help="Print the resulting AtomicNote in TOON format"),
    batch_file: Optional[str] = typer.Option(None, "--batch", help="Path to a file with one text per line (batch ingest)"),
):
    """
    [PRISM] Ingest text with write-time near-duplicate detection (SketchGate).

    Checks incoming text against all existing notes via MinHash LSH before
    making any LLM calls.  Near-duplicates are silently merged (access_count++)
    instead of creating redundant notes.

    Examples:

      oml prism-ingest "Victor Frankenstein created the creature."

      oml prism-ingest --file chapter1.txt --context "Frankenstein, Ch.5"

      oml prism-ingest "Repeated fact." --dedup 0.5

      oml prism-ingest --batch facts.txt --save
    """
    from oml.memory.prism_pipeline import PRISMPipeline
    from pathlib import Path as _Path

    pipeline = PRISMPipeline(
        artifacts_dir=artifacts_dir, model=model, dedup_threshold=dedup_threshold
    )

    # -- batch mode ---------------------------------------------------------
    if batch_file:
        batch_path = _Path(batch_file)
        if not batch_path.exists():
            typer.secho(f"Batch file not found: {batch_file}", fg=typer.colors.RED)
            raise typer.Exit(1)
        texts = [ln.strip() for ln in batch_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        typer.echo(f"[PRISM] Ingesting {len(texts)} texts from {batch_file}...")
        results = [pipeline.ingest(t, context_hint=context) for t in texts]
        created = sum(1 for r in results if not r.was_deduplicated)
        skipped = sum(1 for r in results if r.was_deduplicated)
        typer.secho(f"[PRISM] OK Ingested {created} notes, {skipped} dedup skips", fg=typer.colors.GREEN)
        if show_note:
            for r in results:
                if not r.was_deduplicated:
                    typer.echo(r.note.to_toon())
                    typer.echo("---")
        if save:
            pipeline.save()
        return

    # -- single note mode ---------------------------------------------------
    if file:
        file_path = _Path(file)
        if not file_path.exists():
            typer.secho(f"File not found: {file}", fg=typer.colors.RED)
            raise typer.Exit(1)
        raw = file_path.read_text(encoding="utf-8")
        source_id = source_id or file_path.name
    elif text:
        raw = text
    else:
        typer.secho("[PRISM] Provide text as argument, --file, or --batch.", fg=typer.colors.RED)
        raise typer.Exit(1)

    result = pipeline.ingest(raw, context_hint=context)

    if result.was_deduplicated:
        typer.secho(
            f"[PRISM] Near-duplicate detected -> merged into {result.merged_into}",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.secho(
            f"[PRISM] Stored note {result.note.note_id}",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"  content:  {result.note.content[:100]}")
        typer.echo(f"  keywords: {', '.join(result.note.keywords)}")
        if show_note:
            typer.echo("\n" + result.note.to_toon())

    s = pipeline._sketch.stats()
    typer.echo(
        f"\n[PRISM] Store: {pipeline._store.active_count()} active notes  |  "
        f"Dedup rate: {s['dedup_rate']:.1%}  |  "
        f"Skips total: {s['skips_total']}"
    )

    if save:
        pipeline.save()

@app.command("prism-batch")
def prism_batch(
    texts_file: str = typer.Argument(..., help="File with one text per line"),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="LLM model"),
    artifacts_dir: str = typer.Option("teeg_store", "--dir", help="PRISM storage directory"),
    batch_size: int = typer.Option(8, "--batch-size", help="Max texts per LLM call"),
    dedup_threshold: float = typer.Option(0.75, "--dedup", help="MinHash dedup threshold"),
):
    """
    [PRISM] Batch-ingest a file of texts with N-to-1 LLM call coalescing (CallBatcher).

    Reduces LLM API calls from 2N (naive) to 2 by packing all distillation
    requests into one structured LLM prompt and all evolution judgments into a
    second prompt.  Call efficiency = 1 - 1/N (87.5% for N=8).

    Examples:

      oml prism-batch texts.txt

      oml prism-batch texts.txt --batch-size 16 --model ollama:qwen3:4b
    """
    from oml.memory.prism_pipeline import PRISMPipeline

    path = typer.open_file(texts_file)
    texts = [line.strip() for line in path if line.strip()]
    if not texts:
        typer.secho("[PRISM] File is empty.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    typer.echo(f"[PRISM] Batch-ingesting {len(texts)} texts...")
    typer.echo(f"  Naive LLM calls: {2 * len(texts)}   |   PRISM target: 2")

    pipeline = PRISMPipeline(
        artifacts_dir=artifacts_dir,
        model=model,
        dedup_threshold=dedup_threshold,
        batch_size=batch_size,
    )
    result = pipeline.batch_ingest(texts)
    pipeline.save()

    typer.secho("\n[PRISM] Batch complete:", fg=typer.colors.GREEN)
    typer.echo(f"  Notes created:   {len(result.notes) - result.dedup_count}")
    typer.echo(f"  Dedup skips:     {result.dedup_count}")
    typer.echo(f"  Delta patches:   {result.delta_count}")
    typer.echo(f"  LLM calls made:  {result.llm_calls_made}")
    typer.echo(f"  LLM calls saved: {result.llm_calls_saved}")
    typer.echo(f"  Call efficiency: {result.call_efficiency:.1%}")


@app.command("prism-query")
def prism_query(
    question: str = typer.Argument(..., help="Question to answer from PRISM memory"),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="LLM model"),
    artifacts_dir: str = typer.Option("teeg_store", "--dir", help="PRISM storage directory"),
    top_k: int = typer.Option(8, "--top-k", help="Max notes to retrieve"),
    max_hops: int = typer.Option(2, "--hops", help="Graph traversal depth"),
    show_context: bool = typer.Option(False, "--show-context", help="Print TOON context block"),
    explain: bool = typer.Option(False, "--explain", help="Show traversal explanation instead of generating an answer"),
    search_only: bool = typer.Option(False, "--search", help="Show matching notes without generating an answer"),
):
    """
    [PRISM] Query memory using Scout graph traversal + TieredContextPacker.

    Identical retrieval quality to 'oml teeg-query' but runs through the full
    PRISM stack (SketchGate, DeltaStore reconstruction, importance scoring).

    Examples:

      oml prism-query "Who created the creature?"

      oml prism-query "What happened in the Arctic?" --show-context

      oml prism-query "What happened at 1am?" --explain

      oml prism-query "Frankenstein's motivation" --search --top-k 5
    """
    from oml.memory.prism_pipeline import PRISMPipeline

    pipeline = PRISMPipeline(artifacts_dir=artifacts_dir, model=model)

    if pipeline._store.active_count() == 0:
        typer.secho(
            "[PRISM] No notes found. Run 'oml prism-ingest' first.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(0)

    typer.echo(f"[PRISM] Querying {pipeline._store.active_count()} notes...")

    # -- explain mode -------------------------------------------------------
    if explain:
        typer.secho("[PRISM] Traversal explanation:", fg=typer.colors.CYAN)
        typer.echo(pipeline.explain_query(question, top_k=top_k))
        return

    # -- search-only mode ---------------------------------------------------
    if search_only:
        results = pipeline.search(question, top_k=top_k)
        if not results:
            typer.secho("[PRISM] No matching notes found.", fg=typer.colors.YELLOW)
            return
        typer.secho(f"[PRISM] Top {len(results)} notes for: {question!r}", fg=typer.colors.CYAN)
        for i, (note, score, hops) in enumerate(results, 1):
            label = "seed" if hops == 0 else f"hop-{hops}"
            typer.echo(f"\n  {i}. [{label}]  score={score:.3f}  id={note.note_id}")
            typer.echo(f"     {note.content[:120]}")
            if note.tags:
                typer.echo(f"     tags: {', '.join(note.tags)}")
        return

    # -- full query mode ----------------------------------------------------
    answer, context = pipeline.query(question, top_k=top_k)

    if show_context:
        typer.secho("\n=== PRISM MEMORY CONTEXT ===", fg=typer.colors.CYAN)
        typer.echo(context)
        typer.secho("============================\n", fg=typer.colors.CYAN)

    typer.secho("\nAnswer:", fg=typer.colors.GREEN)
    typer.echo(answer)


@app.command("prism-stats")
def prism_stats(
    artifacts_dir: str = typer.Option("teeg_store", "--dir", help="PRISM storage directory"),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="Model label used for stats context"),
):
    """
    [PRISM] Show aggregated efficiency statistics from all three PRISM layers.

    Displays:
      SketchGate — dedup rate, MinHash threshold, Bloom FP rate
      DeltaStore — patch count, token savings estimate
      CallBatcher — calls made/saved, average efficiency

    Example:

      oml prism-stats
    """
    from oml.memory.prism_pipeline import PRISMPipeline

    pipeline = PRISMPipeline(artifacts_dir=artifacts_dir, model=model)
    raw = pipeline.raw_stats()

    typer.secho("[PRISM] Statistics", fg=typer.colors.CYAN)
    typer.echo("\n  Store:")
    typer.echo(f"    Active notes:        {raw['store']['active_notes']}")
    typer.echo(f"    Total notes:         {raw['store']['total_notes']}")
    typer.echo(f"    Graph edges:         {raw['store']['graph_edges']}")
    typer.echo("\n  SketchGate (Layer 1 — write dedup):")
    typer.echo(f"    Registered notes:    {raw['sketch_gate']['registered_notes']}")
    typer.echo(f"    Dedup threshold:     {raw['sketch_gate']['dedup_threshold']}")
    typer.echo(f"    Checks total:        {raw['sketch_gate']['checks_total']}")
    typer.echo(f"    Skips total:         {raw['sketch_gate']['skips_total']}")
    typer.echo(f"    Dedup rate:          {raw['sketch_gate']['dedup_rate']:.1%}")
    typer.echo("\n  DeltaStore (Layer 2 — semantic patches):")
    typer.echo(f"    Bases with patches:  {raw['delta_store']['bases_with_patches']}")
    typer.echo(f"    Total patches:       {raw['delta_store']['total_patches']}")
    typer.echo(f"    Token savings est.:  ~{raw['delta_store']['token_savings_est']} tokens/query")
    typer.echo("\n  CallBatcher (Layer 3 — call coalescing):")
    typer.echo(f"    Calls made:          {raw['call_batcher']['calls_made']}")
    typer.echo(f"    Calls saved:         {raw['call_batcher']['calls_saved']}")
    typer.echo(f"    Call efficiency:     {raw['call_batcher']['call_efficiency']:.1%}")


@app.command("prism-consolidate")
def prism_consolidate(
    artifacts_dir: str = typer.Option("teeg_store", "--dir", help="PRISM storage directory"),
    model: str = typer.Option(DEFAULT_MODEL, envvar="OML_MODEL", help="LLM for summary generation"),
    min_cluster: int = typer.Option(3, "--min-cluster", help="Minimum notes per cluster to consolidate"),
    max_clusters: int = typer.Option(10, "--max-clusters", help="Maximum clusters to consolidate per run"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show projected savings without modifying the store"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use heuristic summaries instead of LLM (faster, free)"),
):
    """
    [PRISM] Consolidate clusters of related notes into compressed summaries.

    Same cluster-compression logic as 'oml teeg-consolidate' but prints
    PRISM-style efficiency stats alongside the consolidation results.

    Examples:

      oml prism-consolidate

      oml prism-consolidate --dry-run

      oml prism-consolidate --no-llm --min-cluster 2
    """
    from oml.storage.teeg_store import TEEGStore as _TEEGStore
    from oml.memory.consolidator import MemoryConsolidator as _Consolidator

    store = _TEEGStore(artifacts_dir=artifacts_dir)
    consolidator = _Consolidator(
        store,
        model_name=model,
        min_cluster_size=min_cluster,
        use_llm_summary=not no_llm,
    )

    stats_before = store.stats()
    typer.echo(
        f"[PRISM] Store: {stats_before['active_notes']} active / "
        f"{stats_before['total_notes']} total notes"
    )

    if dry_run:
        result = consolidator.dry_run()
        typer.secho("\n[PRISM] Dry-run projection:", fg=typer.colors.CYAN)
    else:
        result = consolidator.consolidate(max_clusters=max_clusters)
        typer.secho("\n[PRISM] Consolidation complete:", fg=typer.colors.GREEN)

    typer.echo(f"  Clusters found      : {result.clusters_found}")
    typer.echo(f"  Notes archived      : {result.notes_archived}")
    typer.echo(f"  Summary notes       : {result.summaries_created}")
    typer.echo(f"  Token savings (est) : ~{result.token_savings_est} per query")
    typer.echo(f"  Small clusters skip : {result.skipped_small_clusters}")

    if not dry_run:
        stats_after = store.stats()
        typer.echo(
            f"\n[PRISM] Store after: {stats_after['active_notes']} active / "
            f"{stats_after['total_notes']} total notes"
        )


@app.command("cache-stats")
def cache_stats_cmd(
    artifacts_dir: str = typer.Option("artifacts", "--dir", help="Cache directory (where llm_cache.json lives)"),
) -> None:
    """Show LLM response cache statistics (hit rate, entries, estimated savings)."""
    from oml.llm.cache import LLMCache
    import pathlib

    cache_file = pathlib.Path(artifacts_dir) / "llm_cache.json"
    if not cache_file.exists():
        typer.secho(f"[Cache] No cache file found at {cache_file}", fg=typer.colors.YELLOW)
        typer.echo("  Run any experiment with OML_CACHE_MODE=auto to start caching.")
        raise typer.Exit()

    cache = LLMCache(cache_path=artifacts_dir, mode="off")  # mode=off -> load only, no reads/writes
    stats = cache.stats()

    typer.secho("[LLM Cache] Statistics", fg=typer.colors.CYAN)
    typer.echo(f"  Cache file:             {stats['cache_file']}")
    typer.echo(f"  Total entries:          {stats['total_entries']}")
    typer.echo(f"  Cache hits (session):   {stats['cache_hits']}")
    typer.echo(f"  Cache misses (session): {stats['cache_misses']}")
    typer.echo(f"  Hit rate (session):     {stats['hit_rate']:.1%}")
    typer.echo(f"  Est. calls saved:       {stats['estimated_calls_saved']}")
    typer.echo(f"  Mode:                   {stats['mode']}")

    # Show per-model breakdown
    models: dict = {}
    for entry in cache._entries.values():
        models[entry.model] = models.get(entry.model, 0) + 1
    if models:
        typer.echo("\n  Entries by model:")
        for m, count in sorted(models.items(), key=lambda x: -x[1]):
            typer.echo(f"    {m:<35} {count} entries")


@app.command("cache-clear")
def cache_clear_cmd(
    artifacts_dir: str = typer.Option("artifacts", "--dir", help="Cache directory"),
    model: str = typer.Option("", "--model", help="Clear only entries for this model (e.g. openai:gpt-4o-mini). Clears ALL if omitted."),
    confirm: bool = typer.Option(False, "--confirm", help="Required safety flag to actually clear the cache."),
) -> None:
    """Clear cached LLM responses. Requires --confirm to prevent accidental deletion."""
    if not confirm:
        typer.secho(
            "Safety: pass --confirm to clear the cache. "
            "Example: oml cache-clear --confirm",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(1)

    from oml.llm.cache import LLMCache
    import pathlib

    cache_file = pathlib.Path(artifacts_dir) / "llm_cache.json"
    if not cache_file.exists():
        typer.secho(f"[Cache] No cache file at {cache_file} — nothing to clear.", fg=typer.colors.GREEN)
        raise typer.Exit()

    cache = LLMCache(cache_path=artifacts_dir, mode="off")
    removed = cache.clear(model=model if model else None)
    cache.save()

    label = f"model '{model}'" if model else "all models"
    typer.secho(
        f"[Cache] Cleared {removed} entr{'y' if removed == 1 else 'ies'} for {label}.",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
