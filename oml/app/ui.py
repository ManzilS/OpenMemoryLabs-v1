"""
oml/app/ui.py  OpenMemoryLab Experiment Lab
=============================================
Multi-tab Streamlit UI.  Run with:  oml ui  (or  streamlit run oml/app/ui.py)

Tabs
----
   Experiment Lab   define N retrieval configs, run all against one query, compare side-by-side
   Chat             RAG chat with full pipeline controls (HyDE / Graph / GTCC / Rerank)
   Ingest           ingest data with all pipeline options (+ drag-and-drop file upload)
   TEEG             TEEG ingest + query UI
   Reports          browse saved Markdown reports from reports/
  Explorer          browse, search, and delete individual chunks and documents

v1.0.0 additions
-----------------
  * Drag-and-drop file uploader in the Ingest tab (PDF, TXT, MD, CSV)
  * Dynamic API key inputs in the sidebar (OpenAI / Gemini, session-only, never written to disk)
  * Database Explorer tab  paginated chunk/document table with keyword search and delete
  * Multi-user isolation  each browser session gets its own UUID-scoped SQLite path by default
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import streamlit as st

#  Page config (must be the very first Streamlit call) 
st.set_page_config(
    page_title="OpenMemoryLab",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _safe_sqlite_counts(db_path: str) -> dict[str, int]:
    counts = {"documents": 0, "chunks": 0, "memory_notes": 0}
    if not Path(db_path).exists():
        return counts
    try:
        conn = sqlite3.connect(db_path)
        try:
            counts["documents"] = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        except Exception:
            pass
        try:
            counts["chunks"] = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        except Exception:
            pass
        try:
            counts["memory_notes"] = int(conn.execute("SELECT COUNT(*) FROM memory_notes").fetchone()[0])
        except Exception:
            pass
        conn.close()
    except Exception:
        pass
    return counts


def _artifact_status(artifacts_dir: str) -> dict[str, bool]:
    root = Path(artifacts_dir)
    return {
        "bm25": (root / "bm25.pkl").exists(),
        "vector_index": (root / "vector.index").exists(),
        "vector_map": (root / "vector_map.json").exists(),
        "notes_index": (root / "notes_vector.index").exists(),
        "notes_map": (root / "notes_vector_map.json").exists(),
    }


def _safe_cache_stats(artifacts_dir: str) -> tuple[dict | None, dict[str, int]]:
    try:
        from oml.llm.cache import LLMCache

        cache = LLMCache(cache_path=artifacts_dir, mode="off")
        stats = cache.stats()
        by_model: dict[str, int] = {}
        for entry in cache._entries.values():
            by_model[entry.model] = by_model.get(entry.model, 0) + 1
        return stats, by_model
    except Exception:
        return None, {}


def _safe_teeg_store_stats(store_dir: str) -> dict[str, int]:
    try:
        from oml.storage.teeg_store import TEEGStore

        stats = TEEGStore(artifacts_dir=store_dir).stats()
        return {
            "active_notes": int(stats.get("active_notes", 0)),
            "total_notes": int(stats.get("total_notes", 0)),
            "graph_edges": int(stats.get("graph_edges", 0)),
        }
    except Exception:
        return {"active_notes": 0, "total_notes": 0, "graph_edges": 0}


def _available_eval_tasks() -> list[str]:
    try:
        import oml.eval.ablations  # noqa: F401
        import oml.eval.tasks  # noqa: F401
        from oml.eval.run import _TASK_REGISTRY

        return sorted(_TASK_REGISTRY.keys())
    except Exception:
        return [
            "faithfulness",
            "lost-in-middle",
            "retrieval_precision",
            "cost_latency",
            "oml_vs_rag",
            "global_trends",
            "ablations",
        ]


def _run_cluster_consolidation(
    store_dir: str,
    model_name: str,
    min_cluster: int,
    max_clusters: int,
    dry_run: bool,
    no_llm: bool,
):
    from oml.memory.consolidator import MemoryConsolidator
    from oml.storage.teeg_store import TEEGStore

    store = TEEGStore(artifacts_dir=store_dir)
    stats_before = store.stats()
    consolidator = MemoryConsolidator(
        store,
        model_name=model_name,
        min_cluster_size=min_cluster,
        use_llm_summary=not no_llm,
    )
    result = consolidator.dry_run() if dry_run else consolidator.consolidate(max_clusters=max_clusters)
    stats_after = store.stats()
    return result, stats_before, stats_after


def _inject_theme_css(theme_mode: str) -> None:
    dark = theme_mode.lower() == "dark"
    if dark:
        palette = {
            "bg": "#0a111f",
            "ink": "#dfe8ff",
            "muted": "#9fb2d7",
            "panel": "rgba(12, 23, 44, 0.72)",
            "border": "rgba(132, 160, 219, 0.24)",
            "sidebar_a": "#0f1a31",
            "sidebar_b": "#0a1428",
            "accent": "#48d1b2",
            "accent_soft": "#8c7bff",
        }
    else:
        palette = {
            "bg": "#f4f7fb",
            "ink": "#16233a",
            "muted": "#476086",
            "panel": "rgba(255, 255, 255, 0.84)",
            "border": "rgba(21, 45, 76, 0.18)",
            "sidebar_a": "#f7faff",
            "sidebar_b": "#edf3ff",
            "accent": "#0ea5a4",
            "accent_soft": "#6d6bf3",
        }

    st.markdown(
        f"""
        <style>
        :root {{
          --oml-bg: {palette["bg"]};
          --oml-ink: {palette["ink"]};
          --oml-muted: {palette["muted"]};
          --oml-panel: {palette["panel"]};
          --oml-border: {palette["border"]};
          --oml-accent: {palette["accent"]};
          --oml-accent-soft: {palette["accent_soft"]};
        }}
        .stApp {{
          background:
            radial-gradient(1100px 520px at 95% -15%, color-mix(in srgb, var(--oml-accent-soft) 28%, transparent) 0%, transparent 60%),
            radial-gradient(980px 460px at -10% -10%, color-mix(in srgb, var(--oml-accent) 20%, transparent) 0%, transparent 58%),
            linear-gradient(180deg, var(--oml-bg) 0%, color-mix(in srgb, var(--oml-bg) 75%, black) 100%);
          color: var(--oml-ink);
          font-family: "Avenir Next", "Segoe UI", sans-serif;
        }}
        h1, h2, h3 {{
          font-family: "Trebuchet MS", "Avenir Next", "Segoe UI", sans-serif;
          letter-spacing: -0.01em;
        }}
        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp li,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMarkdownContainer"] * {{
          color: var(--oml-ink);
        }}
        .stApp a {{
          color: color-mix(in srgb, var(--oml-accent) 72%, white);
        }}
        .stApp input,
        .stApp textarea,
        .stApp select,
        .stApp [data-baseweb="input"] input,
        .stApp [data-baseweb="select"] *,
        .stApp [data-baseweb="textarea"] textarea {{
          color: var(--oml-ink) !important;
        }}
        /* Form element backgrounds — critical for light mode readability */
        .stApp [data-baseweb="input"],
        .stApp [data-baseweb="base-input"],
        .stApp [data-baseweb="textarea"],
        .stApp [data-baseweb="select"] > div:first-child {{
          background-color: {"#ffffff" if not dark else "rgba(8, 16, 34, 0.55)"} !important;
          border-color: var(--oml-border) !important;
        }}
        .stApp input,
        .stApp textarea {{
          background-color: transparent !important;
        }}
        /* Sidebar widget backgrounds */
        section[data-testid="stSidebar"] [data-baseweb="base-input"],
        section[data-testid="stSidebar"] [data-baseweb="input"],
        section[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {{
          background-color: {"#ffffff" if not dark else "rgba(8, 16, 34, 0.55)"} !important;
        }}
        /* Selectbox dropdown and value containers */
        .stApp [data-baseweb="popover"],
        .stApp [data-baseweb="menu"],
        .stApp [role="listbox"] {{
          background-color: {"#ffffff" if not dark else "#0d1a2f"} !important;
          color: var(--oml-ink) !important;
        }}
        .stApp [data-baseweb="menu"] li,
        .stApp [role="option"] {{
          color: var(--oml-ink) !important;
        }}
        .stApp [data-baseweb="menu"] li:hover,
        .stApp [role="option"]:hover {{
          background-color: {"#e8f0fe" if not dark else "rgba(72, 209, 178, 0.12)"} !important;
        }}
        /* Number input buttons */
        .stApp [data-testid="stNumberInput"] button {{
          color: var(--oml-ink) !important;
          background-color: {"#f0f4f9" if not dark else "rgba(15, 25, 48, 0.6)"} !important;
        }}
        /* JSON viewer in light mode */
        .stApp .react-json-view,
        .stApp .pretty-json-container {{
          background-color: {"#f8f9fb !important" if not dark else "inherit"};
        }}
        .stApp .react-json-view .object-key {{
          color: {"#1a1a2e !important" if not dark else "inherit"};
        }}
        .stApp .react-json-view .string-value {{
          color: {"#c2185b !important" if not dark else "inherit"};
        }}
        .stApp .react-json-view .variable-row {{
          border-left-color: {"#d0d7e3 !important" if not dark else "inherit"};
        }}
        /* Expander backgrounds */
        .stApp [data-testid="stExpander"] {{
          background-color: {"rgba(255,255,255,0.7)" if not dark else "rgba(12, 23, 44, 0.45)"};
          border: 1px solid var(--oml-border);
          border-radius: 12px;
        }}
        /* Code blocks */
        .stApp pre, .stApp code {{
          background-color: {"#f0f3f8" if not dark else "#0b1425"} !important;
          color: {"#1a1a2e" if not dark else "#dfe8ff"} !important;
        }}
        .stApp input::placeholder,
        .stApp textarea::placeholder {{
          color: color-mix(in srgb, var(--oml-muted) 82%, transparent) !important;
          opacity: 1;
        }}
        section[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, {palette["sidebar_a"]} 0%, {palette["sidebar_b"]} 100%);
          border-right: 1px solid var(--oml-border);
        }}
        section[data-testid="stSidebar"] [data-testid="stTextInput"],
        section[data-testid="stSidebar"] [data-testid="stTextArea"],
        section[data-testid="stSidebar"] [data-testid="stNumberInput"],
        section[data-testid="stSidebar"] [data-testid="stSelectbox"],
        section[data-testid="stSidebar"] [data-testid="stMultiSelect"] {{
          margin-bottom: 0.55rem;
        }}
        section[data-testid="stSidebar"] [data-testid="InputInstructions"] {{
          margin-top: 0.32rem !important;
          line-height: 1.28 !important;
          opacity: 0.92;
        }}
        section[data-testid="stSidebar"] button[title*="password"],
        section[data-testid="stSidebar"] button[aria-label*="password"] {{
          margin-top: 0.12rem !important;
        }}
        section[data-testid="stSidebar"] input[type="text"],
        section[data-testid="stSidebar"] input[type="password"] {{
          padding-right: 2.35rem !important;
        }}
        div[data-testid="stMetric"] {{
          background: var(--oml-panel);
          border: 1px solid var(--oml-border);
          border-radius: 14px;
          box-shadow: 0 10px 26px rgba(0, 0, 0, 0.16);
          padding: 0.45rem 0.65rem;
        }}
        .stMetric label {{ font-size: 0.8rem !important; color: var(--oml-muted) !important; }}
        div[data-testid="stHorizontalBlock"] > div {{ min-width: 0; }}
        .stButton button, .stDownloadButton button {{
          border-radius: 12px;
          border: 1px solid var(--oml-border);
          transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }}
        .stButton button:hover, .stDownloadButton button:hover {{
          transform: translateY(-1px);
          box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
          border-color: color-mix(in srgb, var(--oml-accent) 45%, white);
        }}
        [data-testid="stChatMessage"] {{
          border: 1px solid var(--oml-border);
          border-radius: 14px;
          background: color-mix(in srgb, var(--oml-panel) 90%, transparent);
          padding: 0.25rem 0.4rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
          gap: 0.35rem;
          overflow-x: auto;
          overflow-y: hidden;
          flex-wrap: nowrap;
          scrollbar-width: thin;
          padding-bottom: 0.2rem;
        }}
        .stTabs [data-baseweb="tab"] {{
          border-radius: 999px;
          border: 1px solid var(--oml-border);
          background: color-mix(in srgb, var(--oml-panel) 90%, transparent);
          white-space: nowrap;
          flex: 0 0 auto;
          min-height: 1.95rem;
          font-size: 0.84rem;
          padding: 0.22rem 0.7rem;
        }}
        .stTabs [aria-selected="true"] {{
          border-color: color-mix(in srgb, var(--oml-accent) 55%, white);
          box-shadow: 0 0 0 1px color-mix(in srgb, var(--oml-accent) 35%, transparent) inset;
        }}
        .oml-hero {{
          padding: 1.0rem 1.15rem;
          border-radius: 16px;
          border: 1px solid var(--oml-border);
          background:
            linear-gradient(125deg, color-mix(in srgb, var(--oml-panel) 88%, transparent), color-mix(in srgb, var(--oml-accent-soft) 12%, transparent));
          box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
          margin-bottom: 0.8rem;
          animation: oml-fade-up 280ms ease-out;
        }}
        .oml-hero h2 {{
          margin: 0 0 0.28rem 0;
          font-size: 1.22rem;
          font-weight: 700;
        }}
        .oml-hero p {{
          margin: 0;
          opacity: 0.92;
        }}
        .oml-chip {{
          display: inline-block;
          margin-top: 0.5rem;
          padding: 0.18rem 0.55rem;
          border-radius: 999px;
          border: 1px solid var(--oml-border);
          background: color-mix(in srgb, var(--oml-accent) 18%, transparent);
          color: var(--oml-ink);
          font-size: 0.75rem;
          font-weight: 600;
        }}
        div[data-testid="stDataFrame"] {{
          border-radius: 14px;
          border: 1px solid var(--oml-border);
          overflow: hidden;
        }}
        div[data-testid="stProgressBar"] > div > div {{
          background: linear-gradient(90deg, var(--oml-accent), var(--oml-accent-soft));
        }}
        @keyframes oml-fade-up {{
          from {{ opacity: 0; transform: translateY(6px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero_banner(title: str, subtitle: str, chip: str = "") -> None:
    chip_html = f"<span class='oml-chip'>{chip}</span>" if chip else ""
    st.markdown(
        f"""
        <div class="oml-hero">
          <h2>{title}</h2>
          <p>{subtitle}</p>
          {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _mount_toolbar_theme_button(current_theme: str) -> None:
    """Render a sun/moon theme toggle as a native Streamlit button.

    Uses st.query_params for reliable theme switching across reruns.
    """
    next_theme = "Light" if current_theme == "Dark" else "Dark"
    is_dark = current_theme == "Dark"
    icon = "\u2600\uFE0F" if is_dark else "\U0001F319"

    if st.button(icon, key="theme_toggle", help=f"Switch to {next_theme} mode"):
        st.query_params["oml_theme"] = next_theme
        st.rerun()


#  Preset configurations (mirrors COMBINATIONS_REPORT.md) 
PRESETS = [
    {"name": "BM25 Only",            "bm25": True,  "vector": False, "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False, "alpha": 0.0, "top_k": 5},
    {"name": "Vector Only",          "bm25": False, "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False, "alpha": 1.0, "top_k": 5},
    {"name": "Hybrid (Baseline)",    "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False, "alpha": 0.5, "top_k": 5},
    {"name": "Hybrid + Rerank",      "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": True,  "summary": False, "gtcc": False, "alpha": 0.5, "top_k": 5},
    {"name": "Hybrid + HyDE",        "bm25": True,  "vector": True,  "hyde": True,  "graph": False, "rerank": False, "summary": False, "gtcc": False, "alpha": 0.5, "top_k": 5},
    {"name": "Hybrid + Graph",       "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": False, "gtcc": False, "alpha": 0.5, "top_k": 5},
    {"name": "HyDE + Rerank",        "bm25": True,  "vector": True,  "hyde": True,  "graph": False, "rerank": True,  "summary": False, "gtcc": False, "alpha": 0.5, "top_k": 5},
    {"name": "HyDE + Graph",         "bm25": True,  "vector": True,  "hyde": True,  "graph": True,  "rerank": False, "summary": False, "gtcc": False, "alpha": 0.5, "top_k": 5},
    {"name": "Graph + Rerank",       "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": True,  "summary": False, "gtcc": False, "alpha": 0.5, "top_k": 5},
    {"name": "Everything",           "bm25": True,  "vector": True,  "hyde": True,  "graph": True,  "rerank": True,  "summary": False, "gtcc": False, "alpha": 0.5, "top_k": 5},
    {"name": "Hybrid + T5 Summary",  "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": True,  "gtcc": False, "alpha": 0.5, "top_k": 5},
    {"name": "Hybrid + GTCC",        "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": True,  "alpha": 0.5, "top_k": 5},
    {"name": "Everything + T5",      "bm25": True,  "vector": True,  "hyde": True,  "graph": True,  "rerank": True,  "summary": True,  "gtcc": False, "teeg": False, "alpha": 0.5, "top_k": 5},
    {"name": "Full Stack",           "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": True,  "gtcc": True,  "teeg": False, "alpha": 0.5, "top_k": 5},
    #  TEEG presets 
    {"name": "TEEG Memory",          "bm25": False, "vector": False, "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False, "teeg": True,  "prism": False, "alpha": 0.5, "top_k": 8},
    {"name": "Hybrid + TEEG",        "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False, "teeg": True,  "prism": False, "alpha": 0.5, "top_k": 5},
    {"name": "TEEG + Graph",         "bm25": False, "vector": False, "hyde": False, "graph": True,  "rerank": False, "summary": False, "gtcc": False, "teeg": True,  "prism": False, "alpha": 0.5, "top_k": 8},
    {"name": "Full Stack + TEEG",    "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": True,  "gtcc": True,  "teeg": True,  "prism": False, "alpha": 0.5, "top_k": 5},
    #  PRISM presets 
    {"name": "PRISM Memory",         "bm25": False, "vector": False, "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False, "teeg": False, "prism": True,  "alpha": 0.5, "top_k": 8},
    {"name": "Hybrid + PRISM",       "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False, "teeg": False, "prism": True,  "alpha": 0.5, "top_k": 5},
    {"name": "TEEG vs PRISM",        "bm25": False, "vector": False, "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False, "teeg": True,  "prism": True,  "alpha": 0.5, "top_k": 8},
    {"name": "Full Stack + PRISM",   "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": True,  "gtcc": True,  "teeg": False, "prism": True,  "alpha": 0.5, "top_k": 5},
]
PRESET_NAMES = [p["name"] for p in PRESETS]


# 
# Pipeline runner helper (used by Experiment Lab & Chat)
# 

def _run_pipeline(
    query: str,
    cfg: dict,
    model_str: str,
    artifacts_dir: str,
    db_path: str,
    budget: int = 4000,
) -> dict:
    """
    Run one retrieval configuration against a query.
    Returns: {answer, context, prompt, latency, chunk_count, approx_tokens}
    Raises on hard failure so callers can catch and show an error.
    """
    from oml.retrieval.hybrid import HybridRetriever
    from oml.memory.context import ContextBudgeter, ContextChunk
    from oml.llm import get_llm_client

    artifacts_path = Path(artifacts_dir)
    start = time.time()

    #  HyDE: generate hypothetical document for denser vector search 
    vector_query = None
    if cfg.get("hyde"):
        from oml.retrieval.hyde import generate_hypothetical_document
        vector_query = generate_hypothetical_document(query, model_str)

    #  Hybrid retrieval 
    retriever = HybridRetriever(artifacts_path)
    top_k = int(cfg.get("top_k", 5))
    candidate_k = top_k * 5 if cfg.get("rerank") else top_k

    results = retriever.search(
        query,
        top_k=candidate_k,
        use_bm25=cfg.get("bm25", True),
        use_vector=cfg.get("vector", True),
        vector_query=vector_query,
        alpha=float(cfg.get("alpha", 0.5)),
    )

    #  Fetch chunk texts from SQLite 
    chunk_map: dict[str, str] = {}
    if results:
        try:
            from oml.storage.sqlite import get_chunks_by_ids
            rows = get_chunks_by_ids(db_path, [r.chunk_id for r in results])
            chunk_map = {c.chunk_id: c.chunk_text for c in rows}
        except Exception:
            pass

    #  Rerank 
    if cfg.get("rerank") and results:
        try:
            from oml.retrieval.rerank import Reranker
            reranker = Reranker()
            doc_texts, valid = [], []
            for r in results:
                txt = chunk_map.get(r.chunk_id)
                if txt:
                    doc_texts.append(txt)
                    valid.append(r)
            if valid:
                results = reranker.rerank(query, doc_texts, valid)[:top_k]
            else:
                results = results[:top_k]
        except Exception:
            results = results[:top_k]
    else:
        results = results[:top_k]

    #  Build base context chunks 
    context_chunks: list[ContextChunk] = []
    for res in results:
        text = chunk_map.get(res.chunk_id, "")
        if text:
            context_chunks.append(ContextChunk(chunk_id=res.chunk_id, text=text, score=res.score))

    #  GTCC: bridge chunk expansion 
    if cfg.get("gtcc"):
        try:
            from oml.retrieval.gtcc import GTCCRetriever
            from oml.storage.sqlite import get_chunks_by_ids
            gtcc = GTCCRetriever(artifacts_path)
            expanded = gtcc.expand_results([r.chunk_id for r in results], max_bridges=3)
            bridge_ids = [cid for cid, _, src in expanded if src == "bridge"]
            if bridge_ids:
                bridge_rows = get_chunks_by_ids(db_path, bridge_ids)
                bridge_map = {c.chunk_id: c.chunk_text for c in bridge_rows}
                for cid, score, src in expanded:
                    if src == "bridge" and cid in bridge_map:
                        context_chunks.append(
                            ContextChunk(chunk_id=cid, text=f"[BRIDGE]\n{bridge_map[cid]}", score=score)
                        )
        except Exception:
            pass

    #  Graph: knowledge graph context injection 
    if cfg.get("graph"):
        try:
            from oml.retrieval.graph_retriever import GraphRetriever
            gc = GraphRetriever(artifacts_path).search_graph(query, model_str)
            if gc:
                context_chunks.append(ContextChunk(chunk_id="kg_context", text=gc, score=1.0))
        except Exception:
            pass

    #  T5 Summary: inject document summary 
    if cfg.get("summary"):
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT summary FROM documents WHERE summary IS NOT NULL AND summary != '' LIMIT 1"
            ).fetchone()
            conn.close()
            if row:
                context_chunks.append(
                    ContextChunk(chunk_id="doc_summary", text=f"[DOC SUMMARY]\n{row[0]}", score=0.9)
                )
        except Exception:
            pass

    #  TEEG: inject evolving graph memory context 
    if cfg.get("teeg"):
        try:
            from oml.memory.teeg_pipeline import TEEGPipeline
            from oml.memory.context import ContextChunk
            teeg_store_dir = str(Path(artifacts_dir).parent / "teeg_store")
            teeg_pipeline = TEEGPipeline(
                artifacts_dir=teeg_store_dir,
                model=model_str,
                token_budget=budget,
                scout_top_k=int(cfg.get("top_k", 8)),
            )
            stats = teeg_pipeline.stats()
            if stats["active_notes"] > 0:
                _, teeg_context = teeg_pipeline.query(query, top_k=int(cfg.get("top_k", 8)), return_context=True)
                if teeg_context:
                    context_chunks.append(
                        ContextChunk(chunk_id="teeg_memory", text=teeg_context, score=1.0)
                    )
        except Exception:
            pass

    #  PRISM: inject efficient memory context (dedup + delta + batching) 
    if cfg.get("prism"):
        try:
            from oml.memory.prism_pipeline import PRISMPipeline
            from oml.memory.context import ContextChunk
            prism_store_dir = str(Path(artifacts_dir).parent / "prism_store")
            prism_pipeline = PRISMPipeline(
                artifacts_dir=prism_store_dir,
                model=model_str,
                token_budget=budget,
                scout_top_k=int(cfg.get("top_k", 8)),
            )
            raw = prism_pipeline.raw_stats()
            if raw["store"].get("active_notes", 0) > 0:
                _, prism_context = prism_pipeline.query(query, top_k=int(cfg.get("top_k", 8)), return_context=True)
                if prism_context:
                    context_chunks.append(
                        ContextChunk(chunk_id="prism_memory", text=prism_context, score=1.0)
                    )
        except Exception:
            pass

    #  Generate answer 
    llm = get_llm_client(model_str)
    budgeter = ContextBudgeter()
    prompt, approx_tokens = budgeter.construct_prompt_with_tokens(query, context_chunks, max_tokens=budget)
    answer = llm.generate(prompt)

    context_text = "\n\n---\n\n".join(
        f"[{c.chunk_id}] score={c.score:.3f}\n{c.text}" for c in context_chunks
    )
    return {
        "answer": answer,
        "context": context_text,
        "prompt": prompt,
        "latency": time.time() - start,
        "chunk_count": len(context_chunks),
        "approx_tokens": approx_tokens or 0,
    }


# 
# Multi-user session isolation  one UUID per browser connection
# 

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "api_server_proc" not in st.session_state:
    st.session_state.api_server_proc = None
if "api_server_cmd" not in st.session_state:
    st.session_state.api_server_cmd = ""
if "api_server_host" not in st.session_state:
    st.session_state.api_server_host = "127.0.0.1"
if "api_server_port" not in st.session_state:
    st.session_state.api_server_port = 8000
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"
_qp_theme = st.query_params.get("oml_theme")
if isinstance(_qp_theme, list):
    _qp_theme = _qp_theme[0] if _qp_theme else None
if _qp_theme in {"Dark", "Light"}:
    st.session_state.theme_mode = _qp_theme


# 
# Sidebar  Global settings shared across all tabs
# 

with st.sidebar:
    st.title("OpenMemoryLab")
    st.divider()
    st.header("Global Settings")

    try:
        from oml.config import DEFAULT_MODEL, DEFAULT_STORAGE, DEFAULT_SQLITE_PATH
    except Exception:
        DEFAULT_MODEL = "mock"
        DEFAULT_STORAGE = "sqlite"
        DEFAULT_SQLITE_PATH = "data/oml.db"

    g_model = st.text_input(
        "LLM Model",
        value=os.getenv("OML_MODEL", DEFAULT_MODEL),
        help="mock | ollama:<name> | openai:<name> | gemini:<name>",
    )
    g_storage = st.selectbox(
        "Storage Backend",
        options=["sqlite", "lancedb", "memory"],
        index=["sqlite", "lancedb", "memory"].index(os.getenv("OML_STORAGE", DEFAULT_STORAGE)),
    )
    g_artifacts = st.text_input("Artifacts Dir", value="artifacts")

    # Per-session default DB path for multi-user isolation
    _session_default_db = f"data/session_{st.session_state.session_id}/oml.db"
    g_db_path = st.text_input(
        "SQLite DB Path",
        value=_session_default_db,
        help="Each browser session gets its own isolated path by default. "
             "Override to share data across sessions.",
    )
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    _ui_host = os.getenv("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    _ui_port = os.getenv("STREAMLIT_SERVER_PORT", "8501")
    st.caption(f"Local UI: `http://{_ui_host}:{_ui_port}`")

    #  API Key Management (session-only, never written to disk) 
    st.divider()
    st.subheader("API Keys (Session-Only)")
    st.caption("Keys apply to this browser session only and are never saved to disk.")

    _openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        placeholder="sk-",
        key="sidebar_openai_key",
        help="Overrides the OPENAI_API_KEY environment variable for this session.",
    )
    _gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        placeholder="AIza",
        key="sidebar_gemini_key",
        help="Overrides the GEMINI_API_KEY environment variable for this session.",
    )
    if _openai_key:
        os.environ["OPENAI_API_KEY"] = _openai_key
    if _gemini_key:
        os.environ["GEMINI_API_KEY"] = _gemini_key

    if _openai_key or _gemini_key:
        n_set = sum([bool(_openai_key), bool(_gemini_key)])
        st.success(f"{n_set} key(s) active for this session.")

    #  Hardware status
    st.divider()
    try:
        from oml.utils.device import get_device_info
        _hw = get_device_info()
        if _hw["cuda_available"]:
            st.success(f"GPU: {_hw['gpu_name']} ({_hw['vram_gb']} GB)")
        elif _hw["gpu_visible"]:
            st.warning(
                f"GPU detected: {_hw['gpu_name']} ({_hw['vram_gb']} GB) "
                "— but PyTorch was installed without CUDA support. "
                "Reinstall with: pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124"
            )
        else:
            st.info("CPU only (no GPU detected)")
    except Exception:
        pass

    st.caption("v1.0.0 - OpenMemoryLab")

_inject_theme_css(st.session_state.theme_mode)

# Theme toggle — native Streamlit button, pinned top-right via CSS on its key class
_mount_toolbar_theme_button(st.session_state.theme_mode)

st.markdown(
    """
    <style>
    /* Pin the theme toggle (st-key-theme_toggle) to the top-right corner,
       offset to avoid overlapping the Streamlit Deploy / kebab buttons */
    .st-key-theme_toggle {
        position: fixed !important;
        top: 7px;
        right: 90px;
        z-index: 999999;
        width: auto !important;
    }
    .st-key-theme_toggle button {
        font-size: 1.4rem !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px 8px !important;
        cursor: pointer !important;
        opacity: 0.85;
        transition: opacity 0.15s, transform 0.15s;
        min-height: 0 !important;
        line-height: 1 !important;
    }
    .st-key-theme_toggle button:hover {
        opacity: 1 !important;
        transform: scale(1.15);
        background: transparent !important;
        border: none !important;
    }
    .st-key-theme_toggle [data-testid="stTooltipIcon"] {
        display: contents;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


#
# Main tabs
#

tab_home, tab_ingest, tab_chat, tab_lab, tab_techniques, tab_teeg, tab_prism, tab_ops, tab_reports, tab_bench, tab_explorer = st.tabs([
    "Home",
    "Ingest",
    "Chat",
    "Lab",
    "Techniques",
    "TEEG",
    "PRISM",
    "Ops",
    "Reports",
    "Benchmarks",
    "Explorer",
])


# ===============================================================
# TAB 0 - HOME
# ===============================================================

with tab_home:
    _hero_banner(
        "OpenMemoryLab Control Center",
        "One place to ingest data, run retrieval/chat, manage memory layers, and execute evaluations.",
        chip="System overview",
    )

    db_exists = Path(g_db_path).exists()
    counts = _safe_sqlite_counts(g_db_path)
    artifacts = _artifact_status(g_artifacts)
    memory_dir = str(Path(g_artifacts).parent / "teeg_store")
    teeg_stats = _safe_teeg_store_stats(memory_dir)
    cache_stats, cache_by_model = _safe_cache_stats(g_artifacts)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Documents", f"{counts['documents']:,}")
    m2.metric("Chunks", f"{counts['chunks']:,}")
    m3.metric("Thread Notes", f"{counts['memory_notes']:,}")
    m4.metric("TEEG Active", f"{teeg_stats['active_notes']:,}")
    m5.metric("LLM Cache", f"{(cache_stats or {}).get('total_entries', 0):,}")

    st.divider()
    st.subheader("Workflow")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        with st.container(border=True):
            st.markdown("**1. Ingest**")
            st.caption("Load files, build BM25 + vector indices, and prepare graph/summary artifacts.")
    with w2:
        with st.container(border=True):
            st.markdown("**2. Query + Chat**")
            st.caption("Ask questions with Hybrid, HyDE, Graph, GTCC, rerank, and context budget controls.")
    with w3:
        with st.container(border=True):
            st.markdown("**3. Memory Layers**")
            st.caption("Use TEEG/PRISM to ingest evolving notes, query graph memory, and consolidate clusters.")
    with w4:
        with st.container(border=True):
            st.markdown("**4. Evaluate + Report**")
            st.caption("Run eval tasks, compare benchmarks, and inspect generated reports.")

    st.divider()
    st.subheader("System Status")
    status_rows = [
        {"Component": "SQLite DB", "Status": "Ready" if db_exists else "Missing", "Location": g_db_path},
        {"Component": "BM25 Index", "Status": "Ready" if artifacts["bm25"] else "Missing", "Location": str(Path(g_artifacts) / "bm25.pkl")},
        {"Component": "Vector Index", "Status": "Ready" if artifacts["vector_index"] else "Missing", "Location": str(Path(g_artifacts) / "vector.index")},
        {"Component": "Vector Map", "Status": "Ready" if artifacts["vector_map"] else "Missing", "Location": str(Path(g_artifacts) / "vector_map.json")},
        {"Component": "Notes Index", "Status": "Ready" if artifacts["notes_index"] else "Missing", "Location": str(Path(g_artifacts) / "notes_vector.index")},
        {"Component": "TEEG Store", "Status": "Ready" if teeg_stats["total_notes"] > 0 else "Empty", "Location": memory_dir},
    ]
    st.table(status_rows)

    if cache_stats:
        with st.expander("Cache breakdown"):
            st.caption(f"Cache file: `{cache_stats['cache_file']}`")
            if cache_by_model:
                st.table([{"Model": k, "Entries": v} for k, v in sorted(cache_by_model.items(), key=lambda x: -x[1])])
            else:
                st.caption("No cached responses yet.")

    st.info(
        "Use the `Ops` tab for consolidation, cache management, single-task evaluation, and API controls."
    )


# 
# TAB 1  EXPERIMENT LAB
# 

with tab_lab:
    _hero_banner(
        "Experiment Lab",
        "Define up to 4 retrieval configurations, run them against the same query, and compare "
        "answers, latency, and chunk counts side-by-side.",
        chip="A/B retrieval testing",
    )

    #  Session state init 
    if "lab_configs" not in st.session_state:
        st.session_state.lab_configs = [
            {"id": 0, **{k: v for k, v in PRESETS[2].items()}},  # Hybrid baseline
            {"id": 1, **{k: v for k, v in PRESETS[3].items()}},  # Hybrid + Rerank
        ]
        st.session_state.lab_next_id = 2

    if "lab_results" not in st.session_state:
        st.session_state.lab_results = {}

    #  Query row 
    q_col, budget_col = st.columns([3, 1])
    with q_col:
        lab_query = st.text_input(
            "Query",
            placeholder="e.g.  Who is Robert Walton and why is he travelling to the North Pole?",
            key="lab_query_input",
        )
    with budget_col:
        lab_budget = st.number_input("Token Budget", min_value=500, max_value=32000, value=4000, key="lab_budget")

    #  Add / clear row 
    add_c, preset_c, _, clear_c = st.columns([1, 2, 2, 1])
    with add_c:
        add_btn = st.button("Add Config", key="lab_add")
    with preset_c:
        preset_sel = st.selectbox(
            "From preset",
            options=["(blank)"] + PRESET_NAMES,
            label_visibility="collapsed",
            key="lab_preset_sel",
        )
    with clear_c:
        if st.button("Clear All", key="lab_clear"):
            st.session_state.lab_configs = []
            st.session_state.lab_results = {}
            st.rerun()

    if add_btn:
        if preset_sel == "(blank)":
            new_cfg = {
                "id": st.session_state.lab_next_id,
                "name": f"Config {st.session_state.lab_next_id + 1}",
                "bm25": True, "vector": True, "hyde": False, "graph": False,
                "rerank": False, "summary": False, "gtcc": False,
                "alpha": 0.5, "top_k": 5,
            }
        else:
            base = next(p for p in PRESETS if p["name"] == preset_sel)
            new_cfg = {"id": st.session_state.lab_next_id, **{k: v for k, v in base.items()}}
        st.session_state.lab_configs.append(new_cfg)
        st.session_state.lab_next_id += 1
        st.rerun()

    #  Config cards 
    configs = st.session_state.lab_configs
    MAX_COLS = 4

    if not configs:
        st.info("No configurations yet. Click **Add Config** above to start.")
    else:
        visible = configs[:MAX_COLS]
        card_cols = st.columns(len(visible))
        to_remove_ids = []

        for slot, cfg in enumerate(visible):
            cid = cfg["id"]
            with card_cols[slot]:
                with st.container(border=True):
                    # Config name
                    cfg["name"] = st.text_input(
                        "Name", value=cfg["name"], key=f"name_{cid}",
                        label_visibility="collapsed",
                        placeholder="Config name",
                    )

                    # Preset quick-load
                    quick = st.selectbox(
                        "Load preset",
                        [""] + PRESET_NAMES,
                        key=f"quick_{cid}",
                        label_visibility="collapsed",
                    )
                    if quick != "":
                        base = next(p for p in PRESETS if p["name"] == quick)
                        for k, v in base.items():
                            cfg[k] = v
                        st.rerun()

                    st.caption("**Retrieval methods**")
                    left, right = st.columns(2)
                    with left:
                        cfg["bm25"]    = st.checkbox("BM25",    value=cfg.get("bm25",    True),  key=f"bm25_{cid}",    help="Keyword search via BM25")
                        cfg["vector"]  = st.checkbox("Vector",  value=cfg.get("vector",  True),  key=f"vec_{cid}",     help="Dense vector (FAISS) search")
                        cfg["hyde"]    = st.checkbox("HyDE",    value=cfg.get("hyde",    False), key=f"hyde_{cid}",    help="Hypothetical Document Embeddings")
                        cfg["graph"]   = st.checkbox("Graph",   value=cfg.get("graph",   False), key=f"graph_{cid}",   help="Knowledge graph context injection")
                    with right:
                        cfg["rerank"]  = st.checkbox("Rerank",     value=cfg.get("rerank",  False), key=f"rerank_{cid}",  help="Cross-encoder reranking")
                        cfg["summary"] = st.checkbox("T5 Summary", value=cfg.get("summary", False), key=f"summ_{cid}",    help="Inject document summary")
                        cfg["gtcc"]    = st.checkbox("GTCC",       value=cfg.get("gtcc",    False), key=f"gtcc_{cid}",    help="Bridge-chunk context expansion")
                        cfg["teeg"]    = st.checkbox("TEEG",       value=cfg.get("teeg",    False), key=f"teeg_{cid}",    help="Inject TEEG evolving graph memory context")
                        cfg["prism"]   = st.checkbox("PRISM",      value=cfg.get("prism",   False), key=f"prism_{cid}",   help="Inject PRISM memory (MinHash dedup + delta encoding + call batching)")

                    cfg["alpha"] = st.slider(
                        "Alpha BM25 <-> Vector",
                        min_value=0.0, max_value=1.0,
                        value=float(cfg.get("alpha", 0.5)), step=0.05,
                        key=f"alpha_{cid}",
                        help="0 = BM25 only, 1 = Vector only",
                    )
                    cfg["top_k"] = st.number_input(
                        "Top-K chunks", min_value=1, max_value=20,
                        value=int(cfg.get("top_k", 5)), key=f"topk_{cid}",
                    )

                    # Active flag summary
                    active = [k.upper() for k in ("bm25","vector","hyde","graph","rerank","summary","gtcc","teeg","prism") if cfg.get(k)]
                    badge = " | ".join(active) if active else "none"
                    st.caption(f"Active: {badge}")

                    # Delete button
                    if st.button("Remove", key=f"del_{cid}", width="stretch"):
                        to_remove_ids.append(cid)

                    # Quick result badge
                    r = st.session_state.lab_results.get(cfg["name"])
                    if r:
                        if "error" in r:
                            st.error(str(r["error"])[:55])
                        else:
                            st.success(
                                f"{r['chunk_count']} chunks | {r['latency']:.2f}s | ~{r['approx_tokens']} tok"
                            )

        if to_remove_ids:
            st.session_state.lab_configs = [c for c in configs if c["id"] not in to_remove_ids]
            st.rerun()

        if len(configs) > MAX_COLS:
            st.caption(f" Showing first {MAX_COLS} of {len(configs)} configs (add/remove to manage).")

    st.divider()

    #  Run button 
    run_disabled = not (lab_query or "").strip() or not st.session_state.lab_configs
    if st.button(
        "Run All Configurations",
        type="primary",
        disabled=run_disabled,
        key="lab_run",
        width="content",
    ):
        st.session_state.lab_results = {}
        n = len(st.session_state.lab_configs)
        prog = st.progress(0, text="Starting")
        status_box = st.empty()

        for i, cfg in enumerate(st.session_state.lab_configs):
            prog.progress(i / n, text=f"Running [{i+1}/{n}]: {cfg['name']}")
            status_box.info(f"Running {cfg['name']}...")
            try:
                result = _run_pipeline(
                    query=lab_query.strip(),
                    cfg=cfg,
                    model_str=g_model,
                    artifacts_dir=g_artifacts,
                    db_path=g_db_path,
                    budget=int(lab_budget),
                )
                st.session_state.lab_results[cfg["name"]] = result
            except Exception as exc:
                st.session_state.lab_results[cfg["name"]] = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }

        prog.progress(1.0, text="All done!")
        status_box.empty()
        st.rerun()

    #  Results display 
    results = st.session_state.lab_results
    if results:
        st.subheader("Results")

        # Summary comparison table
        table_rows = []
        for cfg_name, r in results.items():
            if "error" in r:
                table_rows.append({
                    "Configuration": cfg_name,
                    "Status": " Error",
                    "Chunks": "",
                    "Latency (s)": "",
                    "~Tokens": "",
                    "Answer preview": f"ERROR: {r['error'][:80]}",
                })
            else:
                table_rows.append({
                    "Configuration": cfg_name,
                    "Status": " OK",
                    "Chunks": r["chunk_count"],
                    "Latency (s)": f"{r['latency']:.2f}",
                    "~Tokens": r["approx_tokens"],
                    "Answer preview": (r["answer"] or "").replace("\n", " ")[:100],
                })
        st.table(table_rows)

        # Per-config detail columns
        n_det = min(len(results), 3)
        det_cols = st.columns(n_det)
        for j, (cfg_name, r) in enumerate(results.items()):
            with det_cols[j % n_det]:
                with st.expander(f"  {cfg_name}", expanded=True):
                    if "error" in r:
                        st.error(r["error"])
                        with st.expander("Traceback"):
                            st.code(r.get("traceback", ""), language="python")
                    else:
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Chunks", r["chunk_count"])
                        mc2.metric("Latency", f"{r['latency']:.2f}s")
                        mc3.metric("~Tokens", r["approx_tokens"])

                        st.markdown("**Answer**")
                        st.write(r["answer"] or "*(empty)*")

                        with st.expander("Retrieved context"):
                            st.text_area(
                                "ctx",
                                value=(r.get("context") or "")[:4000],
                                height=220,
                                label_visibility="collapsed",
                                disabled=True,
                                key=f"ctx_{cfg_name}",
                            )
                        with st.expander("Packed prompt"):
                            st.code((r.get("prompt") or "")[:3000], language="markdown")


# 
# TAB 2  CHAT
# 

with tab_chat:
    _hero_banner(
        "RAG Chat",
        "Conversational session with full retrieval pipeline controls.",
        chip="Interactive QA",
    )

    #  Session state 
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    #  Chat pipeline config (sidebar-style column) 
    cfg_col, chat_col = st.columns([1, 3])

    with cfg_col:
        with st.container(border=True):
            st.subheader("Pipeline Config")
            chat_top_k   = st.slider("Top-K", 1, 20, 5, key="chat_top_k")
            chat_alpha   = st.slider("Alpha BM25<->Vec", 0.0, 1.0, 0.5, 0.05, key="chat_alpha")
            chat_budget  = st.number_input("Token Budget", 500, 32000, 4000, key="chat_budget")
            st.caption("**Augmentation**")
            chat_rerank  = st.checkbox("Rerank",     value=True,  key="chat_rerank")
            chat_hyde    = st.checkbox("HyDE",       value=False, key="chat_hyde")
            chat_graph   = st.checkbox("Graph",      value=False, key="chat_graph")
            chat_gtcc    = st.checkbox("GTCC",       value=False, key="chat_gtcc")
            chat_summary = st.checkbox("T5 Summary", value=False, key="chat_summary")
            chat_bm25    = st.checkbox("BM25",       value=True,  key="chat_bm25")
            chat_vector  = st.checkbox("Vector",     value=True,  key="chat_vector")

            st.divider()
            show_ctx    = st.checkbox("Show context", value=False, key="chat_show_ctx")
            show_prompt = st.checkbox("Show prompt",  value=False, key="chat_show_prompt")

            if st.button("Reset Conversation", key="chat_reset", width="stretch"):
                st.session_state.chat_messages = []
                st.rerun()

    with chat_col:
        # Display history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("meta"):
                    meta = msg["meta"]
                    with st.expander("Diagnostics"):
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Chunks", meta.get("chunk_count", ""))
                        mc2.metric("Latency", f"{meta.get('latency', 0):.2f}s")
                        mc3.metric("~Tokens", meta.get("approx_tokens", ""))
                        if show_ctx:
                            st.text_area("Context", value=(meta.get("context") or "")[:3000], height=160, disabled=True, key=f"chat_ctx_{len(st.session_state.chat_messages)}")
                        if show_prompt:
                            st.code((meta.get("prompt") or "")[:2000], language="markdown")

        # Chat input
        if user_input := st.chat_input("Ask about your ingested data", key="chat_input"):
            st.session_state.chat_messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                chat_progress = st.progress(0, text="Preparing query...")
                with st.spinner("Retrieving and generating"):
                    chat_cfg = {
                        "bm25": chat_bm25, "vector": chat_vector,
                        "hyde": chat_hyde, "graph": chat_graph,
                        "rerank": chat_rerank, "summary": chat_summary,
                        "gtcc": chat_gtcc,
                        "alpha": chat_alpha, "top_k": chat_top_k,
                    }
                    try:
                        chat_progress.progress(20, text="Retrieving contextual chunks...")
                        result = _run_pipeline(
                            query=user_input,
                            cfg=chat_cfg,
                            model_str=g_model,
                            artifacts_dir=g_artifacts,
                            db_path=g_db_path,
                            budget=int(chat_budget),
                        )
                        chat_progress.progress(85, text="Generating final answer...")
                        answer = result["answer"]
                        st.markdown(answer)

                        with st.expander("Diagnostics"):
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Chunks", result["chunk_count"])
                            mc2.metric("Latency", f"{result['latency']:.2f}s")
                            mc3.metric("~Tokens", result["approx_tokens"])
                            if show_ctx:
                                st.text_area("ctx", value=(result.get("context") or "")[:3000], height=160, disabled=True, key="chat_ctx_new")
                            if show_prompt:
                                st.code((result.get("prompt") or "")[:2000], language="markdown")

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": answer,
                            "meta": result,
                        })
                        chat_progress.progress(100, text="Done.")
                        time.sleep(0.12)
                        chat_progress.empty()

                    except Exception as exc:
                        err_msg = f" Pipeline error: {exc}"
                        chat_progress.empty()
                        st.error(err_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": err_msg,
                        })


# 
# TAB 3  INGEST
# 

with tab_ingest:
    _hero_banner(
        "Ingest Data",
        "Parse -> Summarize (optional) -> Segment -> Store -> Index.",
        chip="Data pipeline",
    )

    left_col, right_col = st.columns([2, 1])

    with left_col:
        with st.form("ingest_form"):
            st.subheader("Source")
            ingest_demo = st.checkbox("Use built-in demo dataset (tiny, instant)", value=False, key="ingest_demo")
            ingest_path = st.text_input(
                "Data path",
                placeholder="/path/to/your/data or leave blank to use uploaded files",
                key="ingest_path",
                disabled=False,
            )

            #  Drag-and-drop file uploader 
            st.caption("or drag & drop files directly")
            ingest_uploaded = st.file_uploader(
                "Upload files (PDF, TXT, MD, CSV)",
                type=["pdf", "txt", "md", "csv"],
                accept_multiple_files=True,
                key="ingest_uploaded",
                help="Files are saved to a temporary directory and ingested. "
                     "Ignored when a Data path is set or Demo mode is enabled.",
            )

            st.subheader("Limits (for fast testing)")
            c1, c2 = st.columns(2)
            with c1:
                ingest_limit = st.number_input("Max files", min_value=0, value=0, help="0 = no limit", key="ing_limit")
            with c2:
                ingest_limit_chunks = st.number_input("Max chunks", min_value=0, value=0, help="0 = no limit", key="ing_limit_chunks")

            st.subheader("Options")
            o1, o2 = st.columns(2)
            with o1:
                ingest_summarize = st.checkbox("Summarise documents", value=False, key="ing_summ")
                ingest_graph = st.checkbox("Build knowledge graph", value=False, key="ing_graph")
                ingest_rebuild = st.checkbox("Rebuild indices after ingest", value=True, key="ing_rebuild")
                ingest_only_index = st.checkbox("Only rebuild indices (skip ingest)", value=False, key="ing_only_idx")
            with o2:
                ingest_summarizer = st.selectbox("Summariser", ["t5", "llm"], key="ing_summarizer")
                ingest_graph_model = st.selectbox("Graph model", ["rebel", "llm"], key="ing_graph_model")
                ingest_device = st.selectbox(
                    "Embedding device",
                    ["auto", "cpu", "cuda"],
                    index=0,
                    key="ing_device",
                    help="auto = use GPU if available, else CPU",
                )
                ingest_storage = st.selectbox("Storage backend", ["sqlite", "lancedb", "memory"], key="ing_storage")

            submitted = st.form_submit_button("Start Ingest", type="primary", width="stretch")

    with right_col:
        st.subheader("What each option does")
        st.markdown("""
| Option | Effect |
|--------|--------|
| **Demo** | Ingest a tiny built-in dataset instantly for testing |
| **Summarise** | Run T5/LLM summarisation per document, stored in DB |
| **Build graph** | Extract entities + relations via REBEL or LLM |
| **Rebuild indices** | Re-builds BM25 pkl and FAISS vector index after ingest |
| **Only index** | Skip parsing/chunking, just rebuild existing indices |
| **T5** | Fast local summariser (no LLM required) |
| **rebel** | Fast local graph extractor (Babelscape model) |
""")

    if submitted:
        # Resolve effective data path: explicit path > uploaded files > demo
        _effective_path = ingest_path.strip() if ingest_path and ingest_path.strip() else None
        _tmp_upload_dir = None

        if not _effective_path and not ingest_demo and ingest_uploaded:
            # Save uploaded files to a temp directory
            _tmp_upload_dir = tempfile.mkdtemp(prefix="oml_upload_")
            for _uf in ingest_uploaded:
                (_dest := Path(_tmp_upload_dir) / _uf.name).write_bytes(_uf.getvalue())
            _effective_path = _tmp_upload_dir
            st.info(
                f"Saved {len(ingest_uploaded)} uploaded file(s) to temporary directory. "
                "will be cleaned up after this session."
            )

        ingest_progress = st.progress(0, text="Preparing ingest job...")
        with st.spinner("Running ingest pipeline (may take a while for large datasets)"):
            try:
                from oml.ingest.pipeline import IngestionPipeline

                ingest_progress.progress(20, text="Initializing pipeline...")
                storage_config = {"db_path": g_db_path} if ingest_storage == "sqlite" else None
                pipeline = IngestionPipeline(
                    storage_type=ingest_storage,
                    device=ingest_device,
                    storage_config=storage_config,
                )

                ingest_progress.progress(45, text="Parsing/chunking and building indices...")
                pipeline.run(
                    path=_effective_path,
                    limit=ingest_limit or None,
                    limit_chunks=ingest_limit_chunks or None,
                    rebuild_indices=ingest_rebuild,
                    only_index=ingest_only_index,
                    summarize=ingest_summarize,
                    summarizer_type=ingest_summarizer,
                    build_graph=ingest_graph,
                    graph_model=ingest_graph_model,
                    model=g_model,
                    demo=ingest_demo,
                )
                ingest_progress.progress(88, text="Finalizing outputs...")
                ingest_progress.progress(100, text="Ingest complete.")
                st.success("Ingest complete!")
                if ingest_storage == "sqlite":
                    st.caption(f"SQLite DB: `{Path(g_db_path).resolve()}`")
                if ingest_rebuild or ingest_only_index:
                    st.caption(f"BM25 index: `{(Path(g_artifacts) / 'bm25.pkl').resolve()}`")
                    st.caption(f"Vector index: `{(Path(g_artifacts) / 'vector.index').resolve()}`")
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc(), language="python")
            finally:
                # Clean up temp upload dir
                if _tmp_upload_dir:
                    try:
                        import shutil
                        shutil.rmtree(_tmp_upload_dir, ignore_errors=True)
                    except Exception:
                        pass


#
# TAB  TECHNIQUES
#

# ── Technique metadata (static catalogue) ─────────────────────────────────
_TECHNIQUE_CATALOG = [
    # Write-time — Ingestion
    {
        "name": "LLM Distiller",
        "key": "llm-distiller",
        "category": "ingest",
        "protocol": "IngestTechnique",
        "source": "teeg_pipeline.py",
        "description": "Converts raw text into a structured TOON-encoded AtomicNote via LLM. Falls back to Heuristic Distiller on parse failure.",
        "icon": "🧠",
    },
    {
        "name": "Heuristic Distiller",
        "key": "heuristic-distiller",
        "category": "ingest",
        "protocol": "IngestTechnique",
        "source": "teeg_pipeline.py",
        "description": "Rule-based distillation without LLM. Extracts first 40 words, auto-generates keywords. Zero-cost fallback.",
        "icon": "⚡",
    },
    # Write-time — Evolution
    {
        "name": "Stage 1 Pre-Screen",
        "key": "stage1-prescreen",
        "category": "evolution",
        "protocol": "EvolutionTechnique",
        "source": "evolver.py",
        "description": "Fast YES/NO contradiction filter using 33 regex patterns. Recall-biased: defaults YES on ambiguity.",
        "icon": "🔍",
    },
    {
        "name": "Stage 2 Judge",
        "key": "stage2-judge",
        "category": "evolution",
        "protocol": "EvolutionTechnique",
        "source": "evolver.py",
        "description": "Full LLM relationship classifier. Outputs one of 4 verdicts: CONTRADICTS, EXTENDS, SUPPORTS, UNRELATED.",
        "icon": "⚖️",
    },
    {
        "name": "Confidence Engine",
        "key": "confidence-engine",
        "category": "evolution",
        "protocol": "EvolutionTechnique",
        "source": "evolver.py",
        "description": "Bayesian confidence decay/boost. Handles archival when confidence drops below 0.15 threshold.",
        "icon": "📊",
    },
    {
        "name": "Belief Propagator",
        "key": "belief-propagator",
        "category": "evolution",
        "protocol": "EvolutionTechnique",
        "source": "evolver.py",
        "description": "Single-hop directed graph propagation. Spreads confidence changes to neighbors via weighted edges.",
        "icon": "🌊",
    },
    # Read-time — Retrieval
    {
        "name": "Vector Seeder",
        "key": "vector-seeder",
        "category": "retrieval",
        "protocol": "SeedingTechnique",
        "source": "scout.py",
        "description": "Seed discovery via vector similarity search with optional importance-weighted re-ranking.",
        "icon": "🎯",
    },
    {
        "name": "Graph Walker",
        "key": "graph-walker",
        "category": "retrieval",
        "protocol": "WalkingTechnique",
        "source": "scout.py",
        "description": "BFS traversal from seeds with per-hop decay and bidirectional edge exploration.",
        "icon": "🕸️",
    },
    # Read-time — Generation
    {
        "name": "Answer Generator",
        "key": "answer-generator",
        "category": "generation",
        "protocol": "—",
        "source": "teeg_pipeline.py",
        "description": "LLM answer generation from TEEG context block. Wraps context in [TEEG MEMORY] tags.",
        "icon": "💬",
    },
]

# Category metadata for display
_CATEGORY_META = {
    "ingest": {"label": "Ingestion", "color": "#2563eb", "desc": "Raw text → structured notes"},
    "evolution": {"label": "Evolution", "color": "#7c3aed", "desc": "Note relationship & confidence management"},
    "retrieval": {"label": "Retrieval", "color": "#059669", "desc": "Query → relevant notes"},
    "generation": {"label": "Generation", "color": "#d97706", "desc": "Context → natural language answer"},
}

# Built-in presets (mirrors registry.py PRESETS, but with full technique keys)
_BUILTIN_PRESETS = {
    "TEEG": {
        "description": "TOON-Encoded Evolving Graph — full memory lifecycle with LLM judge and graph walking.",
        "techniques": ["llm-distiller", "stage1-prescreen", "stage2-judge", "confidence-engine", "belief-propagator", "vector-seeder", "graph-walker", "answer-generator"],
    },
    "PRISM": {
        "description": "Probabilistic Retrieval with Intelligent Sparse Memory — superset of TEEG with vector seeding and multi-hop retrieval.",
        "techniques": ["llm-distiller", "stage1-prescreen", "stage2-judge", "confidence-engine", "belief-propagator", "vector-seeder", "graph-walker", "answer-generator"],
    },
    "Lightweight": {
        "description": "Heuristic-only ingestion with vector retrieval. No LLM calls during ingest. Suitable for demos or resource-constrained environments.",
        "techniques": ["heuristic-distiller", "vector-seeder", "answer-generator"],
    },
}

def _load_user_presets(artifacts_dir: str) -> dict:
    """Load user-created presets from a JSON file."""
    fp = Path(artifacts_dir) / "technique_presets.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_user_presets(artifacts_dir: str, presets: dict) -> None:
    """Persist user-created presets to a JSON file."""
    fp = Path(artifacts_dir) / "technique_presets.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(presets, indent=2), encoding="utf-8")


with tab_techniques:
    _hero_banner(
        "Technique Composer",
        "Browse the atomic technique catalog, explore built-in presets, and compose custom pipelines by mixing and matching techniques.",
        chip="Composable AI",
    )

    tech_composer_tab, tech_catalog_tab, tech_presets_tab = st.tabs(["Composer", "Catalog", "Presets"])

    # ── Sub-tab 0: Composer ───────────────────────────────────────────────
    with tech_composer_tab:
        from oml.app.composer import render_composer as _render_composer
        _current_theme = st.session_state.get("oml_theme") or st.query_params.get("oml_theme", "Dark")
        _render_composer(model=g_model, artifacts=g_artifacts, theme=_current_theme)

    # ── Sub-tab 1: Catalog ────────────────────────────────────────────────
    with tech_catalog_tab:
        st.subheader("Technique Catalog")
        st.caption("All atomic technique modules extracted from the monolithic PRISM, TEEG, and evolver pipelines.")

        # Group by category
        for cat_key, cat_meta in _CATEGORY_META.items():
            cat_techniques = [t for t in _TECHNIQUE_CATALOG if t["category"] == cat_key]
            if not cat_techniques:
                continue

            st.markdown(
                f"#### {cat_meta['label']}  "
                f"<span style='font-size:0.85em; color: var(--oml-muted);'>{cat_meta['desc']}</span>",
                unsafe_allow_html=True,
            )

            cols = st.columns(min(len(cat_techniques), 3))
            for idx, tech in enumerate(cat_techniques):
                with cols[idx % len(cols)]:
                    st.markdown(
                        f"""<div style="
                            border: 1px solid var(--oml-border);
                            border-radius: 12px;
                            padding: 16px;
                            margin-bottom: 12px;
                            background: var(--oml-card);
                        ">
                        <div style="font-size: 1.4em; margin-bottom: 4px;">{tech['icon']} <strong>{tech['name']}</strong></div>
                        <div style="font-size: 0.82em; color: var(--oml-muted); margin-bottom: 8px;">
                            <code>{tech['key']}</code> &middot; {tech['protocol']} &middot; from <code>{tech['source']}</code>
                        </div>
                        <div style="font-size: 0.9em; color: var(--oml-ink);">{tech['description']}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.markdown(
            "**Protocol interfaces** define the contract each technique must satisfy.  "
            "See `oml/techniques/protocols.py` for the 6 Protocol definitions: "
            "`IngestTechnique`, `EvolutionTechnique`, `CompressionTechnique`, "
            "`RetrievalTechnique`, `SeedingTechnique`, `WalkingTechnique`."
        )

    # ── Sub-tab 2: Presets ────────────────────────────────────────────────
    with tech_presets_tab:
        st.subheader("Pipeline Presets")
        st.caption("Built-in and user-created technique presets. Each preset is a named combination of techniques that form a complete pipeline.")

        _all_technique_keys = {t["key"] for t in _TECHNIQUE_CATALOG}

        # Built-in presets
        st.markdown("#### Built-in Presets")
        preset_cols = st.columns(len(_BUILTIN_PRESETS))
        for col, (preset_name, preset_data) in zip(preset_cols, _BUILTIN_PRESETS.items()):
            with col:
                st.markdown(
                    f"""<div style="
                        border: 2px solid var(--oml-accent);
                        border-radius: 12px;
                        padding: 16px;
                        background: var(--oml-card);
                        min-height: 200px;
                    ">
                    <div style="font-size: 1.3em; font-weight: bold; margin-bottom: 6px;">{preset_name}</div>
                    <div style="font-size: 0.85em; color: var(--oml-muted); margin-bottom: 12px;">{preset_data['description']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                # Show technique chips
                for tk in preset_data["techniques"]:
                    _t = next((t for t in _TECHNIQUE_CATALOG if t["key"] == tk), None)
                    if _t:
                        st.markdown(
                            f"<span style='display:inline-block; background: var(--oml-accent); color: #fff; "
                            f"border-radius: 20px; padding: 2px 10px; font-size: 0.8em; margin: 2px;'>"
                            f"{_t['icon']} {_t['name']}</span>",
                            unsafe_allow_html=True,
                        )

        # User presets
        st.markdown("---")
        st.markdown("#### User Presets")
        user_presets = _load_user_presets(g_artifacts)
        if not user_presets:
            st.info("No user presets yet. Use the **Builder** tab to create custom technique combinations.")
        else:
            u_cols = st.columns(min(len(user_presets), 3))
            for idx, (up_name, up_data) in enumerate(user_presets.items()):
                with u_cols[idx % len(u_cols)]:
                    st.markdown(
                        f"""<div style="
                            border: 1px solid var(--oml-border);
                            border-radius: 12px;
                            padding: 16px;
                            background: var(--oml-card);
                        ">
                        <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 4px;">{up_name}</div>
                        <div style="font-size: 0.85em; color: var(--oml-muted); margin-bottom: 8px;">{up_data.get('description', '')}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    for tk in up_data.get("techniques", []):
                        _t = next((t for t in _TECHNIQUE_CATALOG if t["key"] == tk), None)
                        label = _t["icon"] + " " + _t["name"] if _t else tk
                        st.markdown(
                            f"<span style='display:inline-block; background: var(--oml-accent); color: #fff; "
                            f"border-radius: 20px; padding: 2px 10px; font-size: 0.8em; margin: 2px;'>"
                            f"{label}</span>",
                            unsafe_allow_html=True,
                        )
                    if st.button("Delete", key=f"del_preset_{up_name}", type="secondary"):
                        del user_presets[up_name]
                        _save_user_presets(g_artifacts, user_presets)
                        st.success(f"Deleted preset **{up_name}**.")
                        st.rerun()



#
# TAB  TEEG
#

with tab_teeg:
    _hero_banner(
        "TEEG Memory",
        "TOON-encoded evolving graph memory for ingesting atomic notes and querying relation context.",
        chip="Long-term memory",
    )

    teeg_ingest_tab, teeg_query_tab, teeg_stats_tab = st.tabs(["Ingest", "Query", "Stats"])

    #  Shared TEEG settings 
    teeg_dir     = g_artifacts.rstrip("/") + "/teeg_store"
    teeg_model   = g_model
    teeg_budget  = 3000

    #  TEEG Ingest 
    with teeg_ingest_tab:
        st.subheader("Ingest raw text as an AtomicNote")

        teeg_mode = st.radio("Input mode", ["Text", "File", "Batch file"], horizontal=True, key="teeg_mode")

        if teeg_mode == "Text":
            teeg_raw = st.text_area("Raw text", height=120, placeholder="Victor Frankenstein created the creature in 1797", key="teeg_raw_text")
            teeg_file_path = None
            teeg_batch_path = None
        elif teeg_mode == "File":
            teeg_file_path = st.text_input("File path", placeholder="/path/to/chapter5.txt", key="teeg_file_path")
            teeg_raw = None
            teeg_batch_path = None
        else:
            teeg_batch_path = st.text_input("Batch file (one text per line)", placeholder="facts.txt", key="teeg_batch_path")
            teeg_raw = None
            teeg_file_path = None

        with st.expander("Advanced options"):
            teeg_context_hint = st.text_input("Context hint (source / chapter / date)", key="teeg_ctx_hint")
            teeg_source_id    = st.text_input("Source ID (optional link to document)", key="teeg_src_id")
            teeg_artifacts    = st.text_input("TEEG storage dir", value="teeg_store", key="teeg_artifacts")
            teeg_show_note    = st.checkbox("Show resulting note (TOON format)", value=True, key="teeg_show_note")

        if st.button("Ingest Note", type="primary", key="teeg_ingest_btn"):
            teeg_ingest_progress = st.progress(0, text="Preparing TEEG ingest...")
            with st.spinner("Distilling and evolving memory"):
                try:
                    from oml.memory.teeg_pipeline import TEEGPipeline

                    teeg_ingest_progress.progress(20, text="Loading TEEG pipeline...")
                    pipeline = TEEGPipeline(
                        artifacts_dir=teeg_artifacts,
                        model=teeg_model,
                        token_budget=teeg_budget,
                    )

                    if teeg_mode == "Batch file" and teeg_batch_path:
                        bp = Path(teeg_batch_path)
                        if not bp.exists():
                            st.error(f"File not found: {teeg_batch_path}")
                        else:
                            lines = [l.strip() for l in bp.read_text(encoding="utf-8").splitlines() if l.strip()]
                            teeg_ingest_progress.progress(60, text="Ingesting batch notes...")
                            notes = pipeline.ingest_batch(lines, context_hint=teeg_context_hint)
                            pipeline.save()
                            st.success(f" Ingested {len(notes)} notes.")
                            if teeg_show_note:
                                for n in notes:
                                    st.code(n.to_toon(), language="yaml")
                                    st.divider()

                    elif teeg_mode == "File" and teeg_file_path:
                        fp = Path(teeg_file_path)
                        if not fp.exists():
                            st.error(f"File not found: {teeg_file_path}")
                        else:
                            raw = fp.read_text(encoding="utf-8")
                            teeg_ingest_progress.progress(60, text="Ingesting file note...")
                            note = pipeline.ingest(raw, context_hint=teeg_context_hint, source_id=teeg_source_id or fp.name)
                            pipeline.save()
                            st.success(f" Note stored: `{note.note_id}`")
                            if teeg_show_note:
                                st.code(note.to_toon(), language="yaml")

                    elif teeg_mode == "Text" and teeg_raw and teeg_raw.strip():
                        teeg_ingest_progress.progress(60, text="Ingesting text note...")
                        note = pipeline.ingest(teeg_raw, context_hint=teeg_context_hint, source_id=teeg_source_id)
                        pipeline.save()
                        st.success(f" Note stored: `{note.note_id}`")
                        if teeg_show_note:
                            st.code(note.to_toon(), language="yaml")
                    else:
                        st.warning("Provide text, file path, or batch file.")
                    teeg_ingest_progress.progress(100, text="TEEG ingest complete.")

                except Exception as exc:
                    teeg_ingest_progress.empty()
                    st.error(f"TEEG ingest failed: {exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc(), language="python")

    #  TEEG Query 
    with teeg_query_tab:
        st.subheader("Query the TEEG memory graph")

        teeg_question = st.text_input(
            "Question",
            placeholder="Who created the creature?",
            key="teeg_question",
        )

        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            teeg_top_k  = st.number_input("Top-K notes", 1, 30, 8, key="teeg_topk")
        with q_col2:
            teeg_hops   = st.number_input("Graph hops", 1, 5, 2, key="teeg_hops")
        with q_col3:
            teeg_q_mode = st.selectbox("Mode", ["Full answer", "Search only", "Explain traversal"], key="teeg_q_mode")

        teeg_show_ctx = st.checkbox("Show TOON context block", value=False, key="teeg_show_ctx")
        teeg_q_artifacts = st.text_input("TEEG storage dir", value="teeg_store", key="teeg_q_artifacts")

        if st.button("Query TEEG", type="primary", key="teeg_query_btn", disabled=not (teeg_question or "").strip()):
            teeg_query_progress = st.progress(0, text="Preparing TEEG query...")
            with st.spinner("Traversing graph and generating answer"):
                try:
                    from oml.memory.teeg_pipeline import TEEGPipeline

                    teeg_query_progress.progress(20, text="Loading TEEG store...")
                    pipeline = TEEGPipeline(
                        artifacts_dir=teeg_q_artifacts,
                        model=teeg_model,
                        token_budget=teeg_budget,
                        scout_top_k=teeg_top_k,
                        scout_max_hops=teeg_hops,
                    )
                    stats = pipeline.stats()
                    if stats["active_notes"] == 0:
                        teeg_query_progress.empty()
                        st.warning("No notes found. Run **Ingest** first.")
                    else:
                        teeg_query_progress.progress(55, text="Traversing graph memory...")
                        if teeg_q_mode == "Explain traversal":
                            explanation = pipeline.explain_query(teeg_question, top_k=teeg_top_k)
                            st.subheader("Traversal Explanation")
                            st.text(explanation)

                        elif teeg_q_mode == "Search only":
                            results_list = pipeline.search(teeg_question, top_k=teeg_top_k)
                            if not results_list:
                                st.warning("No matching notes found.")
                            else:
                                st.subheader(f"Top {len(results_list)} matching notes")
                                for i, (note, score, hops) in enumerate(results_list, 1):
                                    label = "seed" if hops == 0 else f"hop-{hops}"
                                    with st.expander(f"{i}. [{label}]  score={score:.3f}  id={note.note_id}"):
                                        st.write(note.content)
                                        if note.tags:
                                            st.caption("Tags: " + ", ".join(note.tags))
                                        if note.keywords:
                                            st.caption("Keywords: " + ", ".join(note.keywords))

                        else:  # Full answer
                            answer, context_str = pipeline.query(
                                teeg_question, top_k=teeg_top_k, return_context=True
                            )
                            st.subheader("Answer")
                            st.write(answer)
                            if teeg_show_ctx:
                                st.subheader("TOON Context Block")
                                st.code(context_str, language="yaml")

                        st.caption(
                            f"Store: {stats['active_notes']} active notes  |  "
                            f"{stats['graph_edges']} edges  |  model: {teeg_model}"
                        )
                        teeg_query_progress.progress(100, text="TEEG query complete.")

                except Exception as exc:
                    teeg_query_progress.empty()
                    st.error(f"TEEG query failed: {exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc(), language="python")

    #  TEEG Stats 
    with teeg_stats_tab:
        st.subheader("TEEG Store Statistics")
        teeg_stats_dir = st.text_input("TEEG storage dir", value="teeg_store", key="teeg_stats_dir")

        if st.button("Refresh Stats", key="teeg_stats_btn"):
            try:
                from oml.memory.teeg_pipeline import TEEGPipeline
                pipeline = TEEGPipeline(artifacts_dir=teeg_stats_dir, model=teeg_model)
                stats = pipeline.stats()
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Active Notes", stats.get("active_notes", 0))
                col_s2.metric("Total Notes", stats.get("total_notes", 0))
                col_s3.metric("Graph Edges", stats.get("graph_edges", 0))
            except Exception as exc:
                st.error(f"Could not load stats: {exc}")



# 
# TAB 5  PRISM
# 


with tab_prism:
    _hero_banner(
        "PRISM Memory",
        "Probabilistic Retrieval with Intelligent Sparse Memory: MinHash dedup, semantic delta encoding, "
        "and LLM call coalescing on top of TEEG.",
        chip="Efficiency layers",
    )

    prism_ingest_tab, prism_query_tab, prism_stats_tab = st.tabs(["Ingest", "Query", "Stats"])

    prism_dir    = g_artifacts.rstrip("/") + "/teeg_store"
    prism_model  = g_model
    prism_budget = 3000

    with prism_ingest_tab:
        st.subheader("Ingest text with PRISM efficiency layers")

        prism_mode = st.radio(
            "Input mode", ["Text", "File", "Batch (multi-line)"],
            horizontal=True, key="prism_mode"
        )

        if prism_mode == "Text":
            prism_raw = st.text_area("Raw text", height=120,
                placeholder="Victor Frankenstein created the creature\u2026", key="prism_raw_text")
            prism_file_path = None
            prism_batch_path = None
        elif prism_mode == "File":
            prism_file_path = st.text_input("File path", placeholder="/path/to/chapter5.txt", key="prism_file_path")
            prism_raw = None
            prism_batch_path = None
        else:
            prism_batch_path = st.text_input("Batch file (one text per line)", placeholder="facts.txt", key="prism_batch_path")
            prism_raw = None
            prism_file_path = None

        with st.expander("Advanced options"):
            prism_context_hint  = st.text_input("Context hint", key="prism_ctx_hint")
            prism_source_id     = st.text_input("Source ID (optional)", key="prism_src_id")
            prism_artifacts     = st.text_input("Storage dir", value="teeg_store", key="prism_artifacts")
            prism_dedup_thresh  = st.slider(
                "Dedup threshold (SketchGate Jaccard)", 0.5, 1.0, 0.75, 0.05,
                key="prism_dedup", help="Similarity above which a new ingest is skipped as near-duplicate."
            )
            prism_batch_size    = st.number_input("Batch size (LLM coalescing)", 2, 32, 8, key="prism_bsize")
            prism_show_note     = st.checkbox("Show resulting note (TOON format)", value=True, key="prism_show_note")

        if st.button("Ingest via PRISM", type="primary", key="prism_ingest_btn"):
            prism_ingest_progress = st.progress(0, text="Preparing PRISM ingest...")
            with st.spinner("Running SketchGate \u2192 distil \u2192 evolve\u2026"):
                try:
                    from oml.memory.prism_pipeline import PRISMPipeline

                    prism_ingest_progress.progress(20, text="Loading PRISM pipeline...")
                    pipeline = PRISMPipeline(
                        artifacts_dir=prism_artifacts,
                        model=prism_model,
                        token_budget=prism_budget,
                        dedup_threshold=prism_dedup_thresh,
                        batch_size=prism_batch_size,
                    )

                    if prism_mode == "Batch (multi-line)" and prism_batch_path:
                        bp = Path(prism_batch_path)
                        if not bp.exists():
                            st.error(f"File not found: {prism_batch_path}")
                        else:
                            lines = [l.strip() for l in bp.read_text(encoding="utf-8").splitlines() if l.strip()]
                            prism_ingest_progress.progress(60, text="Running batch ingest...")
                            batch_result = pipeline.batch_ingest(lines, context_hints=[prism_context_hint] * len(lines))
                            pipeline.save()
                            st.success(
                                f"\u2705 Batch complete: {len(batch_result.notes)} notes  |  "
                                f"{batch_result.dedup_count} deduped  |  "
                                f"{batch_result.delta_count} deltas  |  "
                                f"efficiency {batch_result.call_efficiency:.1%}"
                            )
                            if prism_show_note:
                                for n in batch_result.notes[:10]:
                                    st.code(n.to_toon(), language="yaml")
                                    st.divider()

                    elif prism_mode == "File" and prism_file_path:
                        fp = Path(prism_file_path)
                        if not fp.exists():
                            st.error(f"File not found: {prism_file_path}")
                        else:
                            raw = fp.read_text(encoding="utf-8")
                            prism_ingest_progress.progress(60, text="Ingesting file note...")
                            result = pipeline.ingest(raw, context_hint=prism_context_hint, source_id=prism_source_id or fp.name)
                            pipeline.save()
                            if result.was_deduplicated:
                                st.info(f"\u267b\ufe0f Near-duplicate \u2014 merged into existing note `{result.merged_into}`")
                            else:
                                st.success(f"\u2705 Note stored: `{result.note.note_id}`")
                                if prism_show_note:
                                    st.code(result.note.to_toon(), language="yaml")

                    elif prism_mode == "Text" and prism_raw and prism_raw.strip():
                        prism_ingest_progress.progress(60, text="Ingesting text note...")
                        result = pipeline.ingest(prism_raw, context_hint=prism_context_hint, source_id=prism_source_id)
                        pipeline.save()
                        if result.was_deduplicated:
                            st.info(f"\u267b\ufe0f Near-duplicate \u2014 merged into existing note `{result.merged_into}`")
                        else:
                            st.success(f"\u2705 Note stored: `{result.note.note_id}`")
                            if prism_show_note:
                                st.code(result.note.to_toon(), language="yaml")
                    else:
                        st.warning("Provide text, file path, or batch file.")
                    prism_ingest_progress.progress(100, text="PRISM ingest complete.")

                except Exception as exc:
                    prism_ingest_progress.empty()
                    st.error(f"PRISM ingest failed: {exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc(), language="python")

    with prism_query_tab:
        st.subheader("Query PRISM memory (Scout + TieredContextPacker)")

        prism_question = st.text_input(
            "Question",
            placeholder="Who created the creature?",
            key="prism_question",
        )

        pq_col1, pq_col2, pq_col3 = st.columns(3)
        with pq_col1:
            prism_top_k  = st.number_input("Top-K notes", 1, 30, 8, key="prism_topk")
        with pq_col2:
            prism_hops   = st.number_input("Graph hops", 1, 5, 2, key="prism_hops")
        with pq_col3:
            prism_q_mode = st.selectbox("Mode", ["Full answer", "Search only"], key="prism_q_mode")

        prism_show_ctx      = st.checkbox("Show TOON context block", value=False, key="prism_show_ctx")
        prism_q_artifacts   = st.text_input("Storage dir", value="teeg_store", key="prism_q_artifacts")

        if st.button("Query PRISM", type="primary", key="prism_query_btn",
                     disabled=not (prism_question or "").strip()):
            prism_query_progress = st.progress(0, text="Preparing PRISM query...")
            with st.spinner("Traversing graph and generating answer\u2026"):
                try:
                    from oml.memory.prism_pipeline import PRISMPipeline

                    prism_query_progress.progress(20, text="Loading PRISM store...")
                    pipeline = PRISMPipeline(
                        artifacts_dir=prism_q_artifacts,
                        model=prism_model,
                        token_budget=prism_budget,
                        scout_top_k=prism_top_k,
                        scout_max_hops=prism_hops,
                    )
                    raw_s = pipeline.raw_stats()
                    active = raw_s["store"].get("active_notes", 0)

                    if active == 0:
                        prism_query_progress.empty()
                        st.warning("No notes found. Run **Ingest** first.")
                    elif prism_q_mode == "Search only":
                        prism_query_progress.progress(55, text="Searching PRISM memory...")
                        results_list = pipeline.search(prism_question, top_k=prism_top_k)
                        if not results_list:
                            st.warning("No matching notes found.")
                        else:
                            st.subheader(f"Top {len(results_list)} matching notes")
                            for i, (note, score, hops) in enumerate(results_list, 1):
                                label = "seed" if hops == 0 else f"hop-{hops}"
                                with st.expander(f"{i}. [{label}]  score={score:.3f}  id={note.note_id}"):
                                    st.write(note.content)
                                    if note.tags:
                                        st.caption("Tags: " + ", ".join(note.tags))
                    else:
                        prism_query_progress.progress(55, text="Generating PRISM answer...")
                        answer, context_str = pipeline.query(prism_question, top_k=prism_top_k, return_context=True)
                        st.subheader("Answer")
                        st.write(answer)
                        if prism_show_ctx:
                            st.subheader("TOON Context Block")
                            st.code(context_str, language="yaml")

                    st.caption(
                        f"Store: {active} active notes  |  "
                        f"Dedup rate: {raw_s['sketch_gate'].get('dedup_rate', 0):.1%}  |  "
                        f"model: {prism_model}"
                    )
                    prism_query_progress.progress(100, text="PRISM query complete.")

                except Exception as exc:
                    prism_query_progress.empty()
                    st.error(f"PRISM query failed: {exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc(), language="python")

    with prism_stats_tab:
        st.subheader("PRISM Efficiency Statistics")
        prism_stats_dir = st.text_input("Storage dir", value="teeg_store", key="prism_stats_dir")

        if st.button("Refresh Stats", key="prism_stats_btn"):
            try:
                from oml.memory.prism_pipeline import PRISMPipeline
                pipeline = PRISMPipeline(artifacts_dir=prism_stats_dir, model=prism_model)
                raw_s   = pipeline.raw_stats()
                store   = raw_s["store"]
                sketch  = raw_s["sketch_gate"]
                delta   = raw_s["delta_store"]
                batcher = raw_s["call_batcher"]

                st.subheader("Memory Store")
                c1, c2, c3 = st.columns(3)
                c1.metric("Active Notes",  store.get("active_notes", 0))
                c2.metric("Total Notes",   store.get("total_notes", 0))
                c3.metric("Graph Edges",   store.get("graph_edges", 0))

                st.subheader("Layer 1 \u2014 SketchGate (MinHash dedup)")
                s1, s2, s3 = st.columns(3)
                s1.metric("Dedup Rate",      f"{sketch.get('dedup_rate', 0):.1%}")
                s2.metric("Sketches stored", sketch.get("total_sketches", 0))
                s3.metric("Threshold",       sketch.get("dedup_threshold", 0.75))

                st.subheader("Layer 2 \u2014 DeltaStore (semantic patches)")
                d1, d2 = st.columns(2)
                d1.metric("Delta patches",     delta.get("total_patches", 0))
                d2.metric("Token savings est.", delta.get("token_savings_est", 0))

                st.subheader("Layer 3 \u2014 CallBatcher (LLM coalescing)")
                b1, b2, b3 = st.columns(3)
                b1.metric("LLM calls made",  batcher.get("calls_made", 0))
                b2.metric("LLM calls saved", batcher.get("calls_saved", 0))
                b3.metric("Call efficiency", f"{batcher.get('call_efficiency', 0):.1%}")

            except Exception as exc:
                st.error(f"Could not load PRISM stats: {exc}")

# ===============================================================
# TAB 6 - OPS
# ===============================================================

with tab_ops:
    _hero_banner(
        "Operations",
        "Administrative and advanced workflows that mirror CLI capabilities.",
        chip="Admin tooling",
    )

    ops_cons_tab, ops_eval_tab, ops_cache_tab, ops_api_tab = st.tabs(
        ["Consolidation", "Eval Task Runner", "Cache", "API Server"]
    )

    with ops_cons_tab:
        st.subheader("Thread + Memory Consolidation")
        oc1, oc2 = st.columns(2)

        with oc1:
            with st.container(border=True):
                st.markdown("**Thread-memory consolidation (`oml consolidate`)**")
                tc_db = st.text_input("SQLite DB path", value=g_db_path, key="ops_tc_db")
                tc_art = st.text_input("Artifacts dir", value=g_artifacts, key="ops_tc_art")
                tc_model = st.text_input("Model", value=g_model, key="ops_tc_model")
                tc_limit = st.number_input("Thread limit (0 = no limit)", min_value=0, max_value=100000, value=10, key="ops_tc_limit")

                if st.button("Run Thread Consolidation", type="primary", key="ops_tc_run"):
                    tc_progress = st.progress(0, text="Preparing consolidation...")
                    with st.spinner("Consolidating threads into MemoryNotes..."):
                        try:
                            tc_progress.progress(20, text="Loading thread metadata...")
                            from oml.memory.consolidate import consolidate_threads

                            tc_progress.progress(55, text="Summarizing and writing MemoryNotes...")
                            consolidate_threads(
                                db_path=tc_db,
                                model_name=tc_model,
                                limit=int(tc_limit),
                                artifacts_dir=tc_art,
                            )
                            new_counts = _safe_sqlite_counts(tc_db)
                            tc_progress.progress(100, text="Thread consolidation complete.")
                            st.success("Thread consolidation complete.")
                            st.caption(f"MemoryNotes in DB: {new_counts['memory_notes']:,}")
                        except Exception as exc:
                            st.error(f"Thread consolidation failed: {exc}")

        with oc2:
            with st.container(border=True):
                st.markdown("**TEEG cluster consolidation (`oml teeg-consolidate`)**")
                teeg_dir = st.text_input("TEEG store dir", value="teeg_store", key="ops_teeg_cons_dir")
                teeg_model = st.text_input("Model", value=g_model, key="ops_teeg_cons_model")
                teeg_min = st.number_input("Min cluster size", min_value=2, max_value=100, value=3, key="ops_teeg_cons_min")
                teeg_max = st.number_input("Max clusters per run", min_value=1, max_value=1000, value=10, key="ops_teeg_cons_max")
                teeg_dry = st.checkbox("Dry run", value=False, key="ops_teeg_cons_dry")
                teeg_no_llm = st.checkbox("No LLM summaries", value=False, key="ops_teeg_cons_no_llm")

                if st.button("Run TEEG Consolidation", type="primary", key="ops_teeg_cons_run"):
                    teeg_progress = st.progress(0, text="Preparing TEEG consolidation...")
                    with st.spinner("Running TEEG consolidation..."):
                        try:
                            teeg_progress.progress(30, text="Loading store and clustering notes...")
                            result, before, after = _run_cluster_consolidation(
                                store_dir=teeg_dir,
                                model_name=teeg_model,
                                min_cluster=int(teeg_min),
                                max_clusters=int(teeg_max),
                                dry_run=teeg_dry,
                                no_llm=teeg_no_llm,
                            )
                            teeg_progress.progress(100, text="TEEG consolidation complete.")
                            r1, r2, r3 = st.columns(3)
                            r1.metric("Clusters", result.clusters_found)
                            r2.metric("Archived", result.notes_archived)
                            r3.metric("Summaries", result.summaries_created)
                            st.caption(
                                f"Estimated token savings: ~{result.token_savings_est} per query | "
                                f"Active notes {before.get('active_notes', 0)} -> {after.get('active_notes', 0)}"
                            )
                        except Exception as exc:
                            st.error(f"TEEG consolidation failed: {exc}")

        st.divider()
        with st.container(border=True):
            st.markdown("**PRISM cluster consolidation (`oml prism-consolidate`)**")
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                prism_cons_dir = st.text_input("PRISM store dir", value="teeg_store", key="ops_prism_cons_dir")
                prism_cons_model = st.text_input("Model", value=g_model, key="ops_prism_cons_model")
            with pc2:
                prism_cons_min = st.number_input("Min cluster size", min_value=2, max_value=100, value=3, key="ops_prism_cons_min")
                prism_cons_max = st.number_input("Max clusters per run", min_value=1, max_value=1000, value=10, key="ops_prism_cons_max")
            with pc3:
                prism_cons_dry = st.checkbox("Dry run", value=False, key="ops_prism_cons_dry")
                prism_cons_no_llm = st.checkbox("No LLM summaries", value=False, key="ops_prism_cons_no_llm")

            if st.button("Run PRISM Consolidation", type="primary", key="ops_prism_cons_run"):
                prism_progress = st.progress(0, text="Preparing PRISM consolidation...")
                with st.spinner("Running PRISM consolidation..."):
                    try:
                        prism_progress.progress(30, text="Loading store and clustering notes...")
                        result, before, after = _run_cluster_consolidation(
                            store_dir=prism_cons_dir,
                            model_name=prism_cons_model,
                            min_cluster=int(prism_cons_min),
                            max_clusters=int(prism_cons_max),
                            dry_run=prism_cons_dry,
                            no_llm=prism_cons_no_llm,
                        )
                        prism_progress.progress(100, text="PRISM consolidation complete.")
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Clusters", result.clusters_found)
                        r2.metric("Archived", result.notes_archived)
                        r3.metric("Summaries", result.summaries_created)
                        r4.metric("Token savings est.", result.token_savings_est)
                        st.caption(
                            f"Active notes {before.get('active_notes', 0)} -> {after.get('active_notes', 0)}"
                        )
                    except Exception as exc:
                        st.error(f"PRISM consolidation failed: {exc}")

    with ops_eval_tab:
        st.subheader("Run one evaluation task (`oml eval`)")
        eval_tasks = _available_eval_tasks()
        e1, e2, e3 = st.columns([2, 2, 1])
        with e1:
            eval_task_name = st.selectbox("Task", options=eval_tasks, key="ops_eval_task_name")
        with e2:
            eval_model_name = st.text_input("Model", value=g_model, key="ops_eval_model_name")
        with e3:
            eval_limit = st.number_input("Limit", min_value=1, max_value=10000, value=10, key="ops_eval_limit")

        if st.button("Run Eval Task", type="primary", key="ops_eval_run"):
            eval_progress = st.progress(0, text=f"Preparing eval task '{eval_task_name}'...")
            with st.spinner(f"Running eval task '{eval_task_name}'..."):
                try:
                    eval_progress.progress(20, text="Loading eval task registry...")
                    import oml.eval.ablations  # noqa: F401
                    import oml.eval.tasks  # noqa: F401
                    from oml.eval.run import run_task

                    eval_progress.progress(55, text="Executing task...")
                    result = run_task(eval_task_name, model_name=eval_model_name, config={"limit": int(eval_limit)})
                    st.session_state.ops_last_eval = {
                        "task_name": result.task_name,
                        "score": result.score,
                        "details": result.details,
                        "model": eval_model_name,
                    }
                    eval_progress.progress(100, text="Evaluation complete.")
                    st.success("Evaluation finished. Reports were saved under `reports/`.")
                except Exception as exc:
                    st.error(f"Eval run failed: {exc}")
                    st.session_state.ops_last_eval = None

        if st.session_state.get("ops_last_eval"):
            last = st.session_state["ops_last_eval"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Task", last["task_name"])
            m2.metric("Score", f"{last['score']:.3f}")
            m3.metric("Model", last["model"])
            with st.expander("Details"):
                st.json(last["details"])

    with ops_cache_tab:
        st.subheader("LLM Cache Controls (`oml cache-stats`, `oml cache-clear`)")
        cache_dir = st.text_input("Cache/artifacts directory", value=g_artifacts, key="ops_cache_dir")
        stats, by_model = _safe_cache_stats(cache_dir)

        if not stats:
            st.warning("Cache stats unavailable. Check the directory path and permissions.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Entries", stats.get("total_entries", 0))
            c2.metric("Session hit rate", f"{stats.get('hit_rate', 0):.1%}")
            c3.metric("Estimated calls saved", stats.get("estimated_calls_saved", 0))
            st.caption(f"Cache file: `{stats.get('cache_file', 'unknown')}`")

            if by_model:
                with st.expander("Entries by model"):
                    st.table([{"Model": m, "Entries": c} for m, c in sorted(by_model.items(), key=lambda x: -x[1])])

        st.divider()
        clear_model = st.text_input(
            "Optional model filter for clear",
            value="",
            placeholder="e.g. openai:gpt-4o-mini",
            key="ops_cache_clear_model",
        )
        clear_confirm = st.checkbox(
            "I understand this deletes cached responses.",
            value=False,
            key="ops_cache_clear_confirm",
        )

        if st.button("Clear Cache", disabled=not clear_confirm, key="ops_cache_clear_btn"):
            try:
                from oml.llm.cache import LLMCache

                cache = LLMCache(cache_path=cache_dir, mode="off")
                removed = cache.clear(model=clear_model.strip() or None)
                cache.save()
                scope = f"model '{clear_model.strip()}'" if clear_model.strip() else "all models"
                st.success(f"Cleared {removed} cache entr{'y' if removed == 1 else 'ies'} for {scope}.")
            except Exception as exc:
                st.error(f"Cache clear failed: {exc}")

    with ops_api_tab:
        st.subheader("API Server Controls (`oml api`)")
        a1, a2, a3, a4 = st.columns([2, 1, 1, 1])
        with a1:
            api_host = st.text_input("Host", value=st.session_state.api_server_host, key="ops_api_host")
        with a2:
            api_port = st.number_input("Port", min_value=1, max_value=65535, value=int(st.session_state.api_server_port), key="ops_api_port")
        with a3:
            api_reload = st.checkbox("Reload", value=False, key="ops_api_reload")
        with a4:
            api_workers = st.number_input("Workers", min_value=1, max_value=16, value=1, key="ops_api_workers")

        api_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "oml.api.server:app",
            "--host",
            api_host,
            "--port",
            str(int(api_port)),
        ]
        if api_reload:
            api_cmd.append("--reload")
        elif int(api_workers) > 1:
            api_cmd.extend(["--workers", str(int(api_workers))])

        st.code(" ".join(api_cmd), language="bash")

        proc = st.session_state.api_server_proc
        running = bool(proc and proc.poll() is None)
        if proc and proc.poll() is not None:
            st.session_state.api_server_proc = None
            running = False

        if running:
            st.success(f"API server running (PID {proc.pid})")
        else:
            st.info("API server is stopped.")

        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button("Start API Server", type="primary", key="ops_api_start"):
                if running:
                    st.warning("API server is already running.")
                else:
                    try:
                        new_proc = subprocess.Popen(
                            api_cmd,
                            cwd=str(Path.cwd()),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        st.session_state.api_server_proc = new_proc
                        st.session_state.api_server_cmd = " ".join(api_cmd)
                        st.session_state.api_server_host = api_host
                        st.session_state.api_server_port = int(api_port)
                        st.success("API server started.")
                    except Exception as exc:
                        st.error(f"Failed to start API server: {exc}")
        with stop_col:
            if st.button("Stop API Server", key="ops_api_stop"):
                if not running:
                    st.info("API server is not running.")
                else:
                    try:
                        proc.terminate()
                        time.sleep(0.8)
                        if proc.poll() is None:
                            proc.kill()
                        st.session_state.api_server_proc = None
                        st.success("API server stopped.")
                    except Exception as exc:
                        st.error(f"Failed to stop API server: {exc}")

        if st.session_state.api_server_cmd:
            st.caption(f"Last launch command: `{st.session_state.api_server_cmd}`")
        docs_url = f"http://{st.session_state.api_server_host}:{st.session_state.api_server_port}/docs"
        st.markdown(f"[Swagger Docs]({docs_url})")

# 
# TAB 5  REPORTS
# 

with tab_reports:
    _hero_banner(
        "Reports",
        "Browse experiment and evaluation reports saved under `reports/`.",
        chip="Run outputs",
    )

    reports_dir = Path("reports")

    #  File picker 
    if not reports_dir.exists():
        st.warning(f"No `{reports_dir}` directory found. Run an eval script to generate reports.")
    else:
        # Markdown reports
        md_files = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        # JSON run files
        json_files = sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        report_col, json_col = st.columns([3, 1])

        with report_col:
            if not md_files:
                st.info("No Markdown reports found yet.")
            else:
                selected_md = st.selectbox(
                    "Select report",
                    options=[f.name for f in md_files],
                    key="report_sel",
                )
                report_path = reports_dir / selected_md
                md_content = report_path.read_text(encoding="utf-8")

                st.markdown(md_content)

                with st.expander("Raw Markdown"):
                    st.code(md_content, language="markdown")

        with json_col:
            st.subheader("JSON run files")
            if not json_files:
                st.caption("No JSON files.")
            else:
                selected_json = st.selectbox(
                    "Run file",
                    options=[f.name for f in json_files],
                    key="json_sel",
                )
                json_path = reports_dir / selected_json
                import json as _json
                try:
                    data = _json.loads(json_path.read_text(encoding="utf-8"))
                    st.json(data)
                except Exception as exc:
                    st.error(f"Could not parse JSON: {exc}")

    #  Run combination eval 
    st.divider()
    st.subheader("Run Combination Evaluation")
    st.caption("Runs all 18 preset configurations against a dataset and writes a new report to `reports/`.")

    eval_model = st.text_input("Model for evaluation", value=g_model, key="eval_model")
    if st.button("Run Eval (Combinations)", key="run_eval_combo"):
        eval_combo_progress = st.progress(0, text="Preparing combination evaluation...")
        with st.spinner("Running combination evaluation  this may take several minutes"):
            try:
                eval_combo_progress.progress(20, text="Launching evaluation script...")
                result = subprocess.run(
                    [sys.executable, "scripts/eval_combinations.py"],
                    capture_output=True, text=True,
                    env={**os.environ, "OML_MODEL": eval_model},
                    timeout=600,
                )
                eval_combo_progress.progress(85, text="Collecting report output...")
                if result.returncode == 0:
                    eval_combo_progress.progress(100, text="Evaluation complete.")
                    st.success("Evaluation complete! Refresh the report list above.")
                    st.code(result.stdout[-3000:], language="text")
                else:
                    eval_combo_progress.empty()
                    st.error("Evaluation script exited with an error.")
                    st.code(result.stderr[-2000:], language="text")
            except subprocess.TimeoutExpired:
                eval_combo_progress.empty()
                st.error("Timed out after 10 minutes.")
            except Exception as exc:
                eval_combo_progress.empty()
                st.error(f"Failed to run eval: {exc}")


# 
# TAB 7  BENCHMARKS
# 

with tab_bench:
    _hero_banner(
        "Model Benchmarks",
        "Compare all LM Studio models across faithfulness, retrieval recall, "
        "TEEG/PRISM memory quality, and latency. "
        "Run `python scripts/benchmark_models.py` to populate this dashboard.",
        chip="Ranking + diagnostics",
    )

    _bench_results_path = Path("reports/benchmark_results.json")
    _bench_pred_path    = Path("reports/model_predictions.json")

    #  Load data 
    _bench_data = None
    _bench_preds = None

    if _bench_results_path.exists():
        try:
            with open(_bench_results_path, encoding="utf-8") as _f:
                _bench_data = json.load(_f)
        except Exception as _e:
            st.error(f"Could not read benchmark_results.json: {_e}")

    if _bench_pred_path.exists():
        try:
            with open(_bench_pred_path, encoding="utf-8") as _f:
                _bench_preds = json.load(_f)
        except Exception:
            pass

    if _bench_data is None:
        st.info(
            " No benchmark results found yet.\n\n"
            "Run the benchmark harness to generate data:\n"
            "```\npython scripts/benchmark_models.py\n```\n"
            "Or for a quick 2-task run:\n"
            "```\npython scripts/benchmark_models.py --quick\n```"
        )
    else:
        _models = _bench_data.get("models", {})
        _updated_at = _bench_data.get("updated_at", _bench_data.get("generated_at", "unknown"))

        b1, b2, b3 = st.columns(3)
        b1.metric("Models tested", len(_models))
        b2.metric("Last updated", str(_updated_at)[:16] if _updated_at else "unknown")
        b3.metric("Predictions file", "Loaded" if _bench_preds is not None else "Missing")

        if not _models:
            st.warning("benchmark_results.json exists but contains no model results yet.")
        else:
            #  Build comparison DataFrame 
            import pandas as pd

            _rows = []
            for _mid, _m in _models.items():
                _tasks = _m.get("tasks", {})
                _lat   = _m.get("latency", {})

                def _score(task_key):
                    """Return score for a task, or None if missing/errored."""
                    t = _tasks.get(task_key, {})
                    if t.get("status") == "error":
                        return None
                    v = t.get("score")
                    return round(v, 3) if isinstance(v, (int, float)) else None

                def _score_key(task_key, score_key="score"):
                    """Return a specific sub-key from a task dict."""
                    t = _tasks.get(task_key, {})
                    if t.get("status") == "error":
                        return None
                    v = t.get(score_key)
                    return round(v, 3) if isinstance(v, (int, float)) else None

                _rows.append({
                    "Model":            _m.get("display_name", _mid),
                    "Model ID":         _mid,
                    "Faithfulness":     _score("faithfulness"),
                    "Lost-in-Middle":   _score("lost_in_middle"),
                    "LiM Extended":     _score_key("lost_in_middle_extended", "mean_score"),
                    "Ret. Precision":   _score("retrieval_precision"),
                    "Cost/Latency":     _score("cost_latency"),
                    "OML vs RAG":       _score("oml_vs_rag"),
                    "Global Trends":    _score("global_trends"),
                    "TEEG Recall":      _score("teeg_cycle"),
                    "PRISM Score":      _score("prism_cycle"),
                    "Overall":          _m.get("overall_score"),
                    "Latency (s)":      _lat.get("mean_s"),
                    "Wall Time (s)":    _m.get("total_wall_s"),
                    "Benchmarked":      _m.get("benchmarked_at", "")[:16],
                })

            _df = pd.DataFrame(_rows).set_index("Model")

            #  Metric tabs 
            _btab_overview, _btab_charts, _btab_pred, _btab_raw = st.tabs([
                "Overview", "Charts", "Predictions", "Raw Data"
            ])

            # 
            # Sub-tab 1  Overview table
            # 
            with _btab_overview:
                st.subheader("Overall Comparison")

                # Colour-code numeric columns
                # Always show self-contained tasks; show artifact-dependent ones
                # only when at least one model has a non-None value for them.
                _self_contained_cols = [
                    "Faithfulness", "Lost-in-Middle", "LiM Extended",
                    "TEEG Recall", "PRISM Score", "Overall",
                ]
                _artifact_cols = [
                    "Ret. Precision", "Cost/Latency", "OML vs RAG", "Global Trends",
                ]
                _display_cols = list(_self_contained_cols)
                for _ac in _artifact_cols:
                    if _ac in _df.columns and _df[_ac].notna().any():
                        _display_cols.append(_ac)
                _display_cols.append("Latency (s)")

                _score_display_cols = [c for c in _display_cols if c != "Latency (s)"]
                try:
                    _styled = _df[_display_cols].style.format(
                        {c: "{:.3f}" for c in _score_display_cols},
                        na_rep=""
                    ).format({"Latency (s)": "{:.2f}s"}, na_rep="").background_gradient(
                        subset=_score_display_cols,
                        cmap="RdYlGn", vmin=0.0, vmax=1.0
                    ).background_gradient(
                        subset=["Latency (s)"],
                        cmap="RdYlGn_r", vmin=0, vmax=30
                    )
                    st.dataframe(_styled, width="stretch")
                except ImportError:
                    _fallback = _df[_display_cols].copy()
                    for _col in _score_display_cols:
                        if _col in _fallback.columns:
                            _fallback[_col] = _fallback[_col].round(3)
                    if "Latency (s)" in _fallback.columns:
                        _fallback["Latency (s)"] = _fallback["Latency (s)"].round(2)
                    st.info("Install `matplotlib` to enable gradient coloring in this table.")
                    st.dataframe(_fallback, width="stretch")

                # Quick winner callout
                _overall_valid = {
                    k: v for k, v in
                    {_m.get("display_name", _mid): _m.get("overall_score")
                     for _mid, _m in _models.items()}.items()
                    if isinstance(v, (int, float))
                }
                if _overall_valid:
                    _best_name = max(_overall_valid, key=_overall_valid.get)
                    _best_score = _overall_valid[_best_name]
                    st.success(f" **Best overall**: **{_best_name}**  score `{_best_score:.3f}`")

                # Latency champion
                _lat_valid = {
                    _m.get("display_name", _mid): _m.get("latency", {}).get("mean_s")
                    for _mid, _m in _models.items()
                    if isinstance(_m.get("latency", {}).get("mean_s"), (int, float))
                }
                if _lat_valid:
                    _fastest = min(_lat_valid, key=_lat_valid.get)
                    st.info(f" **Fastest**: **{_fastest}**  `{_lat_valid[_fastest]:.2f}s` avg latency")

            # 
            # Sub-tab 2  Charts
            # 
            with _btab_charts:
                _metric_cols = [
                    "Faithfulness", "Lost-in-Middle", "LiM Extended",
                    "TEEG Recall", "PRISM Score", "Overall",
                ]
                _chart_df = _df[_metric_cols].dropna(how="all")

                if not _chart_df.empty:
                    st.subheader("Per-Metric Bar Charts")

                    _chart_metric = st.selectbox(
                        "Select metric to chart",
                        _metric_cols,
                        key="bench_chart_metric"
                    )
                    _col_data = _chart_df[[_chart_metric]].dropna()
                    if not _col_data.empty:
                        _col_data = _col_data.sort_values(_chart_metric, ascending=False)
                        st.bar_chart(_col_data, width="stretch", height=300)
                        st.caption("Higher is better - Scale 0-1")

                    st.divider()
                    st.subheader("All Metrics Side-by-Side")
                    _all_data = _chart_df.dropna(how="all")
                    if not _all_data.empty:
                        st.bar_chart(_all_data, width="stretch", height=350)

                    st.divider()
                    st.subheader("Latency Comparison")
                    _lat_df = pd.DataFrame([
                        {
                            "Model": _m.get("display_name", _mid),
                            "Avg Latency (s)": _m.get("latency", {}).get("mean_s"),
                        }
                        for _mid, _m in _models.items()
                        if isinstance(_m.get("latency", {}).get("mean_s"), (int, float))
                    ]).set_index("Model").sort_values("Avg Latency (s)")
                    if not _lat_df.empty:
                        st.bar_chart(_lat_df, width="stretch", height=250)
                        st.caption("Lower is better")
                else:
                    st.info("No numeric scores available yet  run the benchmark first.")

            # 
            # Sub-tab 3  Predictions vs Actual
            # 
            with _btab_pred:
                st.subheader("Prediction vs Actual Results")

                if _bench_preds is None:
                    st.info("No predictions file found at `reports/model_predictions.json`.")
                else:
                    _preds = _bench_preds.get("predictions", {})
                    _pred_rows = []
                    _metric_map = {
                        "Faithfulness":   "faithfulness",
                        "Lost-in-Middle": "lost_in_middle",
                        "TEEG Recall":    "teeg_cycle",
                        "PRISM Score":    "prism_cycle",
                    }

                    for _mid, _m in _models.items():
                        _model_str = f"lmstudio:{_mid}"
                        _pred = _preds.get(_model_str, {})
                        _tasks = _m.get("tasks", {})
                        _name = _m.get("display_name", _mid)

                        for _col, _task_key in _metric_map.items():
                            _pred_val = _pred.get({
                                "Faithfulness":   "faithfulness",
                                "Lost-in-Middle": "lost_in_middle",
                                "TEEG Recall":    "teeg_recall",
                                "PRISM Score":    "prism_score",
                            }.get(_col, ""))
                            _actual_val = _tasks.get(_task_key, {}).get("score")

                            if _pred_val is not None or _actual_val is not None:
                                _delta = (
                                    round(_actual_val - _pred_val, 3)
                                    if isinstance(_pred_val, (int, float)) and isinstance(_actual_val, (int, float))
                                    else None
                                )
                                _pred_rows.append({
                                    "Model":    _name,
                                    "Metric":   _col,
                                    "Predicted": _pred_val,
                                    "Actual":    _actual_val,
                                    "Delta":     _delta,
                                    "Accurate":  (
                                        "" if _delta is not None and abs(_delta) <= 0.15
                                        else ("" if _delta is not None else "")
                                    ),
                                })

                    if _pred_rows:
                        _pred_df = pd.DataFrame(_pred_rows)

                        def _fmt_score(v):
                            return f"{v:.3f}" if isinstance(v, (int, float)) else ""

                        def _fmt_delta(v):
                            if not isinstance(v, (int, float)):
                                return ""
                            return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"

                        # Summary accuracy
                        _accurate_count = sum(1 for r in _pred_rows if r["Accurate"] == "")
                        _total_count    = sum(1 for r in _pred_rows if r["Accurate"] != "")
                        if _total_count:
                            st.metric(
                                "Prediction Accuracy (+/-0.15 tolerance)",
                                f"{_accurate_count}/{_total_count}",
                                f"{_accurate_count/_total_count*100:.0f}%"
                            )

                        _pred_styled = _pred_df.style.format({
                            "Predicted": _fmt_score,
                            "Actual":    _fmt_score,
                            "Delta":     _fmt_delta,
                        })
                        st.dataframe(_pred_styled, width="stretch")

                        # Per-model comparison charts
                        st.divider()
                        st.subheader("Predicted vs Actual per Model")
                        _sel_model = st.selectbox(
                            "Select model",
                            sorted(_pred_df["Model"].unique()),
                            key="bench_pred_model"
                        )
                        _model_pred_df = _pred_df[_pred_df["Model"] == _sel_model].copy()
                        if not _model_pred_df.empty:
                            _cmp_df = _model_pred_df[["Metric", "Predicted", "Actual"]].dropna(
                                subset=["Predicted", "Actual"]
                            ).set_index("Metric")
                            if not _cmp_df.empty:
                                st.bar_chart(_cmp_df, width="stretch", height=280)
                    else:
                        st.info("No prediction/actual data to compare yet.")

                    # Show prediction rationale
                    with st.expander(" Prediction Rationale"):
                        _rationale = _bench_preds.get("rationale", {})
                        for _model_str, _reason in _rationale.items():
                            _short_id = _model_str.split(":")[-1] if ":" in _model_str else _model_str
                            _dn = _models.get(_short_id, {}).get("display_name", _short_id)
                            st.markdown(f"**{_dn}**: {_reason}")

            # 
            # Sub-tab 4  Raw data
            # 
            with _btab_raw:
                st.subheader("Raw Benchmark Data")
                _sel_raw_model = st.selectbox(
                    "Select model to inspect",
                    list(_models.keys()),
                    format_func=lambda k: _models[k].get("display_name", k),
                    key="bench_raw_model"
                )
                if _sel_raw_model:
                    _raw_data = _models[_sel_raw_model]
                    st.json(_raw_data)

                st.divider()
                # Download button for the full JSON
                _bench_json_str = json.dumps(_bench_data, indent=2, default=str)
                st.download_button(
                    label="Download benchmark_results.json",
                    data=_bench_json_str,
                    file_name="benchmark_results.json",
                    mime="application/json",
                )

    #  Quick-run controls 
    st.divider()
    with st.expander("Run Benchmarks from UI"):
        st.markdown(
            "To run benchmarks directly, use the terminal commands below. "
            "Refresh this page when complete."
        )
        _quick_mode = st.toggle("Quick mode (faithfulness + LiM only)", value=True, key="bench_quick_toggle")
        _quick_flag = "--quick" if _quick_mode else ""
        st.code(f"python scripts/benchmark_models.py {_quick_flag}".strip(), language="bash")
        st.caption(
            "Quick mode takes ~2-5 min per model. Full mode (with TEEG/PRISM) can take 15-60 min per model."
        )
        if st.button("Refresh benchmark data", key="bench_refresh"):
            st.rerun()


# 
# TAB 6  CHUNK / DATABASE EXPLORER
# 

with tab_explorer:
    _hero_banner(
        "Database Explorer",
        f"Browse, search, and delete individual chunks and documents from the SQLite store at `{g_db_path}`.",
        chip="Inspect + clean records",
    )

    exp_chunks_tab, exp_docs_tab = st.tabs(["Chunks", "Documents"])

    #  helper: check if DB exists 
    def _db_exists() -> bool:
        return Path(g_db_path).exists()

    if _db_exists():
        try:
            _conn = sqlite3.connect(g_db_path)
            _n_chunks = int(_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            try:
                _n_docs = int(_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            except Exception:
                _n_docs = 0
            _conn.close()
            ex1, ex2, ex3 = st.columns(3)
            ex1.metric("Documents", f"{_n_docs:,}")
            ex2.metric("Chunks", f"{_n_chunks:,}")
            ex3.metric("Database", "Connected")
        except Exception:
            st.info("Connected path exists but counts could not be loaded.")

    if not _db_exists():
        st.warning(
            f"Database not found at `{g_db_path}`. "
            "Ingest some data first ( Ingest tab) or change the SQLite DB Path in the sidebar."
        )

    #  Chunks sub-tab 
    with exp_chunks_tab:
        if _db_exists():
            # Search / filter row
            exp_search_col, exp_page_size_col = st.columns([3, 1])
            with exp_search_col:
                exp_search = st.text_input(
                    "Filter chunks",
                    placeholder="keyword or phrase",
                    key="exp_chunk_search",
                )
            with exp_page_size_col:
                exp_page_size = st.selectbox("Rows / page", [20, 50, 100], index=0, key="exp_page_size")

            # Pagination state
            if "exp_chunk_page" not in st.session_state:
                st.session_state.exp_chunk_page = 0

            try:
                conn = sqlite3.connect(g_db_path)

                # Count rows
                if exp_search:
                    _like = f"%{exp_search}%"
                    total_chunks = conn.execute(
                        "SELECT COUNT(*) FROM chunks WHERE chunk_text LIKE ?", (_like,)
                    ).fetchone()[0]
                else:
                    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

                n_pages = max(1, (total_chunks + exp_page_size - 1) // exp_page_size)
                # Clamp page index
                st.session_state.exp_chunk_page = min(st.session_state.exp_chunk_page, n_pages - 1)

                # Pagination controls
                nav_c1, nav_c2, nav_c3 = st.columns([1, 3, 1])
                with nav_c1:
                    if st.button("Prev", key="exp_chunk_prev", disabled=st.session_state.exp_chunk_page == 0):
                        st.session_state.exp_chunk_page -= 1
                        st.rerun()
                with nav_c2:
                    st.caption(
                        f"Page **{st.session_state.exp_chunk_page + 1}** / {n_pages}  "
                        f"({total_chunks:,} total chunks)"
                    )
                with nav_c3:
                    if st.button("Next", key="exp_chunk_next", disabled=st.session_state.exp_chunk_page >= n_pages - 1):
                        st.session_state.exp_chunk_page += 1
                        st.rerun()

                # Build column list from what actually exists in this DB
                _chunk_cols = [
                    row[1]
                    for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
                ]
                _wanted_chunk = ["chunk_id", "doc_id", "chunk_text", "start_char", "end_char"]
                _select_chunk = [c for c in _wanted_chunk if c in _chunk_cols]
                if not _select_chunk:
                    _select_chunk = _chunk_cols[:5]
                _sel_str = ", ".join(_select_chunk)

                # Fetch page
                _offset = st.session_state.exp_chunk_page * exp_page_size
                if exp_search:
                    rows = conn.execute(
                        f"SELECT {_sel_str} FROM chunks WHERE chunk_text LIKE ? LIMIT ? OFFSET ?",
                        (_like, exp_page_size, _offset),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        f"SELECT {_sel_str} FROM chunks LIMIT ? OFFSET ?",
                        (exp_page_size, _offset),
                    ).fetchall()
                conn.close()

                if rows:
                    import pandas as _pd
                    _df = _pd.DataFrame(rows, columns=_select_chunk)
                    # Truncate text for display
                    if "chunk_text" in _df.columns:
                        _df["chunk_text"] = _df["chunk_text"].str[:300]
                    st.dataframe(_df, width="stretch", hide_index=True)
                else:
                    st.info(
                        "No chunks found."
                        if not exp_search
                        else f"No chunks matching **{exp_search!r}**."
                    )

            except Exception as _exc:
                st.error(f"Could not query chunks: {_exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc(), language="python")

            #  Delete a chunk 
            with st.expander("Delete a chunk by ID"):
                del_chunk_id = st.text_input("Chunk ID", key="del_chunk_id_input")
                if st.button("Delete Chunk", type="secondary", key="del_chunk_btn"):
                    if del_chunk_id.strip():
                        try:
                            _c = sqlite3.connect(g_db_path)
                            _deleted = _c.execute(
                                "DELETE FROM chunks WHERE chunk_id = ?", (del_chunk_id.strip(),)
                            ).rowcount
                            _c.commit()
                            _c.close()
                            if _deleted:
                                st.success(f"Deleted chunk `{del_chunk_id}`.")
                                st.session_state.exp_chunk_page = 0
                                st.rerun()
                            else:
                                st.warning(f"No chunk found with ID `{del_chunk_id}`.")
                        except Exception as _exc:
                            st.error(f"Delete failed: {_exc}")
                    else:
                        st.warning("Enter a chunk ID first.")

    #  Documents sub-tab 
    with exp_docs_tab:
        if _db_exists():
            try:
                conn = sqlite3.connect(g_db_path)
                _doc_cols_available = [
                    row[1]
                    for row in conn.execute("PRAGMA table_info(documents)").fetchall()
                ]
                conn.close()
            except Exception:
                _doc_cols_available = []

            if not _doc_cols_available:
                st.info("No `documents` table found in this database.")
            else:
                try:
                    conn = sqlite3.connect(g_db_path)
                    # Build a safe column select from what actually exists
                    _wanted = ["doc_id", "source_file", "title", "created_at", "summary"]
                    _select_cols = [c for c in _wanted if c in _doc_cols_available]
                    if not _select_cols:
                        _select_cols = _doc_cols_available[:6]
                    _select_str = ", ".join(_select_cols)

                    doc_rows = conn.execute(f"SELECT {_select_str} FROM documents").fetchall()
                    total_docs = len(doc_rows)
                    conn.close()

                    if doc_rows:
                        import pandas as _pd
                        _doc_df = _pd.DataFrame(doc_rows, columns=_select_cols)
                        if "summary" in _doc_df.columns:
                            _doc_df["summary"] = _doc_df["summary"].str[:200]
                        st.dataframe(
                            _doc_df,
                            width="stretch",
                            hide_index=True,
                        )
                        st.caption(f"{total_docs} document(s) in the database.")
                    else:
                        st.info("No documents in the database.")

                except Exception as _exc:
                    st.error(f"Could not query documents: {_exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc(), language="python")

                #  Delete a document + its chunks 
                with st.expander("Delete a document and all its chunks"):
                    del_doc_id = st.text_input("Document ID", key="del_doc_id_input")
                    st.warning(
                        "This deletes the document **and all chunks** derived from it. "
                        "Note: vector indices are **not** automatically rebuilt; "
                        "run *Rebuild Indices* in the Ingest tab afterwards.",
                    )
                    if st.button("Delete Document + Chunks", type="secondary", key="del_doc_btn"):
                        if del_doc_id.strip():
                            try:
                                _c = sqlite3.connect(g_db_path)
                                _chunks_del = _c.execute(
                                    "DELETE FROM chunks WHERE doc_id = ?", (del_doc_id.strip(),)
                                ).rowcount
                                _docs_del = _c.execute(
                                    "DELETE FROM documents WHERE doc_id = ?", (del_doc_id.strip(),)
                                ).rowcount
                                _c.commit()
                                _c.close()
                                if _docs_del:
                                    st.success(
                                        f"Deleted document `{del_doc_id}` "
                                        f"and {_chunks_del} associated chunk(s)."
                                    )
                                    st.rerun()
                                else:
                                    st.warning(f"No document found with ID `{del_doc_id}`.")
                            except Exception as _exc:
                                st.error(f"Delete failed: {_exc}")
                        else:
                            st.warning("Enter a document ID first.")


