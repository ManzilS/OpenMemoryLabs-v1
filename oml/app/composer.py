"""oml/app/composer.py — Visual Technique Composer (v2)
=====================================================
LangChain-flow-inspired visual pipeline builder using a custom Streamlit
component (``declare_component``) for true drag-and-drop.

The HTML/CSS/JS lives in ``composer_component/index.html`` and communicates
bidirectionally with this Python module:

  JS → Python:  ``Streamlit.setComponentValue({action, pipeline, ...})``
  Python → JS:  component args  (nodeTypes, datasets, results, theme, …)

The execution engine (_run_pipeline / _exec_node) is unchanged from v1.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1.  DECLARE CUSTOM COMPONENT
# ═══════════════════════════════════════════════════════════════════

_COMPONENT_DIR = Path(__file__).parent / "composer_component"
_composer_component = components.declare_component(
    "composer_canvas", path=str(_COMPONENT_DIR)
)


# ═══════════════════════════════════════════════════════════════════
# 2.  CATALOGUE
# ═══════════════════════════════════════════════════════════════════

NODE_TYPES: dict[str, dict] = {
    "input": {
        "name": "Data Input",
        "icon": "\U0001f4dd",
        "color": "#3b82f6",
        "category": "data",
        "out_type": "text",
        "tip": "Feed raw text or a sample dataset into the pipeline.",
    },
    "llm-distiller": {
        "name": "LLM Distiller",
        "icon": "\U0001f9e0",
        "color": "#8b5cf6",
        "category": "ingest",
        "out_type": "note",
        "tip": "Raw text → TOON AtomicNote via LLM.",
    },
    "heuristic-distiller": {
        "name": "Heuristic Distiller",
        "icon": "\u26a1",
        "color": "#f59e0b",
        "category": "ingest",
        "out_type": "note",
        "tip": "Rule-based distillation, no LLM needed.",
    },
    "stage1-prescreen": {
        "name": "Pre-Screen",
        "icon": "\U0001f50d",
        "color": "#ef4444",
        "category": "evolution",
        "out_type": "verdict",
        "tip": "Fast YES/NO contradiction check (33 regex patterns).",
    },
    "stage2-judge": {
        "name": "Full Judge",
        "icon": "\u2696\ufe0f",
        "color": "#ec4899",
        "category": "evolution",
        "out_type": "relation",
        "tip": "LLM judge: CONTRADICTS / EXTENDS / SUPPORTS / UNRELATED.",
    },
    "confidence-engine": {
        "name": "Confidence",
        "icon": "\U0001f4ca",
        "color": "#14b8a6",
        "category": "evolution",
        "out_type": "note",
        "tip": "Bayesian confidence decay / boost from verdict.",
    },
    "belief-propagator": {
        "name": "Belief Propagator",
        "icon": "\U0001f30a",
        "color": "#0ea5e9",
        "category": "evolution",
        "out_type": "stats",
        "tip": "Propagates confidence changes through graph edges.",
    },
    "vector-seeder": {
        "name": "Vector Search",
        "icon": "\U0001f3af",
        "color": "#06b6d4",
        "category": "retrieval",
        "out_type": "seeds",
        "tip": "Finds seed notes for a query via vector similarity.",
    },
    "graph-walker": {
        "name": "Graph Walk",
        "icon": "\U0001f578\ufe0f",
        "color": "#22c55e",
        "category": "retrieval",
        "out_type": "context",
        "tip": "BFS traversal from seeds, collecting related notes.",
    },
    "answer-generator": {
        "name": "Answer Generator",
        "icon": "\U0001f4ac",
        "color": "#a855f7",
        "category": "generation",
        "out_type": "answer",
        "tip": "Generates an answer from context via LLM.",
    },
    "output": {
        "name": "Output",
        "icon": "\U0001f4e4",
        "color": "#64748b",
        "category": "data",
        "out_type": "any",
        "tip": "Displays the final result.",
    },
}

SAMPLE_DATASETS: dict[str, list[str]] = {
    "Frankenstein": [
        "Victor Frankenstein created the creature using dead body parts and electricity.",
        "The creature taught himself to read by watching the De Lacey family.",
        "Mary Shelley wrote Frankenstein when she was only 18 years old.",
        "The creature asked Victor to make him a companion, but Victor refused.",
        "The novel was published anonymously in 1818.",
    ],
    "Solar System": [
        "Jupiter is the largest planet with a mass 318 times that of Earth.",
        "Saturn's rings are made mostly of ice particles and rocky debris.",
        "Mars has the tallest volcano: Olympus Mons at 21.9 km high.",
        "Venus rotates backwards \u2014 the sun rises in the west there.",
        "Neptune's winds can reach speeds of 2,100 km/h.",
    ],
    "World History": [
        "The Great Wall of China was started around 700 BCE.",
        "Cleopatra lived closer to the Moon landing than to the Great Pyramid.",
        "The printing press was invented by Gutenberg around 1440.",
        "The French Revolution began in 1789 at the Bastille.",
        "The Wright Brothers' first flight lasted only 12 seconds.",
    ],
    "Contradictions (test)": [
        "The capital of France is Paris.",
        "The capital of France is Lyon.",
        "Water boils at 100\u00b0C at sea level.",
        "Water boils at 80\u00b0C at sea level.",
        "The Earth orbits the Sun.",
    ],
}

RECIPES: dict[str, dict] = {
    "Quick Distill": {
        "desc": "Distill text into a note \u2014 no LLM needed.",
        "nodes": ["input", "heuristic-distiller", "output"],
    },
    "LLM Distill": {
        "desc": "Use an LLM to create a structured TOON note.",
        "nodes": ["input", "llm-distiller", "output"],
    },
    "Evolution Chain": {
        "desc": "Full write pipeline: distill → screen → judge → confidence.",
        "nodes": [
            "input", "llm-distiller", "stage1-prescreen",
            "stage2-judge", "confidence-engine", "output",
        ],
    },
    "Retrieval + Answer": {
        "desc": "Read pipeline: search → walk → answer.",
        "nodes": [
            "input", "vector-seeder", "graph-walker",
            "answer-generator", "output",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════
# 3.  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _trunc(s: str, n: int = 50) -> str:
    return s[:n] + ("\u2026" if len(s) > n else "")


def _preview(data: Any, limit: int = 300) -> str:
    if data is None:
        return "\u2014"
    if isinstance(data, str):
        return _trunc(data, limit)
    if hasattr(data, "content"):  # AtomicNote
        kw = ", ".join(data.keywords[:5]) if getattr(data, "keywords", None) else ""
        return f"AtomicNote: {data.content[:120]}\n  keywords: {kw}\n  confidence: {data.confidence}"
    if isinstance(data, dict):
        return json.dumps(data, indent=2, default=str)[:limit]
    if isinstance(data, list):
        items = [_preview(x, 80) for x in data[:5]]
        return f"[{len(data)} items]\n" + "\n".join(f"  - {x}" for x in items)
    return str(data)[:limit]


def _to_text(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if hasattr(data, "content"):
        return data.content
    if isinstance(data, dict):
        return json.dumps(data, indent=2, default=str)
    if isinstance(data, list):
        parts = []
        for x in data:
            parts.append(x.content if hasattr(x, "content") else str(x))
        return "\n".join(parts)
    return str(data)


def _to_note(data: Any):
    """Coerce any data to an AtomicNote (heuristic fallback)."""
    if hasattr(data, "content"):
        return data
    from oml.memory.techniques.heuristic_distiller import HeuristicDistiller
    return HeuristicDistiller.distil(_to_text(data))


def _get_llm(model: str):
    if not model:
        raise ValueError("No LLM model configured. Set a model in the sidebar.")
    from oml.llm.factory import get_llm_client
    return get_llm_client(model)


# ═══════════════════════════════════════════════════════════════════
# 4.  EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════

def _exec_node(ntype: str, data: Any, *, model: str, query: str, settings: dict | None = None, **kw) -> Any:
    """Execute a single pipeline node.

    ``settings`` is a dict of per-node user overrides (prompt, temperature,
    max_tokens, etc.) coming from the gear-icon settings panel in the UI.
    """
    settings = settings or {}

    if ntype == "input":
        text = kw.get("dataset_text", "")
        if not text:
            raise ValueError("No input text. Select a sample dataset or type custom text.")
        return text

    if ntype == "heuristic-distiller":
        from oml.memory.techniques.heuristic_distiller import HeuristicDistiller
        return HeuristicDistiller.distil(_to_text(data))

    if ntype == "llm-distiller":
        from oml.memory.techniques.llm_distiller import LLMDistiller
        llm = _get_llm(model)
        text = _to_text(data)
        custom_prompt = settings.get("prompt", "").strip()
        if custom_prompt:
            # User provided a custom prompt — call LLM directly, then parse
            full_prompt = f"{custom_prompt}\n\nText: {text[:2000]}"
            response = llm.generate(full_prompt)
            return LLMDistiller._parse_distil_response(response, text, "")
        return LLMDistiller(llm).distil(text)

    if ntype == "stage1-prescreen":
        from oml.memory.techniques.stage1_prescreen import Stage1PreScreen
        note = _to_note(data)
        scr = Stage1PreScreen()
        prompt = scr.build_stage1_prompt(note, note)
        verdict = scr.parse_stage1_verdict("YES")  # demo self-comparison
        return {
            "prompt_sent": prompt,
            "verdict": verdict,
            "note_id": getattr(note, "note_id", "?"),
            "threshold": settings.get("threshold", 0.5),
            "explanation": "Self-comparison demo \u2014 connect to a store for real screening.",
        }

    if ntype == "stage2-judge":
        from oml.memory.techniques.stage2_judge import Stage2Judge
        note = _to_note(data)
        judge = Stage2Judge()
        prompt = judge.build_judge_prompt(note, note)
        return {
            "prompt_sent": prompt,
            "relation": "SUPPORTS",
            "reason": "self-comparison demo",
            "temperature": settings.get("temperature", 0.3),
            "explanation": "Self-comparison demo \u2014 connect an LLM for real judging.",
        }

    if ntype == "confidence-engine":
        note = _to_note(data)
        decay = settings.get("decay_factor", 0.1)
        boost = settings.get("boost_factor", 0.2)
        return {
            "note_content": note.content[:100],
            "original_confidence": note.confidence,
            "decay_factor": decay,
            "boost_factor": boost,
            "explanation": "Would apply Bayesian decay/boost based on verdict. "
                           "Needs a live TEEGStore + a judge verdict.",
        }

    if ntype == "belief-propagator":
        return {
            "input_summary": _trunc(_to_text(data), 100),
            "max_hops": settings.get("max_hops", 3),
            "weight_decay": settings.get("weight_decay", 0.5),
            "explanation": "Would propagate confidence changes to graph neighbours. "
                           "Needs a populated TEEGStore.",
        }

    if ntype == "vector-seeder":
        q = query or _to_text(data)
        if not q:
            raise ValueError("No query text. Provide a question.")
        return {
            "query": q,
            "top_k": settings.get("top_k", 5),
            "threshold": settings.get("threshold", 0.3),
            "explanation": "Would search the TEEG vector index for seed notes. "
                           "Ingest data via the TEEG tab first.",
        }

    if ntype == "graph-walker":
        return {
            "input_summary": _trunc(str(data), 100),
            "max_hops": settings.get("max_hops", 3),
            "min_confidence": settings.get("min_confidence", 0.3),
            "explanation": "Would BFS-walk the knowledge graph from seed notes. "
                           "Needs seeds from a Vector Seeder with a populated store.",
        }

    if ntype == "answer-generator":
        from oml.memory.techniques.answer_generator import AnswerGenerator
        ctx = _to_text(data) if data else "No context available."
        q = query or "What do you know?"
        llm = _get_llm(model)
        custom_prompt = settings.get("prompt", "").strip()
        if custom_prompt:
            # User provided a custom answer prompt — call LLM directly
            full_prompt = f"{custom_prompt}\n\nContext: {ctx}\n\nQuestion: {q}\n\nAnswer:"
            return llm.generate(full_prompt)
        return AnswerGenerator(llm).generate(q, ctx)

    if ntype == "output":
        return data

    raise ValueError(f"Unknown node type: {ntype}")


def _run_pipeline(
    nodes: list[dict],
    *,
    model: str,
    dataset_text: str,
    query_text: str,
    node_settings: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Execute every node in sequence, capturing I/O."""
    results: dict[str, dict] = {}
    current = None
    node_settings = node_settings or {}

    for node in nodes:
        t0 = time.perf_counter()
        settings = node_settings.get(node["id"], {})
        try:
            out = _exec_node(
                node["type"], current,
                model=model,
                query=query_text,
                dataset_text=dataset_text,
                settings=settings,
            )
            ms = (time.perf_counter() - t0) * 1000
            results[node["id"]] = {
                "status": "success",
                "in_data": current,
                "out_data": out,
                "in_preview": _preview(current),
                "out_preview": _preview(out),
                "ms": ms,
            }
            current = out
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            results[node["id"]] = {
                "status": "error",
                "in_data": current,
                "out_data": None,
                "in_preview": _preview(current),
                "out_preview": f"ERROR: {exc}",
                "error": str(exc),
                "ms": ms,
            }
            # keep going — next node gets None
            current = None

    return results


# ═══════════════════════════════════════════════════════════════════
# 5.  PERSISTENCE
# ═══════════════════════════════════════════════════════════════════

def _pipelines_path(artifacts: str) -> Path:
    return Path(artifacts) / "composer_pipelines.json"


def _load_saved(artifacts: str) -> dict:
    fp = _pipelines_path(artifacts)
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_pipeline(artifacts: str, name: str, nodes: list[dict]) -> None:
    saved = _load_saved(artifacts)
    saved[name] = {"nodes": [n["type"] for n in nodes]}
    fp = _pipelines_path(artifacts)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(saved, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════
# 6.  MAIN RENDER — declare_component bridge
# ═══════════════════════════════════════════════════════════════════

def _serialisable_results(results: dict[str, dict]) -> dict[str, dict]:
    """Strip non-JSON-serialisable objects from results before passing to JS."""
    out: dict[str, dict] = {}
    for nid, r in results.items():
        out[nid] = {
            "status": r["status"],
            "in_preview": r.get("in_preview", "\u2014"),
            "out_preview": r.get("out_preview", "\u2014"),
            "ms": r.get("ms", 0),
        }
        if r.get("error"):
            out[nid]["error"] = r["error"]
    return out


def render_composer(model: str, artifacts: str, theme: str) -> None:
    """Full composer UI — call from the Techniques tab.

    The custom Streamlit component handles all visual rendering (palette,
    workspace, inspector).  This function:
      1. Passes catalogue / dataset / result data *into* the component.
      2. Reads the component's return value (action + pipeline state).
      3. On ``"run"`` action, executes the pipeline and stores results.
      4. On ``"update"`` action, syncs the pipeline to session_state.
    """
    # ── session_state defaults ────────────────────────────────────
    if "comp_pipeline" not in st.session_state:
        st.session_state.comp_pipeline: list[dict] = []
    if "comp_results" not in st.session_state:
        st.session_state.comp_results: dict[str, dict] = {}
    if "comp_run_count" not in st.session_state:
        st.session_state.comp_run_count = 0

    pipeline = st.session_state.comp_pipeline
    results = st.session_state.comp_results

    # ── Recipe quick-start (only when pipeline is empty) ──────────
    if not pipeline:
        st.markdown("##### Quick Start — load a recipe")
        rcols = st.columns(len(RECIPES))
        for col, (rkey, rdata) in zip(rcols, RECIPES.items()):
            with col:
                if st.button(
                    rkey,
                    key=f"recipe_{rkey}",
                    help=rdata["desc"],
                    width="stretch",
                ):
                    st.session_state.comp_pipeline = [
                        {"id": uuid.uuid4().hex[:8], "type": t}
                        for t in rdata["nodes"]
                    ]
                    st.session_state.comp_results = {}

                    st.rerun()
                st.caption(rdata["desc"])

    # ── Render the custom component ──────────────────────────────
    comp_value = _composer_component(
        nodeTypes=NODE_TYPES,
        datasets=SAMPLE_DATASETS,
        pipeline=pipeline,
        results=_serialisable_results(results),
        theme=theme.lower(),
        initDs="Solar System",
        initDsIdx=0,
        initQuery="",
        initCustom="",
        key="composer_v2",
        height=620,
    )

    # ── Handle component return value ────────────────────────────
    if comp_value is not None and isinstance(comp_value, dict):
        action = comp_value.get("action")
        js_pipeline = comp_value.get("pipeline", [])

        if action == "update":
            # Sync JS pipeline state → session_state.  Only rerun if
            # the pipeline actually changed (prevents infinite loop
            # since comp_value persists across reruns).
            old = st.session_state.comp_pipeline
            if [n.get("id") for n in js_pipeline] != [n.get("id") for n in old]:
                st.session_state.comp_pipeline = js_pipeline
                st.session_state.comp_results = {}
                st.rerun()

        elif action == "run":
            # Sync pipeline first
            st.session_state.comp_pipeline = js_pipeline
            dataset_text = comp_value.get("datasetText", "")
            query_text = comp_value.get("query", "")
            node_settings = comp_value.get("nodeSettings", {})

            # Execute
            st.session_state.comp_results = _run_pipeline(
                js_pipeline,
                model=model,
                dataset_text=dataset_text,
                query_text=query_text,
                node_settings=node_settings,
            )
            st.session_state.comp_run_count += 1
            st.rerun()

    # ── Results summary (below the component) ────────────────────
    if results:
        st.markdown("---")
        st.markdown("##### Results Summary")
        total_ms = sum(r.get("ms", 0) for r in results.values())
        n_ok = sum(1 for r in results.values() if r["status"] == "success")
        n_err = sum(1 for r in results.values() if r["status"] == "error")

        m1, m2, m3 = st.columns(3)
        m1.metric("Nodes executed", f"{len(results)}/{len(pipeline)}")
        m2.metric("Passed / Failed", f"{n_ok} / {n_err}")
        m3.metric("Total time", f"{total_ms:.0f}ms")

        # Final output
        if pipeline:
            last_node = pipeline[-1]
            last_res = results.get(last_node["id"])
            if last_res and last_res["status"] == "success":
                st.markdown("##### Final Output")
                out = last_res.get("out_data")
                if isinstance(out, str):
                    st.success(out)
                elif isinstance(out, dict):
                    st.json(out)
                else:
                    st.code(_preview(out, 1000), language=None)

    # ── Save / Load (below results) ──────────────────────────────
    if pipeline:
        st.markdown("---")
        s1, s2, s3 = st.columns([3, 1, 1])
        with s1:
            save_name = st.text_input(
                "Pipeline name",
                placeholder="my-pipeline",
                key="comp_save_name",
                label_visibility="collapsed",
            )
        with s2:
            if st.button("\U0001f4be Save", key="comp_save", width="stretch"):
                if save_name:
                    _save_pipeline(artifacts, save_name.strip(), pipeline)
                    st.success(f"Saved **{save_name.strip()}**")
                else:
                    st.warning("Enter a name first.")
        with s3:
            if st.button("\U0001f5d1\ufe0f Clear", key="comp_clear_py", width="stretch"):
                st.session_state.comp_pipeline = []
                st.session_state.comp_results = {}
                st.rerun()

    # Load saved pipelines
    saved = _load_saved(artifacts)
    if saved:
        load_cols = st.columns([3, 1])
        with load_cols[0]:
            load_choice = st.selectbox(
                "Load saved pipeline",
                [""] + list(saved.keys()),
                key="comp_load_sel",
            )
        with load_cols[1]:
            if st.button("Load", key="comp_load_btn") and load_choice:
                st.session_state.comp_pipeline = [
                    {"id": uuid.uuid4().hex[:8], "type": t}
                    for t in saved[load_choice]["nodes"]
                ]
                st.session_state.comp_results = {}
                st.rerun()
