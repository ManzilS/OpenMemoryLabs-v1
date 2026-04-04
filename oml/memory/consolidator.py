"""oml/memory/consolidator.py — Periodic memory consolidation for TEEG.

As a memory store grows, related AtomicNotes accumulate.  Consolidation
compresses clusters of related notes into a single, higher-confidence summary
note — analogous to how human long-term memory gradually integrates episodic
details into semantic knowledge.

Algorithm
---------
1. **Cluster detection** (keyword + graph connectivity)
   Build a note-similarity graph where two notes are connected if they share
   ≥ ``MIN_SHARED_KEYWORDS`` keywords OR are already linked by a direct TEEG
   edge.  Find connected components; any component with
   ≥ ``min_cluster_size`` notes is a cluster candidate.

2. **Cluster summarisation**
   For each cluster, create a summary AtomicNote whose:
     * content   — LLM-generated one-sentence summary (or heuristic join)
     * keywords  — union of all cluster keywords, capped at 8
     * tags      — union of all cluster tags, capped at 4
     * confidence — mean confidence of cluster notes (capped at 0.95)
     * context   — "Consolidated from N notes"
     * supersedes — ID of the oldest note in the cluster (chain anchor)

3. **Archival**
   All original cluster notes are archived (``active = False``).
   Directed edges are added from the summary → each archived note with
   ``relation = "consolidates"``.

4. **Persistence**
   ``store.save()`` is called after consolidation to persist all changes.

The net effect: N related notes (each costing ~87 LLM context tokens) are
replaced by 1 summary note (~87 tokens) — an (N−1)× reduction for that
cluster, while all information is still reachable via the graph.

Usage
-----
    consolidator = MemoryConsolidator(store, model_name="mock")
    result = consolidator.consolidate()
    print(result)
    # ConsolidationResult(clusters_found=3, notes_archived=11,
    #                     summaries_created=3, token_savings_est=870)

CLI / REST
----------
See ``oml teeg-consolidate`` and ``POST /teeg/consolidate``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Set

from oml.memory.atomic_note import AtomicNote
from oml.storage.teeg_store import TEEGStore

logger = logging.getLogger(__name__)

# ── tunables ─────────────────────────────────────────────────────────────────

MIN_SHARED_KEYWORDS: int = 2
"""Minimum number of shared keywords required to connect two notes in the
cluster graph (independent of existing TEEG edges)."""

TOKENS_PER_NOTE_FULL: int = 87
"""Estimated token cost of one FULL-tier TOON note in LLM context.  Used
to compute ``token_savings_est`` in :class:`ConsolidationResult`."""

MAX_CONSOLIDATED_KEYWORDS: int = 8
MAX_CONSOLIDATED_TAGS: int = 4


# ── result dataclass ─────────────────────────────────────────────────────────


@dataclass
class ConsolidationResult:
    """Summary of a consolidation run."""

    clusters_found: int
    """Number of note clusters identified (≥ min_cluster_size notes each)."""

    notes_archived: int
    """Number of original notes archived (replaced by summaries)."""

    summaries_created: int
    """Number of new summary AtomicNotes created and stored."""

    token_savings_est: int
    """Estimated LLM context tokens saved.

    Computed as ``(notes_archived − summaries_created) × TOKENS_PER_NOTE_FULL``.
    """

    skipped_small_clusters: int = 0
    """Clusters that were detected but below ``min_cluster_size``."""

    def __str__(self) -> str:
        return (
            f"Consolidation: {self.clusters_found} cluster(s) found | "
            f"{self.notes_archived} notes archived → "
            f"{self.summaries_created} summaries | "
            f"~{self.token_savings_est} tokens saved in context"
        )


# ── consolidator ─────────────────────────────────────────────────────────────


class MemoryConsolidator:
    """Cluster-and-compress TEEG notes to reduce LLM context token cost.

    Parameters
    ----------
    store:
        The :class:`~oml.storage.teeg_store.TEEGStore` to consolidate.
    model_name:
        LLM provider string for optional summary generation.  If ``"mock"``
        or if the LLM call fails, a heuristic summary is used instead.
    min_cluster_size:
        Minimum notes per cluster to trigger consolidation (default: 3).
    use_llm_summary:
        Whether to call the LLM to generate summary content.  Set to
        ``False`` to use heuristic summaries only (faster, no API cost).
    """

    def __init__(
        self,
        store: TEEGStore,
        model_name: str = "mock",
        min_cluster_size: int = 3,
        use_llm_summary: bool = True,
    ) -> None:
        self.store = store
        self.model_name = model_name
        self.min_cluster_size = min_cluster_size
        self.use_llm_summary = use_llm_summary and model_name != "mock"
        self._llm = None  # lazy-loaded

    # ── public API ────────────────────────────────────────────────────────────

    def consolidate(self, max_clusters: int = 10) -> ConsolidationResult:
        """Run one consolidation pass.

        Parameters
        ----------
        max_clusters:
            Maximum number of clusters to consolidate in this pass
            (safeguard against very large stores).

        Returns
        -------
        ConsolidationResult
            Statistics about what was compressed.
        """
        active = self.store.get_active()
        if not active:
            logger.info("[Consolidator] No active notes — nothing to consolidate")
            return ConsolidationResult(0, 0, 0, 0)

        clusters, small = self._find_clusters(active)
        logger.info(
            f"[Consolidator] Found {len(clusters)} cluster(s) "
            f"(≥{self.min_cluster_size} notes), {small} small cluster(s) skipped"
        )

        archived = 0
        summaries = 0

        for cluster in clusters[:max_clusters]:
            summary = self._summarise_cluster(cluster)
            self.store.add(summary)

            # Archive originals and link summary → each archived note
            for orig in cluster:
                self.store.archive(orig.note_id)
                self.store.add_edge(
                    summary.note_id,
                    orig.note_id,
                    relation="consolidates",
                    weight=0.8,
                )
                archived += 1

            summaries += 1
            logger.info(
                f"[Consolidator] Cluster of {len(cluster)} → summary {summary.note_id}"
            )

        # Persist all changes
        self.store.save()

        token_savings = max(0, archived - summaries) * TOKENS_PER_NOTE_FULL

        return ConsolidationResult(
            clusters_found=len(clusters),
            notes_archived=archived,
            summaries_created=summaries,
            token_savings_est=token_savings,
            skipped_small_clusters=small,
        )

    def dry_run(self) -> ConsolidationResult:
        """Analyse clusters without making any changes to the store.

        Returns a :class:`ConsolidationResult` with projected savings.
        """
        active = self.store.get_active()
        clusters, small = self._find_clusters(active)
        clusters_to_process = clusters[:10]
        archived = sum(len(c) for c in clusters_to_process)
        summaries = len(clusters_to_process)
        return ConsolidationResult(
            clusters_found=len(clusters),
            notes_archived=archived,
            summaries_created=summaries,
            token_savings_est=max(0, archived - summaries) * TOKENS_PER_NOTE_FULL,
            skipped_small_clusters=small,
        )

    # ── cluster detection ─────────────────────────────────────────────────────

    def _find_clusters(
        self, notes: List[AtomicNote]
    ) -> tuple[List[List[AtomicNote]], int]:
        """Return (large_clusters, n_small) using union-find on keyword+edge links."""
        id_to_note: Dict[str, AtomicNote] = {n.note_id: n for n in notes}
        note_ids = list(id_to_note.keys())

        # Union-Find structure
        parent: Dict[str, str] = {nid: nid for nid in note_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Connect notes via shared keywords
        # Index: keyword → set of note_ids that have it
        kw_index: Dict[str, Set[str]] = {}
        for note in notes:
            for kw in note.keywords:
                kw_lower = kw.lower()
                kw_index.setdefault(kw_lower, set()).add(note.note_id)

        # For each keyword shared by ≥ 2 notes, count cross-note occurrences
        pair_shared: Dict[tuple[str, str], int] = {}
        for nids in kw_index.values():
            nid_list = sorted(nids)
            for i in range(len(nid_list)):
                for j in range(i + 1, len(nid_list)):
                    pair = (nid_list[i], nid_list[j])
                    pair_shared[pair] = pair_shared.get(pair, 0) + 1

        for (a, b), count in pair_shared.items():
            if count >= MIN_SHARED_KEYWORDS:
                union(a, b)

        # Connect notes via existing TEEG graph edges
        graph = self.store.get_graph()
        for u, v in graph.edges():
            if u in id_to_note and v in id_to_note:
                union(u, v)

        # Group by root
        groups: Dict[str, List[str]] = {}
        for nid in note_ids:
            root = find(nid)
            groups.setdefault(root, []).append(nid)

        large_clusters: List[List[AtomicNote]] = []
        small_count = 0
        for members in groups.values():
            if len(members) >= self.min_cluster_size:
                large_clusters.append([id_to_note[nid] for nid in members])
            elif len(members) > 1:
                small_count += 1

        return large_clusters, small_count

    # ── summarisation ─────────────────────────────────────────────────────────

    def _summarise_cluster(self, notes: List[AtomicNote]) -> AtomicNote:
        """Create a compressed summary AtomicNote from a cluster."""
        # Aggregated metadata
        all_keywords = []
        all_tags = []
        for n in notes:
            all_keywords.extend(n.keywords)
            all_tags.extend(n.tags)

        # Deduplicate while preserving order (most frequent first)
        def dedup_freq(items: list, cap: int) -> list:
            seen: Dict[str, int] = {}
            for item in items:
                seen[item.lower()] = seen.get(item.lower(), 0) + 1
            return [k for k, _ in sorted(seen.items(), key=lambda x: -x[1])][:cap]

        merged_kw = dedup_freq(all_keywords, MAX_CONSOLIDATED_KEYWORDS)
        merged_tags = dedup_freq(all_tags, MAX_CONSOLIDATED_TAGS)
        mean_conf = min(0.95, sum(n.confidence for n in notes) / len(notes))

        # Anchor supersedes chain to the oldest note
        oldest = min(notes, key=lambda n: n.created_at)

        content = self._generate_content(notes)

        summary = AtomicNote(
            content=content,
            context=f"Consolidated from {len(notes)} notes",
            keywords=merged_kw,
            tags=merged_tags,
            confidence=round(mean_conf, 3),
            supersedes=oldest.note_id,
        )
        return summary

    def _generate_content(self, notes: List[AtomicNote]) -> str:
        """Generate a one-sentence summary of the cluster.

        Tries the LLM first; falls back to a heuristic join.
        """
        if self.use_llm_summary:
            try:
                return self._llm_summary(notes)
            except Exception as exc:
                logger.warning(f"[Consolidator] LLM summary failed: {exc}; using heuristic")

        return self._heuristic_summary(notes)

    def _heuristic_summary(self, notes: List[AtomicNote]) -> str:
        """Simple heuristic: pick the highest-confidence note's content."""
        best = max(notes, key=lambda n: n.confidence)
        others = len(notes) - 1
        suffix = f" (+ {others} related note{'s' if others > 1 else ''})"
        return best.content + suffix

    def _llm_summary(self, notes: List[AtomicNote]) -> str:
        """Call the LLM to generate a compressed one-sentence summary."""

        # Build TOON block of cluster contents
        toon_lines = []
        for note in notes:
            # Only include content + keywords (minimal for prompt efficiency)
            toon_lines.append(
                f"content: {note.content}\n"
                f"keywords: {'|'.join(note.keywords[:4])}"
            )
        toon_block = "\n---\n".join(toon_lines)

        prompt = f"""You are a memory consolidation assistant.
The following {len(notes)} related memory notes should be merged into a single, concise note.

NOTES (TOON format):
{toon_block}

Write ONE sentence that captures the most important shared fact across all notes.
Respond with ONLY the sentence, no labels or TOON keys.
CONSOLIDATED CONTENT:"""

        llm = self._get_llm()
        response = llm.generate(prompt).strip()
        # Sanity: take only first sentence
        for sep in (".", "!", "?", "\n"):
            if sep in response:
                response = response[: response.index(sep) + 1].strip()
                break
        return response if response else self._heuristic_summary(notes)

    def _get_llm(self):
        if self._llm is None:
            from oml.llm.factory import get_llm_client
            self._llm = get_llm_client(self.model_name)
        return self._llm
