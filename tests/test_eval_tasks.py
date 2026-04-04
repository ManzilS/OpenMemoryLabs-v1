"""
Tests for the evaluation task framework and individual tasks.

All tests use the MockLLM so no API keys or running LLM servers are required.
Tasks that need a live retrieval index (cost_latency, oml_vs_rag, retrieval_precision)
are tested for graceful error handling when no artifacts exist.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from oml.llm.mock import MockLLM
from oml.eval.run import run_task, get_task, _TASK_REGISTRY
import oml.eval.tasks  # ensure all tasks are registered
import oml.eval.ablations  # ensure ablations task is registered


# ── helpers ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    return MockLLM()


# ── task registry ──────────────────────────────────────────────────────────

def test_all_expected_tasks_are_registered():
    expected = {
        "lost-in-middle",
        "faithfulness",
        "cost_latency",
        "oml_vs_rag",
        "ablations",
        "retrieval_precision",
    }
    registered = set(_TASK_REGISTRY.keys())
    assert expected.issubset(registered), (
        f"Missing tasks: {expected - registered}"
    )


def test_get_task_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        get_task("nonexistent_task")


# ── lost-in-middle ─────────────────────────────────────────────────────────

class TestLostInMiddle:
    def test_returns_eval_result(self, mock_model):
        from oml.eval.tasks.lost_in_middle import LostInMiddleTask
        task = LostInMiddleTask()
        result = task.run(mock_model, {})
        assert result.task_name == "lost-in-middle"
        assert 0.0 <= result.score <= 1.0

    def test_perfect_score_when_needle_always_found(self, mock_model):
        from oml.eval.tasks.lost_in_middle import LostInMiddleTask
        task = LostInMiddleTask()
        # MockLLM detects "Blaxland" + "1813" in the prompt and echoes "1813",
        # so the task's needle check passes at all three positions.
        result = task.run(mock_model, {"context_length": 200})
        assert result.score == 1.0

    def test_details_contain_all_positions(self, mock_model):
        from oml.eval.tasks.lost_in_middle import LostInMiddleTask
        task = LostInMiddleTask()
        result = task.run(mock_model, {})
        assert "pos_0" in result.details
        assert "pos_50" in result.details
        assert "pos_100" in result.details

    def test_via_run_task(self):
        result = run_task("lost-in-middle", model_name="mock", config={"context_length": 100})
        assert result.task_name == "lost-in-middle"


# ── faithfulness ───────────────────────────────────────────────────────────

class TestFaithfulness:
    def test_returns_eval_result(self, mock_model):
        from oml.eval.tasks.faithfulness import FaithfulnessTask

        # Patch the judge model so it always returns VERDICT: YES
        with patch.object(mock_model, "generate", return_value="VERDICT: YES"):
            task = FaithfulnessTask()
            result = task.run(mock_model, {})

        assert result.task_name == "faithfulness"
        assert 0.0 <= result.score <= 1.0

    def test_all_yes_verdicts_give_score_based_on_expected(self, mock_model):
        from oml.eval.tasks.faithfulness import FaithfulnessTask

        # When judge always says YES:
        # example_0: expected YES  → correct  (+1)
        # example_1: expected NO   → incorrect (0)
        # example_2: expected NO   → incorrect (0)
        # score = 1/3
        with patch.object(mock_model, "generate", return_value="Some reasoning.\nVERDICT: YES"):
            task = FaithfulnessTask()
            result = task.run(mock_model, {})

        assert abs(result.score - 1 / 3) < 0.01

    def test_all_no_verdicts_give_score_based_on_expected(self, mock_model):
        from oml.eval.tasks.faithfulness import FaithfulnessTask

        # When judge always says NO:
        # example_0: expected YES  → incorrect (0)
        # example_1: expected NO   → correct   (+1)
        # example_2: expected NO   → correct   (+1)
        # score = 2/3
        with patch.object(mock_model, "generate", return_value="No support.\nVERDICT: NO"):
            task = FaithfulnessTask()
            result = task.run(mock_model, {})

        assert abs(result.score - 2 / 3) < 0.01

    def test_details_have_all_examples(self, mock_model):
        from oml.eval.tasks.faithfulness import FaithfulnessTask

        with patch.object(mock_model, "generate", return_value="VERDICT: YES"):
            task = FaithfulnessTask()
            result = task.run(mock_model, {})

        assert "example_0" in result.details
        assert "example_1" in result.details
        assert "example_2" in result.details


# ── retrieval_precision ────────────────────────────────────────────────────

class TestRetrievalPrecision:
    def test_graceful_no_artifacts(self, mock_model, tmp_path, monkeypatch):
        """When no artifacts/ exists the task returns score=0 with an error detail."""
        from oml.eval.tasks.retrieval_precision import RetrievalPrecisionTask

        monkeypatch.chdir(tmp_path)  # move to empty temp dir – no artifacts/
        task = RetrievalPrecisionTask()
        result = task.run(mock_model, {})

        assert result.task_name == "retrieval_precision"
        assert result.score == 0.0
        assert "error" in result.details

    def test_with_mock_retriever(self, mock_model, tmp_path, monkeypatch):
        """With a mocked retriever returning no results, precision should be 0."""
        from oml.eval.tasks.retrieval_precision import RetrievalPrecisionTask
        from oml.retrieval.base import SearchResult

        monkeypatch.chdir(tmp_path)
        (tmp_path / "artifacts").mkdir()

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        mock_storage = MagicMock()
        mock_storage.get_chunks_by_ids.return_value = []

        with (
            patch("oml.eval.tasks.retrieval_precision.HybridRetriever", return_value=mock_retriever),
            patch("oml.eval.tasks.retrieval_precision.get_storage", return_value=mock_storage),
        ):
            task = RetrievalPrecisionTask()
            result = task.run(mock_model, {"top_k": 3})

        assert result.task_name == "retrieval_precision"
        assert result.score == 0.0
        assert result.details["top_k"] == 3

    def test_perfect_precision_when_all_chunks_are_hits(self, mock_model, tmp_path, monkeypatch):
        """When every returned chunk contains a relevant keyword, precision = 1.0."""
        from oml.eval.tasks.retrieval_precision import RetrievalPrecisionTask, LABELED_DATASET
        from oml.retrieval.base import SearchResult

        monkeypatch.chdir(tmp_path)
        (tmp_path / "artifacts").mkdir()

        # Return one search result per query
        def fake_search(query, top_k=5, alpha=0.5):
            return [SearchResult(chunk_id="c1", score=0.9, source="test", details={})]

        # Build a chunk text that contains keywords from ALL dataset entries
        all_keywords = " ".join(
            kw for item in LABELED_DATASET for kw in item["keywords"]
        )

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "c1"
        mock_chunk.chunk_text = all_keywords  # contains every relevant keyword

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = fake_search

        mock_storage = MagicMock()
        mock_storage.get_chunks_by_ids.return_value = [mock_chunk]

        with (
            patch("oml.eval.tasks.retrieval_precision.HybridRetriever", return_value=mock_retriever),
            patch("oml.eval.tasks.retrieval_precision.get_storage", return_value=mock_storage),
        ):
            task = RetrievalPrecisionTask()
            result = task.run(mock_model, {"top_k": 1})

        assert result.score == 1.0
