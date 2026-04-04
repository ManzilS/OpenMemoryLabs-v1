import pytest

rdflib = pytest.importorskip("rdflib")

from unittest.mock import patch, MagicMock  # noqa: E402
from oml.eval.fact_checker import SemanticFactChecker  # noqa: E402


@pytest.fixture
def ground_truth_graph():
    g = rdflib.Graph()

    s = rdflib.URIRef("http://openmemorylab.org/entity/victor_frankenstein")
    p = rdflib.URIRef("http://openmemorylab.org/relation/created")
    o = rdflib.Literal("The Monster")
    g.add((s, p, o))

    s2 = rdflib.URIRef("http://openmemorylab.org/entity/the_monster")
    p2 = rdflib.URIRef("http://openmemorylab.org/relation/killed")
    o2 = rdflib.Literal("William")
    g.add((s2, p2, o2))

    return g


def test_fact_checker_valid_claims(ground_truth_graph):
    checker = SemanticFactChecker(use_llm=True)
    checker.graph = ground_truth_graph

    with patch('oml.eval.fact_checker.get_llm_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '```json\n[["Victor Frankenstein", "created", "The Monster"]]\n```'
        )
        mock_get_client.return_value = mock_client

        result = checker.verify("Victor created the monster.", "fake_model")

        assert result["score"] == 1.0
        assert result["total_claims"] == 1
        assert len(result["verified_claims"]) == 1
        assert len(result["unverified_claims"]) == 0


def test_fact_checker_hallucinated_claims(ground_truth_graph):
    checker = SemanticFactChecker(use_llm=True)
    checker.graph = ground_truth_graph

    with patch('oml.eval.fact_checker.get_llm_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '```json\n[["Robert Walton", "created", "The Monster"]]\n```'
        )
        mock_get_client.return_value = mock_client

        result = checker.verify("Robert Walton created the monster.", "fake_model")

        assert result["score"] == 0.0
        assert result["total_claims"] == 1
        assert len(result["verified_claims"]) == 0
        assert len(result["unverified_claims"]) == 1


def test_fact_checker_mixed_claims(ground_truth_graph):
    checker = SemanticFactChecker(use_llm=True)
    checker.graph = ground_truth_graph

    with patch('oml.eval.fact_checker.get_llm_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '```json\n'
            '[["The Monster", "killed", "William"],'
            ' ["The Monster", "killed", "Victor Frankenstein"]]\n```'
        )
        mock_get_client.return_value = mock_client

        result = checker.verify("The monster killed William and Victor.", "fake_model")

        assert result["score"] == 0.5
        assert result["total_claims"] == 2
        assert len(result["verified_claims"]) == 1
        assert len(result["unverified_claims"]) == 1
