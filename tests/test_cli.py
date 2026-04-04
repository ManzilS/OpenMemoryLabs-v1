import os
import pytest
from typer.testing import CliRunner
from oml.cli import app

runner = CliRunner()

@pytest.fixture
def tmp_env(tmp_path):
    """Sets up a temporary directory state for CLI tests."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    # Provide a tiny toy dataset file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc1.txt").write_text("The quick brown fox jumps over the lazy dog.")
    (data_dir / "doc2.txt").write_text("Artificial intelligence is transforming software engineering.")
    
    yield tmp_path
    
    os.chdir(original_cwd)

def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() != ""

def test_cli_ingest_basic_demo(tmp_env):
    """Test the built-in demo ingestion."""
    # Run the demo ingest flow
    result = runner.invoke(app, ["ingest", "--demo"])
    
    assert result.exit_code == 0
    
    # Check if artifacts were created
    assert (tmp_env / "artifacts" / "bm25.pkl").exists()
    assert (tmp_env / "artifacts" / "vector.index").exists()
    assert (tmp_env / "data" / "oml.db").exists()

def test_cli_query_after_ingest(tmp_env):
    """Test that querying works after ingesting data."""
    # 1. Ingest demo
    runner.invoke(app, ["ingest", "--demo"])
    
    # 2. Query
    result = runner.invoke(app, ["query", "dogs"])
    
    # It shouldn't crash
    assert result.exit_code == 0
    # It should find at least one result
    assert "Found" in result.stdout
    assert "results" in result.stdout

def test_cli_eval_fail_gracefully():
    # Assuming 'unknown-task' doesn't exist
    result = runner.invoke(app, ["eval", "unknown-task"])
    assert result.exit_code == 1
    assert "Task 'unknown-task' not found" in result.stdout or "Error running eval" in result.stdout
