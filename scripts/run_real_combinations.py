import os
import subprocess
from pathlib import Path
import time

LOG_DIR = Path("logs/real_eval")
LOG_DIR.mkdir(parents=True, exist_ok=True)

from oml.config import DEFAULT_MODEL
MODEL = DEFAULT_MODEL
DATA_FILE = "data/books/frankenstein.txt"

def run_command(cmd_list, log_path, description):
    print(f"\n[{description}] Running: {' '.join(cmd_list)}")
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"--- RUN: {description} ---\n")
        f.write(f"Command: {' '.join(cmd_list)}\n\n")
        
        env = os.environ.copy()
        env["OML_MODEL"] = MODEL
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', env=env)
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        process.wait()
        
    duration = time.time() - start_time
    print(f"[{description}] Finished in {duration:.2f} seconds.")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n--- TOOK {duration:.2f} SECONDS ---\n")

def check_file_exists():
    if not Path(DATA_FILE).exists():
        print(f"Error: {DATA_FILE} does not exist. Cannot run real test.")
        return False
    return True

def run():
    if not check_file_exists():
        return
        
    print(f"Starting Real Evaluation on {DATA_FILE} using {MODEL}")

    # Step 1: Ingest Frankenstein WITH graph enabled
    # We do not use the demo split since we want the real book.
    # We need to make sure we clear old artifacts to ensure a pristine test.
    print("\n--- PHASE 1: INGESTION ---")
    
    # We will use the main CLI ingest command. 
    # Since we want a clean slate, we could just ingest the single file.
    # The `oml ingest` command processes the `data/` directory by default. 
    # Let's clean the vector index and graph to ensure we only have Frankenstein if possible.
    # Note: openmemorylab ingest doesn't have a specific file flag, it reads `data/` or `data/demo_dataset/`.
    # Assuming data/books/frankenstein.txt is there, it will index it.
    
    ingest_cmd = [
        "uv", "run", "--python", "3.11", "oml", "ingest", 
        "--model", "mock", # Wait, extraction with actual model takes forever for a whole book.
                           # The user said "real llm (ollama:qwen3:4b) for all of the tests".
                           # But graph extraction on an entire book via local LLM will take hours.
                           # Let's comply and use the real model.
    ]
    # Actually, the user asked for testing combinations using the real model. 
    ingest_cmd = [
        ".venv\\Scripts\\oml.exe", "ingest", DATA_FILE,
        "--limit-chunks", "20",
        "--graph", "--model", MODEL
    ]
    
    log_file = LOG_DIR / "00_ingest.txt"
    # To prevent it from taking hours, let's limit ingestion to just the first few chunks if possible, 
    # or just run it and see. The user requested we use it. 
    run_command(ingest_cmd, log_file, "Ingest with Graph")

    # Step 2: Queries (Combinations)
    print("\n--- PHASE 2: COMBINATION QUERIES ---")
    queries = [
        "Who is Robert Walton?",
        "Why did Victor Frankenstein create the monster?",
        "Describe the relationship between Justine Moritz and the Frankenstein family."
    ]

    combinations = [
        {"name": "Baseline", "flags": []},
        {"name": "HyDE_Only", "flags": ["--hyde"]},
        {"name": "Graph_Only", "flags": ["--graph"]},
        {"name": "HyDE_and_Graph", "flags": ["--hyde", "--graph"]},
    ]

    for q_idx, query in enumerate(queries, 1):
        for combo in combinations:
            log_name = LOG_DIR / f"q{q_idx}_{combo['name']}.txt"
            
            cmd = [
                ".venv\\Scripts\\oml.exe", "query", 
                query, 
                "--show-prompt"
            ] + combo["flags"]
            
            run_command(cmd, log_name, f"Q{q_idx} - {combo['name']}")

    print("\nAll combinations tested. Check the logs/real_eval/ directory.")

if __name__ == "__main__":
    run()
