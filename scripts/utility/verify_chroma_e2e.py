import os
import shutil
import subprocess
import sys

def verify_e2e_chroma():
    # Setup
    data_file = "data/chroma_e2e.txt"
    with open(data_file, "w") as f:
        f.write("This is a specific document about quantum entanglement for ChromaDB verification.")
        
    chroma_db = "data/chroma_db"
    if os.path.exists(chroma_db):
        shutil.rmtree(chroma_db)

    print("--- 1. Ingesting ---")
    cmd_ingest = [
        sys.executable, "-m", "oml.cli", "ingest", 
        data_file, 
        "--storage-type", "chroma", 
        "--no-summarize"
    ]
    subprocess.check_call(cmd_ingest)
    
    print("\n--- 2. Querying ---")
    cmd_query = [
        sys.executable, "-m", "oml.cli", "query",
        "quantum entanglement",
        "--storage-type", "chroma",
        "--top-k", "1",
        "--no-rerank" # Skip rerank for simple check
    ]
    
    result = subprocess.check_output(cmd_query, text=True)
    print(result)
    
    if "quantum entanglement" in result:
        print("SUCCESS: Found expected content.")
    else:
        print("FAILURE: Content not found.")
        sys.exit(1)

if __name__ == "__main__":
    verify_e2e_chroma()
