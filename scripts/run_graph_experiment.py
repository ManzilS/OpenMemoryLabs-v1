from pathlib import Path
import subprocess

LOG_DIR = Path("logs/graph")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run():
    print("Step 1: Ingesting demo data with Knowledge Graph extraction (--graph)...")
    ingest_cmd = [
        "uv", "run", "--python", "3.11", "oml", "ingest", "--demo", "--graph"
    ]
    subprocess.run(ingest_cmd, check=False)
    
    print("\nStep 2: Querying with and without Knowledge Graph...")
    queries = [
        "What are cats like?",
        "Tell me about the stock market."
    ]
    
    for q_idx, query in enumerate(queries, start=1):
        for use_graph in [False, True]:
            mode = "on" if use_graph else "off"
            log_path = LOG_DIR / f"q{q_idx}_graph_{mode}.txt"
            cmd = [
                "uv",
                "run",
                "--python",
                "3.11",
                "oml",
                "query",
                query,
                "--alpha",
                "0.5",
                "--top-k",
                "3",
                "--budget",
                "1000",
                "--show-tokens",
            ]
            if use_graph:
                cmd.append("--graph")
                
            print(f"Running Q{q_idx} | Graph: {mode} -> {' '.join(cmd)}")
            with log_path.open("w", encoding="utf-8") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
                
    print(f"Sweep completed. Logs saved to {LOG_DIR}")

if __name__ == "__main__":
    run()
