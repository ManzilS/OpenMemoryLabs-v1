from pathlib import Path
import subprocess

QUERIES = [
    "Why was Justine accused of the murder?",
    "Describe the process by which the creature learned about human history.",
    "What were Victor's reasons for destroying the female creature?",
    "How does the creature explain his turn to violence?",
    "What do the letters of Safie reveal about her character?"
]

LOG_DIR = Path("logs/hyde")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run():
    print("Starting HyDE evaluation sweep...")
    for q_idx, query in enumerate(QUERIES, start=1):
        for use_hyde in [False, True]:
            mode = "on" if use_hyde else "off"
            log_path = LOG_DIR / f"q{q_idx}_hyde_{mode}.txt"
            cmd = [
                "uv",
                "run",
                "--python",
                "3.11",
                "oml",
                "query",
                query,
                "--alpha",
                "0.5",  # We need non-zero alpha so vector search runs
                "--top-k",
                "5",
                # Don't rerank to isolate HyDE effect vs base Bi-Encoder
                "--no-rerank",
                "--full",
                "--budget",
                "4000",
                "--show-tokens",
            ]
            
            if use_hyde:
                cmd.append("--hyde")

            print(f"Running Q{q_idx} | HyDE: {mode} -> {' '.join(cmd)}")
            with log_path.open("w", encoding="utf-8") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
                
    print(f"Sweep completed. Logs saved to {LOG_DIR}")

if __name__ == "__main__":
    run()
