from pathlib import Path
import subprocess

QUERIES = [
    "What did the narrator see on the ice?",
    "Describe the stranger rescued from the ice.",
    "What materials did Frankenstein use?",
    "How did the monster learn to speak?",
    "What was the monster's request to Victor in the mountains?",
]

LOG_DIR = Path("logs/rerank")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run():
    for q_idx, query in enumerate(QUERIES, start=1):
        for mode in ("on", "off"):
            log_path = LOG_DIR / f"q{q_idx}_rerank_{mode}.txt"
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
                "5",
                "--full",
                "--budget",
                "4000",
                "--show-tokens",
            ]
            if mode == "on":
                cmd.append("--rerank")
            else:
                cmd.append("--no-rerank")

            print("Running:", " ".join(cmd))
            with log_path.open("w", encoding="utf-8") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


if __name__ == "__main__":
    run()
