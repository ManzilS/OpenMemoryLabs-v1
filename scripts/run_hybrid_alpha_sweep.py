from pathlib import Path
import subprocess

QUERIES = [
    "Who is the narrator writing to?",
    "What is the goal of the expedition?",
    "Describe the stranger rescued from the ice.",
    "Why did Victor decide to create life?",
    "What materials did Frankenstein use?",
    "How did the monster learn to speak?",
    "Who was Justine Moritz and what happened to her?",
    "What was the monster's request to Victor in the mountains?",
    "Did Victor create a female companion?",
    "How did Elizabeth die?",
]

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
LOG_DIR = Path("logs/alpha_sweep")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run():
    for q_idx, query in enumerate(QUERIES, start=1):
        for alpha in ALPHAS:
            log_path = LOG_DIR / f"q{q_idx}_alpha_{alpha}.txt"
            cmd = [
                "uv",
                "run",
                "--python",
                "3.11",
                "oml",
                "query",
                query,
                "--alpha",
                str(alpha),
                "--top-k",
                "5",
                "--no-rerank",
                "--budget",
                "4000",
                "--show-tokens",
            ]
            print("Running:", " ".join(cmd))
            with log_path.open("w", encoding="utf-8") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


if __name__ == "__main__":
    run()
