"""prepare_wiki_docs.py
Extract target Wikipedia articles from the chunked wiki dump and ingest them
into OML's artifact store so that the retrieval-dependent eval tasks have a
real-world corpus to query against.

Usage:
    python scripts/prepare_wiki_docs.py

What it does:
  1. Scans every wiki chunk file under oml/eval/datasets/wiki/{1of2,2of2}/
  2. Splits each chunk file on blank lines to isolate article blocks
  3. Matches article blocks against a set of target topic keywords
  4. Saves each matched article as an individual .txt file in data/docs/
  5. Runs  `python -m oml ingest data/docs/`  to populate artifacts/

The topics were chosen to be diverse (history, science, religion, sport, myth)
and to back all eight eval tasks that now use Wikipedia-based questions.
"""

import subprocess
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
WIKI_DIRS = [
    BASE / "oml" / "eval" / "datasets" / "wiki" / "1of2",
    BASE / "oml" / "eval" / "datasets" / "wiki" / "2of2",
]
DOCS_DIR = BASE / "data" / "docs"

# ── Target articles ───────────────────────────────────────────────────────────
# (output_stem, list_of_keywords_to_match_in_first_1000_chars_of_article_block)
# Keywords are matched case-insensitively against the first 1 000 chars of each
# article block so we capture both the title and opening paragraphs.
TARGETS = [
    ("alan_turing",          ["alan turing", "turing"]),
    ("bletchley_park",       ["bletchley park", "bletchley"]),
    ("blue_mountains_aus",   ["blue mountains", "blaxland", "wentworth"]),
    ("birmingham_campaign",  ["birmingham campaign", "birmingham, alabama"]),
    ("canelo_alvarez",       ["canelo", "álvarez", "alvarez"]),
    ("canadian_forces",      ["canadian forces", "marcom", "cansofcom"]),
    ("islamic_calendar",     ["islamic calendar", "hijri calendar", "hijri"]),
    ("ovambo_people",        ["ovambo"]),
    ("empusa",               ["empusa"]),
    ("otto_klemperer",       ["klemperer", "otto klemperer"]),
    ("aum_shinrikyo",        ["aum shinrikyo", "shoko asahara"]),
    ("town_privileges",      ["town privileges", "lübeck law", "lubeck law", "magdeburg rights"]),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def iter_chunk_files():
    """Yield all wiki chunk file paths in stable sorted order."""
    files = []
    for wiki_dir in WIKI_DIRS:
        if wiki_dir.exists():
            files.extend(sorted(wiki_dir.iterdir()))
        else:
            print(f"  [WARN] Directory not found: {wiki_dir}")
    return files


def read_articles(filepath: Path) -> list[str]:
    """Read a wiki chunk file and return non-trivial article blocks."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [WARN] Could not read {filepath.name}: {exc}")
        return []
    # Articles are separated by one or more blank lines
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    # Keep only blocks long enough to be real articles (>= 300 chars)
    return [b for b in blocks if len(b) >= 300]


def matches_any_keyword(block: str, keywords: list[str]) -> bool:
    head = block[:1000].lower()
    return any(kw.lower() in head for kw in keywords)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Wikipedia Article Extractor for OML Eval")
    print("=" * 60)

    chunk_files = iter_chunk_files()
    print(f"\nFound {len(chunk_files)} wiki chunk files to scan.\n")

    # Track which targets have already been found
    found: dict[str, str] = {}          # stem -> article text
    remaining = dict(TARGETS)           # stem -> keywords (still looking)

    for chunk_file in chunk_files:
        if not remaining:
            break  # all targets satisfied

        articles = read_articles(chunk_file)
        if not articles:
            continue

        matched_in_file = []
        for stem, keywords in list(remaining.items()):
            for article in articles:
                if matches_any_keyword(article, keywords):
                    found[stem] = article
                    del remaining[stem]
                    matched_in_file.append(f"{stem} ({len(article):,} chars)")
                    break

        if matched_in_file:
            print(f"  {chunk_file.name}: found {', '.join(matched_in_file)}")

    # ── Save found articles ───────────────────────────────────────────────────
    print(f"\nSaving {len(found)} articles to {DOCS_DIR} …")
    for stem, content in found.items():
        out = DOCS_DIR / f"{stem}.txt"
        out.write_text(content, encoding="utf-8")
        print(f"  Saved: {out.name}  ({len(content):,} chars)")

    # ── Report missing targets ────────────────────────────────────────────────
    if remaining:
        print(f"\n[WARN] Could not find articles for: {list(remaining.keys())}")
        print("       Creating minimal placeholder files so ingest still runs.")
        for stem, keywords in remaining.items():
            out = DOCS_DIR / f"{stem}.txt"
            placeholder = (
                f"{stem.replace('_', ' ').title()}\n\n"
                f"This article covers {stem.replace('_', ' ')}. "
                f"Keywords: {', '.join(keywords)}.\n"
            )
            out.write_text(placeholder, encoding="utf-8")
            print(f"  Placeholder: {out.name}")

    # ── Run OML ingest ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Running: python -m oml ingest data/docs/")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "oml", "ingest", str(DOCS_DIR)],
        capture_output=False,   # stream output to console
        cwd=str(BASE),
    )

    if result.returncode == 0:
        print("\n[OK] Ingestion completed successfully.")
    else:
        print(f"\n[ERROR] Ingestion exited with code {result.returncode}.")
        sys.exit(result.returncode)

    print("\nDone!  Run `python scripts/benchmark_models.py` to start the eval.")


if __name__ == "__main__":
    main()
