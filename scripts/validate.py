"""Validation checks for OpenMemoryLab.

Run with the project venv Python:

    .venv\\Scripts\\python.exe scripts\\validate.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"[check] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_py312() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) != (3, 12):
        raise SystemExit(
            f"[error] Python 3.12 is required. Current interpreter: {sys.version.split()[0]}"
        )
    print(f"[ok] Python {sys.version.split()[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run project validation checks.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip full test suite and run only lint.",
    )
    args = parser.parse_args()

    ensure_py312()
    run([sys.executable, "-m", "ruff", "check", "oml", "--select", "F"])
    if not args.quick:
        run([sys.executable, "-m", "pytest", "-q"])
    print("[ok] All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
