###############################################################################
# OpenMemoryLab — Dockerfile
#
# Single-stage build using python:3.12-slim.
# Image size is large (~4-6 GB) due to torch, transformers, faiss-cpu, and
# sentence-transformers — this is expected for an ML inference workload.
#
# Build:  docker build -t openmemorylab:1.0.0 .
# Run:    docker run --rm openmemorylab:1.0.0 oml --help
###############################################################################

FROM python:3.12-slim

# ── Python / pip hygiene ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential is required to compile native extensions (faiss-cpu, scipy).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer (cached unless pyproject.toml changes) ──────────────────
# Strategy:
#   1. Copy only pyproject.toml so Docker can cache this layer independently.
#   2. Create a minimal oml/__init__.py stub so hatchling can build a wheel
#      and pip can resolve all declared deps without needing the full source.
#   3. pip install . — installs ALL runtime deps (torch, transformers, etc.)
#      plus a stub version of the oml package.
#   4. COPY . . — overwrites the stub with the real source.
#   5. pip install --no-deps . — reinstalls oml from real source, registering
#      the correct entry point. Skips deps (already installed in step 3).
#
# This gives optimal Docker layer caching: the heavy ML dep layer (step 3) is
# only invalidated when pyproject.toml changes, not on every source edit.

COPY pyproject.toml ./

# Create a minimal stub package so hatchling can build for dep resolution
RUN mkdir -p oml && printf '__version__ = "1.0.0"\n' > oml/__init__.py

# Install all runtime deps declared in pyproject.toml (the slow, heavy step)
RUN pip install --no-cache-dir .

# ── Application layer ─────────────────────────────────────────────────────────
# Copy full source, overwriting the stub
COPY . .

# Reinstall the real oml package (entry points, all modules) — fast, no deps
RUN pip install --no-cache-dir --no-deps .

# ── Runtime setup ─────────────────────────────────────────────────────────────
# Pre-create writable directories so the non-root user can write to them
# (Docker volume mounts inherit container-side directory permissions).
RUN mkdir -p data artifacts reports \
    && useradd -m -u 1000 omluser \
    && chown -R omluser:omluser /app

USER omluser

# Streamlit web UI port
EXPOSE 8501

# Lightweight health check — just verifies the package is importable
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import oml" || exit 1

# Default: show CLI help. Override in docker-compose or docker run.
CMD ["oml", "--help"]
