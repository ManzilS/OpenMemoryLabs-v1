"""oml/storage/factory.py — Storage backend factory.

Routes a storage-type string to the correct backend implementation.  All
backends implement the ``BaseStorage`` protocol, so the retrieval pipeline
is storage-agnostic and can be hot-swapped without touching business logic.

Supported backends
------------------
``sqlite``   Persistent SQLite database + FAISS vector index.  Default path
             comes from ``oml.config.DEFAULT_SQLITE_PATH`` (``data/oml.db``).
             Override with ``config={"db_path": "/your/path.db"}``.
             Best for: portability, reproducibility, local development.

``memory``   Fully in-memory store — no files written to disk.  Zero setup;
             data is lost when the process exits.
             Best for: unit/integration tests, interactive demos (``--demo``).

``lancedb``  LanceDB columnar vector store.  Default path from
             ``oml.config.DEFAULT_LANCEDB_PATH`` (``data/lancedb/``).
             Override with ``config={"persist_path": "/your/dir"}``.
             Requires ``pip install lancedb``.
             Best for: large corpora, faster ANN search at scale.

The active backend is selected via the ``OML_STORAGE`` environment variable
(or ``oml.yaml`` → ``storage:`` key).  The default is ``sqlite``.
"""

from typing import Dict, Any, Optional
from oml.storage.base import BaseStorage


def get_storage(storage_type: str, config: Optional[Dict[str, Any]] = None) -> BaseStorage:
    """Return a ``BaseStorage`` instance for the given backend type.

    Args:
        storage_type: One of ``"sqlite"``, ``"memory"``, or ``"lancedb"``.
            See the module docstring for a full description of each backend.
        config: Optional backend-specific overrides.  Recognised keys:

            - ``"db_path"`` (*sqlite* only) — path to the SQLite file.
            - ``"persist_path"`` (*lancedb* only) — directory for the
              LanceDB store.

    Returns:
        A concrete ``BaseStorage`` subclass ready to call ``.add()`` /
        ``.search()`` / ``.get_all()`` etc.

    Raises:
        ValueError: If *storage_type* is not one of the supported values.
        ImportError: If the backend's optional dependency is not installed
            (e.g. ``lancedb`` package for the LanceDB backend).
    """
    config = config or {}
    
    if storage_type == "sqlite":
        from oml.storage.sqlite import SQLiteStorage
        from oml.config import DEFAULT_SQLITE_PATH
        db_path = config.get("db_path", DEFAULT_SQLITE_PATH)
        return SQLiteStorage(db_path)
    
    if storage_type == "memory":
        from oml.storage.memory import MemoryStorage
        return MemoryStorage()

    if storage_type == "lancedb":
        from oml.storage.lancedb import LanceDBStorage
        from oml.config import DEFAULT_LANCEDB_PATH
        persist_path = config.get("persist_path", DEFAULT_LANCEDB_PATH)
        return LanceDBStorage(persist_path=persist_path)
        
    raise ValueError(f"Unknown storage type: {storage_type}")
