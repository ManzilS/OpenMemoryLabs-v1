"""
oml/utils/device.py — GPU / CPU device resolution
===================================================
Single source of truth for hardware detection across all OML components.

Usage
-----
    from oml.utils.device import resolve_device

    device = resolve_device("auto")   # → "cuda" if GPU present, else "cpu"
    device = resolve_device("cuda")   # → "cuda" (validated)
    device = resolve_device("cpu")    # → "cpu"

Passing ``"auto"`` is the recommended default — it automatically uses the GPU
when one is available and gracefully falls back to CPU otherwise.
"""

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Cache the auto-detected result so we only probe once per process
_cached_auto: Optional[str] = None


def _nvidia_smi_info() -> Optional[dict]:
    """Query nvidia-smi for GPU info (works even without CUDA-enabled PyTorch)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True, stderr=subprocess.DEVNULL,
        )
        line = out.strip().split("\n")[0]
        name, mem_mib = [s.strip() for s in line.split(",")]
        return {"gpu_name": name, "vram_gb": round(float(mem_mib) / 1024, 1)}
    except Exception:
        return None


def resolve_device(device: str = "auto") -> str:
    """
    Resolve a device string to a concrete ``"cuda"`` or ``"cpu"`` value.

    Args:
        device: One of ``"auto"``, ``"cuda"``, ``"cpu"``, or a specific CUDA
                device like ``"cuda:0"``.  ``"auto"`` selects ``"cuda"`` when
                a GPU is available and falls back to ``"cpu"`` otherwise.

    Returns:
        A concrete device string suitable for passing to PyTorch, HuggingFace
        Transformers, or SentenceTransformers.
    """
    global _cached_auto

    if device != "auto":
        return device

    if _cached_auto is not None:
        return _cached_auto

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(
                f"[device] GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM) → using CUDA"
            )
            _cached_auto = "cuda"
        else:
            logger.info("[device] No CUDA GPU found → using CPU")
            _cached_auto = "cpu"
    except (ImportError, OSError):
        # ImportError  → torch not installed
        # OSError/WinError 127 → torch installed but DLL broken (e.g. Python beta vs stable mismatch)
        logger.warning("[device] PyTorch not loadable → defaulting to CPU")
        _cached_auto = "cpu"

    return _cached_auto


def get_device_info() -> dict:
    """
    Return a dict with GPU/CPU information useful for display in the UI.

    Keys: ``device`` (str), ``gpu_name`` (str|None), ``vram_gb`` (float|None),
    ``cuda_available`` (bool), ``gpu_count`` (int), ``gpu_visible`` (bool).

    ``gpu_visible`` is True when a GPU is physically present (detected via
    nvidia-smi) even if PyTorch cannot use it (e.g. CPU-only torch build).
    """
    info = {
        "device": resolve_device("auto"),
        "gpu_name": None,
        "vram_gb": None,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_visible": False,
    }
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count()
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
            info["gpu_visible"] = True
    except (ImportError, OSError):
        pass

    # Fallback: detect GPU via nvidia-smi even when torch lacks CUDA support
    if not info["cuda_available"]:
        smi = _nvidia_smi_info()
        if smi:
            info["gpu_visible"] = True
            info["gpu_name"] = info["gpu_name"] or smi["gpu_name"]
            info["vram_gb"] = info["vram_gb"] or smi["vram_gb"]

    return info


def reset_cache() -> None:
    """Reset the cached auto-detection result (useful in tests)."""
    global _cached_auto
    _cached_auto = None
