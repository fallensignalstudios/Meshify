"""GPU helpers: VRAM detection, low-VRAM mode and freeing memory between stages."""

from __future__ import annotations

import gc
import importlib.util
import os
from functools import lru_cache

LOW_VRAM_THRESHOLD_GB = 15.0


@lru_cache(maxsize=1)
def gpu_info() -> dict | None:
    """Name, VRAM and compute capability of GPU 0, or None without CUDA."""
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "vram_gb": round(props.total_memory / 1024**3, 1),
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def low_vram() -> bool:
    """True on cards under ~15 GB (e.g. 12 GB RTX 5070). MESHIFY_LOW_VRAM=1/0 overrides."""
    override = os.environ.get("MESHIFY_LOW_VRAM")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    info = gpu_info()
    return info is not None and info["vram_gb"] < LOW_VRAM_THRESHOLD_GB


def free_cuda() -> None:
    gc.collect()
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def blackwell_torch_problem() -> str | None:
    """RTX 50-series (compute 12.x) needs a PyTorch build with CUDA 12.8+."""
    info = gpu_info()
    if info is None or not info["compute_capability"].startswith("12"):
        return None
    major, minor = (int(x) for x in (info["cuda"] or "0.0").split(".")[:2])
    if (major, minor) < (12, 8):
        return (f"{info['name']} needs PyTorch built for CUDA 12.8+, but this one uses "
                f"CUDA {info['cuda']}. See docs/INSTALL.md (RTX 50-series).")
    return None
