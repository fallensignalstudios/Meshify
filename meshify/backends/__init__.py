"""Registry of image-to-3D backends. Loaded models are cached (VRAM is precious)."""

from __future__ import annotations

import threading

from .base import BackendInfo, ImageTo3DBackend
from .dummy import DummyBackend
from .hunyuan3d import Hunyuan3DBackend
from .trellis import TrellisBackend

BACKENDS: dict[str, type[ImageTo3DBackend]] = {
    "trellis": TrellisBackend,
    "hunyuan3d": Hunyuan3DBackend,
    "dummy": DummyBackend,
}

_instances: dict[str, ImageTo3DBackend] = {}
_lock = threading.Lock()


def get_backend(name: str) -> ImageTo3DBackend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Choose from: {', '.join(BACKENDS)}")
    available, reason = BACKENDS[name].is_available()
    if not available:
        raise RuntimeError(reason)
    with _lock:
        if name not in _instances:
            # Only keep one heavy model resident at a time.
            _instances.clear()
            _instances[name] = BACKENDS[name]()
        backend = _instances[name]
        backend.ensure_loaded()
        return backend


def list_backends() -> list[dict]:
    out = []
    for name, cls in BACKENDS.items():
        available, reason = cls.is_available()
        info: BackendInfo = cls.info
        out.append({
            "name": name,
            "license": info.license,
            "commercial_ok": info.commercial_ok,
            "notes": info.notes,
            "available": available,
            "reason": reason,
        })
    return out


__all__ = ["BACKENDS", "ImageTo3DBackend", "get_backend", "list_backends"]
