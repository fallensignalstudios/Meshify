"""The interface every image-to-3D model plugs into."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import trimesh
from PIL import Image


@dataclass
class BackendInfo:
    name: str
    license: str
    commercial_ok: bool
    notes: str
    produces_texture: bool = True


class ImageTo3DBackend(ABC):
    """Wraps one open model. Implementations should lazy-load weights in `load()`."""

    info: BackendInfo

    def __init__(self) -> None:
        self._loaded = False

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    @abstractmethod
    def load(self) -> None:
        """Download/load weights onto the GPU."""

    @abstractmethod
    def generate(self, image: Image.Image, seed: int, texture_size: int,
                 target_faces: int | None) -> trimesh.Trimesh:
        """Turn one RGBA (background removed) reference image into a textured mesh.

        Backends should reduce to `target_faces` *before* baking textures so the
        UVs stay clean (None = keep full density).
        """

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return (available, reason) without loading weights."""
        return True, ""
