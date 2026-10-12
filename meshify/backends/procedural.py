"""Procedural hard-surface backend: models the asset instead of dreaming it.

Builds clean, chamfered, armour-plated geometry from the reference
silhouette (see meshify/hardsurface.py). Best for side-profile hard-surface
assets: weapons, ship components, some props. Not for organic shapes
(characters, Eclipse growths): use a neural backend for those.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import trimesh
from PIL import Image

from .base import BackendInfo, ImageTo3DBackend


class ProceduralBackend(ImageTo3DBackend):
    info = BackendInfo(
        name="procedural",
        license="MIT",
        commercial_ok=True,
        notes="Hard-surface modelling from the silhouette: crisp bevels, armour plates, "
              "clean UVs, tiny triangle counts. Side-profile props (weapons, ship parts).",
    )
    # Wants the full-resolution cut-out rather than the 518 px framed one.
    wants_full_reference = True
    style_name: str | None = None  # set per job: faction id picks the shape language

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        for mod in ("shapely", "mapbox_earcut"):
            if importlib.util.find_spec(mod) is None:
                return False, f"pip install shapely mapbox_earcut (missing {mod})"
        return True, ""

    def load(self) -> None:
        pass

    def generate(self, image: Image.Image, seed: int, texture_size: int,
                 target_faces: int | None) -> trimesh.Trimesh:
        from .. import hardsurface as hs

        style = hs.STYLES.get(self.style_name or "", hs.HardSurfaceStyle())
        rgba = np.asarray(image.convert("RGBA"))
        sil = hs.trace_silhouette(rgba[..., 3] > 127, style)
        mesh, _ = hs.build(sil, style)
        texture = hs.paint_from_reference(mesh, sil, rgba, texture_size)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv,
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture))
        return mesh
