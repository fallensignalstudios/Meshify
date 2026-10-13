"""CPU-only placeholder backend for testing the pipeline without a GPU.

It builds a rounded box textured with the reference image, so every later
stage (delighting, PBR maps, decimation, scaling, export) runs for real.
"""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

from .base import BackendInfo, ImageTo3DBackend


class DummyBackend(ImageTo3DBackend):
    info = BackendInfo(
        name="dummy",
        license="MIT",
        commercial_ok=True,
        notes="Test-only placeholder. Produces a textured blob, not a real model.",
    )

    def load(self) -> None:
        pass

    def generate(self, image: Image.Image, seed: int, texture_size: int,
                 target_faces: int | None) -> trimesh.Trimesh:
        mesh = trimesh.creation.icosphere(subdivisions=5, radius=0.5)
        mesh.vertices *= np.array([1.0, 1.4, 0.8])  # make it less of a perfect ball

        # Spherical UVs so the reference image wraps around the blob.
        v = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        uv = np.column_stack([
            0.5 + np.arctan2(v[:, 0], v[:, 2]) / (2 * np.pi),
            0.5 + np.arcsin(np.clip(v[:, 1], -1, 1)) / np.pi,
        ])
        # Fill the transparent background with the object's mean colour.
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba)[..., 3] > 16
        mean = tuple(int(c) for c in np.asarray(rgba)[..., :3][alpha].mean(axis=0)) \
            if alpha.any() else (128, 128, 128)
        canvas = Image.new("RGBA", rgba.size, mean + (255,))
        canvas.alpha_composite(rgba)
        texture = canvas.convert("RGB").resize((texture_size, texture_size))
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        return mesh
