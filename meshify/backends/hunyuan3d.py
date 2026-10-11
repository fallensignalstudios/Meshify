"""Tencent Hunyuan3D 2.x - very high quality, optional backend.

LICENSE WARNING: the Tencent Hunyuan Community License is NOT open source.
It excludes the EU, UK and South Korea, and caps commercial use by user count.
Read it before shipping assets made with this backend:
https://github.com/Tencent/Hunyuan3D-2/blob/main/LICENSE
"""

from __future__ import annotations

import importlib.util
import os

import trimesh
from PIL import Image

from .base import BackendInfo, ImageTo3DBackend

MODEL_ID = os.environ.get("MESHIFY_HUNYUAN_MODEL", "tencent/Hunyuan3D-2")


class Hunyuan3DBackend(ImageTo3DBackend):
    info = BackendInfo(
        name="hunyuan3d",
        license="Tencent Hunyuan Community License (restricted)",
        commercial_ok=False,
        notes="Excellent detail and textures. Check the license for your region/use.",
    )

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("hy3dgen") is None:
            return False, "Hunyuan3D (hy3dgen) is not installed (see docs/INSTALL.md)."
        return True, ""

    def load(self) -> None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        self._shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_ID)
        self._paint = Hunyuan3DPaintPipeline.from_pretrained(MODEL_ID)

    def generate(self, image: Image.Image, seed: int, texture_size: int,
                 target_faces: int | None) -> trimesh.Trimesh:
        import torch
        from hy3dgen.shapegen import DegenerateFaceRemover, FloaterRemover

        generator = torch.manual_seed(seed)
        mesh = self._shape(image=image, generator=generator)[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        # Reduce before painting: keeps UVs clean and painting fast.
        budget = target_faces or 200_000
        if len(mesh.faces) > budget:
            from hy3dgen.shapegen import FaceReducer

            mesh = FaceReducer()(mesh, max_facenum=budget)
        mesh = self._paint(mesh, image=image)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        return mesh
