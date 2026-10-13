"""Microsoft TRELLIS (MIT license) - the default, fully permissive backend.

Install TRELLIS separately (see docs/INSTALL.md); it needs an NVIDIA GPU with
16GB+ VRAM. https://github.com/microsoft/TRELLIS
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import trimesh
from PIL import Image

from .base import BackendInfo, ImageTo3DBackend

MODEL_ID = os.environ.get("MESHIFY_TRELLIS_MODEL", "JeffreyXiang/TRELLIS-image-large")


class TrellisBackend(ImageTo3DBackend):
    info = BackendInfo(
        name="trellis",
        license="MIT",
        commercial_ok=True,
        notes="Clean geometry, fully permissive. Default choice.",
    )

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("trellis") is None:
            return False, "TRELLIS is not installed (see docs/INSTALL.md)."
        return True, ""

    def load(self) -> None:
        # TRELLIS reads these at import time; spconv is the most portable choice.
        os.environ.setdefault("SPCONV_ALGO", "native")
        from trellis.pipelines import TrellisImageTo3DPipeline

        self._pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID)
        self._pipeline.cuda()

    def generate(self, image: Image.Image, seed: int, texture_size: int,
                 target_faces: int | None) -> trimesh.Trimesh:
        from trellis.utils import postprocessing_utils

        outputs = self._pipeline.run(
            _premultiply_on_black(image),
            seed=seed,
            # Background is already removed by our preprocessing.
            preprocess_image=False,
            sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 12, "cfg_strength": 3.0},
        )
        raw_faces = int(outputs["mesh"][0].faces.shape[0])
        # to_glb simplifies first, then bakes the texture, so UVs stay clean.
        simplify = 0.0 if target_faces is None else max(0.0, 1.0 - target_faces / raw_faces)
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=min(simplify, 0.98),
            texture_size=texture_size,
        )
        return _as_single_mesh(glb)


def _premultiply_on_black(image: Image.Image) -> Image.Image:
    # TRELLIS's own preprocessing composites the cut-out onto black; match it.
    rgba = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:4]
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def _as_single_mesh(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Scene):
        return obj.to_geometry()
    return obj
