"""Generation settings shared by the CLI, the server and the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

OUTPUT_ROOT = Path(os.environ.get("MESHIFY_OUTPUT_DIR", "outputs"))
DEFAULT_BACKEND = os.environ.get("MESHIFY_BACKEND", "trellis")


@dataclass
class GenerationRequest:
    """Everything needed to turn a prompt or image into an Unreal-ready asset."""

    prompt: str | None = None
    image_path: str | None = None
    backend: str = DEFAULT_BACKEND
    seed: int = 42

    # Realism pipeline
    realism: bool = True  # photoreal prompt shaping + delighting + PBR maps
    delight_strength: float = 0.6  # 0 = keep baked lighting, 1 = flatten fully
    texture_size: int = 2048

    # Mesh / export
    target_faces: int | None = 50_000  # None = keep full density (e.g. for Nanite)
    real_world_height_m: float | None = None  # scale so the asset is N metres tall
    asset_name: str = "asset"

    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt and not self.image_path:
            raise ValueError("Provide a prompt, an image, or both.")
        if not 0.0 <= self.delight_strength <= 1.0:
            raise ValueError("delight_strength must be between 0 and 1.")
        if self.texture_size not in (512, 1024, 2048, 4096):
            raise ValueError("texture_size must be 512, 1024, 2048 or 4096.")
        if self.target_faces is not None and self.target_faces < 500:
            raise ValueError("target_faces must be at least 500 (or None).")
        if self.real_world_height_m is not None and self.real_world_height_m <= 0:
            raise ValueError("real_world_height_m must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)
