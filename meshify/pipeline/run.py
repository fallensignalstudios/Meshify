"""The end-to-end pipeline: prompt/image -> reference -> 3D -> realism pass -> export."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from ..backends import get_backend
from ..config import OUTPUT_ROOT, GenerationRequest
from ..gpu import low_vram
from . import images, materials
from .mesh import prepare_mesh

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


@dataclass
class GenerationResult:
    job_id: str
    output_dir: Path
    files: dict[str, str] = field(default_factory=dict)  # label -> filename
    stats: dict = field(default_factory=dict)


def _safe_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    return cleaned[:64] or "asset"


def generate(req: GenerationRequest, progress: ProgressFn | None = None,
             job_id: str | None = None, output_root: Path = OUTPUT_ROOT) -> GenerationResult:
    req.validate()
    progress = progress or (lambda stage, frac: log.info("[%3d%%] %s", frac * 100, stage))
    job_id = job_id or uuid.uuid4().hex[:12]
    out = output_root / job_id
    textures = out / "textures"
    textures.mkdir(parents=True, exist_ok=True)
    name = _safe_name(req.asset_name)
    result = GenerationResult(job_id=job_id, output_dir=out)
    t0 = time.time()

    # 1. Reference image
    if req.image_path:
        progress("Loading reference image", 0.05)
        reference = Image.open(req.image_path)
    else:
        progress("Generating photoreal reference image", 0.05)
        reference = images.text_to_image(req.prompt, req.seed, req.realism)
        if low_vram():
            images.release_text_to_image()
    reference.save(out / "reference.png")
    result.files["reference"] = "reference.png"

    progress("Removing background", 0.2)
    cutout = images.frame_object(images.remove_background(reference))
    cutout.save(out / "cutout.png")
    result.files["cutout"] = "cutout.png"

    # 2. Image -> 3D
    progress(f"Loading {req.backend} model", 0.25)
    backend = get_backend(req.backend)
    progress(f"Generating 3D mesh with {req.backend}", 0.3)
    mesh = backend.generate(cutout, seed=req.seed, texture_size=req.texture_size,
                            target_faces=req.target_faces)

    # 3. Realism pass: delight + PBR maps
    base_color = materials.extract_base_color(mesh)
    if req.realism and base_color is not None:
        progress("Building PBR materials (delight, normal, roughness, AO)", 0.75)
        maps = materials.build_pbr_maps(base_color, req.delight_strength)
        materials.apply_pbr_maps(mesh, maps)
        for label, img, suffix in [
            ("albedo", maps.albedo, "BaseColor"),
            ("normal_dx", maps.normal_dx, "Normal_DX"),
            ("normal_gl", maps.normal_gl, "Normal_GL"),
            ("roughness", maps.roughness, "Roughness"),
            ("ao", maps.ao, "AO"),
            ("orm", maps.orm, "ORM"),
        ]:
            fname = f"T_{name}_{suffix}.png"
            img.save(textures / fname)
            result.files[label] = f"textures/{fname}"
    elif base_color is None:
        log.warning("Backend returned no texture; skipping PBR material pass.")

    # 4. Game-ready mesh
    progress("Cleaning mesh and setting real-world scale", 0.88)
    mesh = prepare_mesh(mesh, req.target_faces, req.real_world_height_m)

    # 5. Export
    progress("Exporting", 0.95)
    glb_name = f"SM_{name}.glb"
    mesh.export(out / glb_name)
    result.files["glb"] = glb_name
    try:
        obj_name = f"SM_{name}.obj"
        mesh.export(out / obj_name)
        result.files["obj"] = obj_name
    except Exception as exc:  # OBJ is a convenience; GLB is the real deliverable.
        log.warning("OBJ export failed: %s", exc)

    extents = mesh.extents
    result.stats = {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "size_m": {"x": round(float(extents[0]), 3), "y": round(float(extents[1]), 3),
                   "z": round(float(extents[2]), 3)},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meshify.json").write_text(json.dumps({
        "request": req.to_dict(),
        "files": result.files,
        "stats": result.stats,
    }, indent=2))
    progress("Done", 1.0)
    return result
