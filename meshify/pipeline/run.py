"""The end-to-end pipeline: prompt/image -> reference -> 3D -> realism pass -> export."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from ..backends import get_backend
from ..config import DEFAULT_TARGET_FACES, DEFAULT_TEXTURE_SIZE, OUTPUT_ROOT, GenerationRequest
from ..gpu import low_vram
from ..styles import resolve as resolve_style
from . import images, judge, materials
from .mesh import prepare_mesh

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]

MAX_REFERENCE_ATTEMPTS = int(os.environ.get("MESHIFY_REFERENCE_ATTEMPTS", "3"))
MAX_BORDER_CONTACT = 0.03  # above this share of the border, the object is cut off
DEFAULT_CANDIDATES = int(os.environ.get("MESHIFY_CANDIDATES", "3"))


def _pick_reference(req, style, style_images, progress, out, result):
    """Generate text-to-image candidates and keep the best one.

    Each candidate is checked for cropping (object touching the frame edge)
    and, when the judge can run, scored against the asset type's good/bad
    descriptions (e.g. "one scope" vs "two scopes"). Without a judge, the
    first uncropped candidate wins, so nothing extra is generated.
    """
    wanted = max(1, req.candidates or DEFAULT_CANDIDATES)
    use_judge = wanted > 1 and judge.available()
    if not use_judge:
        wanted = 1
    good, bad = style.checks(req.prompt) if style else ([], [])
    subject = (req.prompt or "object").strip()
    good = good or [t.replace("{subject}", subject) for t in judge.DEFAULT_GOOD]
    bad = bad or list(judge.DEFAULT_BAD)

    cands = []  # dicts: seed, image, cutout, contact
    limit = max(wanted, MAX_REFERENCE_ATTEMPTS)
    for i in range(limit):
        uncropped = [c for c in cands if c["contact"] <= MAX_BORDER_CONTACT]
        # Stop once we have enough candidates and at least one isn't cropped.
        if len(cands) >= wanted and uncropped:
            break
        seed = req.seed + i
        progress(f"Generating reference image {i + 1}/{wanted}", 0.05 + 0.1 * i / limit)
        image = images.text_to_image(req.prompt, seed, req.realism, style,
                                     style_images=style_images,
                                     style_strength=req.style_strength)
        cut = images.remove_background(image)
        contact = images.border_contact(cut)
        if contact > MAX_BORDER_CONTACT:
            log.warning("Candidate %d (seed %d) is cropped (%.0f%% of the border is object)",
                        i + 1, seed, contact * 100)
        cands.append({"seed": seed, "image": image, "cutout": cut, "contact": contact})

    scores = None
    if use_judge and len(cands) > 1:
        progress("Judging candidates", 0.17)
        scores = judge.score([c["image"] for c in cands], good, bad)
    for i, c in enumerate(cands):
        c["score"] = scores[i] if scores else None
        cropped = c["contact"] > MAX_BORDER_CONTACT
        # Cropping is disqualifying; among the rest the judge decides, else the
        # earliest uncropped one wins.
        c["rank"] = (0 if cropped else 1, c["score"] if scores else -i, -c["contact"])

    best = max(cands, key=lambda c: c["rank"])
    if best["contact"] > MAX_BORDER_CONTACT:
        log.warning("All %d references were cropped; using the least-cropped one. "
                    "Try rewording the prompt or another seed.", len(cands))
    if len(cands) > 1:
        (out / "candidates").mkdir(exist_ok=True)
        for i, c in enumerate(cands, 1):
            fname = f"candidates/candidate_{i}.png"
            c["image"].save(out / fname)
            result.files[f"candidate_{i}"] = fname
    result.stats["candidates"] = [
        {"seed": c["seed"], "score": None if c["score"] is None else round(c["score"], 3),
         "cropped": c["contact"] > MAX_BORDER_CONTACT, "chosen": c is best}
        for c in cands]
    return best["image"], best["cutout"], len(cands)


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
    style = resolve_style(req.style_pack, req.faction, req.category, req.condition)
    texture_size = req.texture_size or (style and style.texture) or DEFAULT_TEXTURE_SIZE
    if req.target_faces is not None:
        faces = req.target_faces
    else:
        faces = (style and style.tris) or DEFAULT_TARGET_FACES
    size_m = req.real_world_size_m or (style and style.size_m) or None
    size_axis = req.size_axis or (style.size_axis if style else "height")
    pivot = req.pivot or (style.pivot if style else "bottom_center")
    progress = progress or (lambda stage, frac: log.info("[%3d%%] %s", frac * 100, stage))
    job_id = job_id or uuid.uuid4().hex[:12]
    out = output_root / job_id
    textures = out / "textures"
    textures.mkdir(parents=True, exist_ok=True)
    name = _safe_name(req.asset_name)
    result = GenerationResult(job_id=job_id, output_dir=out)
    t0 = time.time()

    # 1. Reference image
    attempts = 1
    if req.image_path:
        progress("Loading reference image", 0.05)
        reference = Image.open(req.image_path)
        progress("Removing background", 0.2)
        isolated = images.remove_background(reference)
    else:
        # Uploaded style images replace the library; otherwise the faction's
        # library references (most specific first) are used automatically.
        ref_paths = req.style_image_paths or (style.references if style else [])
        style_images = [Image.open(p) for p in ref_paths]
        for i, img in enumerate(style_images, 1):
            fname = f"style_reference_{i}.png"
            img.save(out / fname)
            result.files[f"style_reference_{i}"] = fname
        reference, isolated, attempts = _pick_reference(req, style, style_images,
                                                        progress, out, result)
        if low_vram():
            images.release_text_to_image()
            judge.release()
    reference.save(out / "reference.png")
    result.files["reference"] = "reference.png"

    cutout = images.frame_object(isolated)
    cutout.save(out / "cutout.png")
    result.files["cutout"] = "cutout.png"

    # 2. Image -> 3D
    progress(f"Loading {req.backend} model", 0.25)
    backend = get_backend(req.backend)
    progress(f"Generating 3D mesh with {req.backend}", 0.3)
    mesh = backend.generate(cutout, seed=req.seed, texture_size=texture_size,
                            target_faces=faces or None)

    # 3. Realism pass: delight + PBR maps
    base_color = materials.extract_base_color(mesh)
    if req.realism and base_color is not None:
        progress("Building PBR materials (delight, normal, roughness, AO)", 0.75)
        maps = materials.build_pbr_maps(base_color, req.delight_strength,
                                        style.material if style else None)
        materials.apply_pbr_maps(mesh, maps)
        for label, img, suffix in [
            ("albedo", maps.albedo, "BaseColor"),
            ("normal_dx", maps.normal_dx, "Normal_DX"),
            ("normal_gl", maps.normal_gl, "Normal_GL"),
            ("roughness", maps.roughness, "Roughness"),
            ("ao", maps.ao, "AO"),
            ("orm", maps.orm, "ORM"),
            ("metallic", maps.metallic, "Metallic"),
            ("emissive", maps.emissive, "Emissive"),
            ("emissive_mask", maps.emissive_mask, "EmissiveMask"),
        ]:
            if img is None:
                continue
            fname = f"T_{name}_{suffix}.png"
            img.save(textures / fname)
            result.files[label] = f"textures/{fname}"
    elif base_color is None:
        log.warning("Backend returned no texture; skipping PBR material pass.")

    # 4. Game-ready mesh
    progress("Cleaning mesh and setting real-world scale", 0.88)
    mesh = prepare_mesh(mesh, faces or None, size_m, size_axis, pivot,
                        style.grid_m if style else None)

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
        **result.stats,
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "size_m": {"x": round(float(extents[0]), 3), "y": round(float(extents[1]), 3),
                   "z": round(float(extents[2]), 3)},
        "seconds": round(time.time() - t0, 1),
        "texture_size": texture_size,
        "emissive": "emissive" in result.files,
        "reference_attempts": attempts,
        "style_references": len([k for k in result.files if k.startswith("style_reference_")]),
    }
    (out / "meshify.json").write_text(json.dumps({
        "request": req.to_dict(),
        "files": result.files,
        "stats": result.stats,
    }, indent=2))
    progress("Done", 1.0)
    return result
