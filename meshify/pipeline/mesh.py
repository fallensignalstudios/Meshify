"""Mesh clean-up and game-engine prep: floaters, decimation, pivot, real-world scale."""

from __future__ import annotations

import importlib.util
import logging

import numpy as np
import trimesh

log = logging.getLogger(__name__)


def remove_floaters(mesh: trimesh.Trimesh, keep_ratio: float = 0.02) -> trimesh.Trimesh:
    """Drop disconnected bits smaller than `keep_ratio` of the largest piece."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    biggest = max(len(p.faces) for p in parts)
    kept = [p for p in parts if len(p.faces) >= biggest * keep_ratio]
    if len(kept) == len(parts):
        return mesh
    log.info("Removed %d floating fragments", len(parts) - len(kept))
    return trimesh.util.concatenate(kept)


def decimate(mesh: trimesh.Trimesh, target_faces: int | None) -> trimesh.Trimesh:
    if target_faces is None or len(mesh.faces) <= target_faces:
        return mesh
    if isinstance(mesh.visual, trimesh.visual.TextureVisuals) and mesh.visual.uv is not None:
        # Backends reduce before baking. Decimating now would scramble the UVs,
        # so leave it and let Nanite / Unreal's auto-LODs handle the rest.
        log.info("Textured mesh has %d faces (over budget); keeping UVs intact.",
                 len(mesh.faces))
        return mesh
    if importlib.util.find_spec("fast_simplification") is None:
        log.warning("fast_simplification not installed; skipping decimation "
                    "(pip install fast-simplification).")
        return mesh
    return mesh.simplify_quadric_decimation(face_count=target_faces)


def place_on_ground(mesh: trimesh.Trimesh, height_m: float | None) -> trimesh.Trimesh:
    """glTF is Y-up, metres. Centre on X/Z, sit the base on Y=0, optionally rescale.

    Unreal's glTF importer converts to Z-up centimetres, so the pivot ends up
    at the bottom centre of the asset, which is what you want for placement.
    """
    if height_m is not None:
        current = mesh.extents[1]
        if current > 0:
            mesh.apply_scale(height_m / current)
    lo, hi = mesh.bounds
    offset = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    mesh.apply_translation(offset)
    return mesh


def prepare_mesh(mesh: trimesh.Trimesh, target_faces: int | None,
                 height_m: float | None) -> trimesh.Trimesh:
    mesh = remove_floaters(mesh)
    mesh = decimate(mesh, target_faces)
    if height_m is None:
        # Unknown real size: default to ~1m tall so it isn't microscopic in-engine.
        height_m = 1.0 if mesh.extents.max() < 0.05 or mesh.extents.max() > 50 else None
    return place_on_ground(mesh, height_m)
