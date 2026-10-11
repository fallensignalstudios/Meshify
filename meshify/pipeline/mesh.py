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


def scale_to_size(mesh: trimesh.Trimesh, size_m: float, axis: str = "height") -> None:
    """Uniformly scale so the height (Y) or the longest dimension is `size_m`."""
    current = mesh.extents[1] if axis == "height" else mesh.extents.max()
    if current > 0:
        mesh.apply_scale(size_m / current)


def snap_to_grid(mesh: trimesh.Trimesh, grid_m: float) -> None:
    """Uniformly scale so the widest footprint side is a whole number of grid cells.

    Modular kit pieces only line up if their widths are grid multiples
    (e.g. 1 m = 100 Unreal units).
    """
    footprint = max(mesh.extents[0], mesh.extents[2])
    if footprint <= 0:
        return
    cells = max(1, round(footprint / grid_m))
    mesh.apply_scale(cells * grid_m / footprint)


def set_pivot(mesh: trimesh.Trimesh, pivot: str = "bottom_center") -> None:
    """glTF is Y-up, metres; Unreal's importer converts to Z-up centimetres.

    bottom_center: props sit on the floor where you drop them.
    bottom_corner: modular pieces snap edge-to-edge on Unreal's grid.
    center: weapons and armour, for attaching to sockets.
    """
    lo, hi = mesh.bounds
    centre = (lo + hi) / 2
    if pivot == "center":
        offset = -centre
    elif pivot == "bottom_corner":
        offset = np.array([-lo[0], -lo[1], -lo[2]])
    else:
        offset = np.array([-centre[0], -lo[1], -centre[2]])
    mesh.apply_translation(offset)


def prepare_mesh(mesh: trimesh.Trimesh, target_faces: int | None, size_m: float | None,
                 size_axis: str = "height", pivot: str = "bottom_center",
                 grid_m: float | None = None) -> trimesh.Trimesh:
    mesh = remove_floaters(mesh)
    mesh = decimate(mesh, target_faces)
    if size_m is not None:
        scale_to_size(mesh, size_m, size_axis)
    elif mesh.extents.max() < 0.05 or mesh.extents.max() > 50:
        # Unknown real size: default to ~1m so it isn't microscopic in-engine.
        scale_to_size(mesh, 1.0, "longest")
    if grid_m:
        snap_to_grid(mesh, grid_m)
    set_pivot(mesh, pivot)
    return mesh
