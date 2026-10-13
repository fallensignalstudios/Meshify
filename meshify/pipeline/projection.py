"""Projection texturing: put the reference image back onto the mesh.

Image-to-3D texture painters repaint the object from their own guess and lose
what made the reference good: crisp detail, exact colours and, worst of all,
glowing channels (they come back as dull paint). But we have the reference: a
clean side view of exactly this object. So:

1. Find how the mesh is oriented by matching its silhouette (along each axis,
   with flips) to the reference cut-out.
2. Rasterize the mesh in UV space. Each texel gets its 3D position, which
   projects to a pixel in the reference.
3. Texels that face the camera, and aren't hidden behind other geometry, take
   that pixel's colour, weighted by how squarely they face it. The far side
   gets the mirrored view (fine for side-profile props like weapons);
   surfaces seen edge-on keep the baked texture.

The same mapping projects any reference-space map, e.g. the glow mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

MIN_SILHOUETTE_IOU = 0.6  # below this, the mesh doesn't match the reference well enough


@dataclass
class Projection:
    """Where each texel lands in the reference, and how much to trust it."""

    screen: np.ndarray  # (H, W, 2) reference pixel coordinates (x, y)
    weight: np.ndarray  # (H, W) 0..1, 0 = keep the baked texture
    iou: float  # silhouette match quality

    def sample(self, ref_map: np.ndarray) -> np.ndarray:
        """Project a reference-space map (H_ref, W_ref[, C]) into texture space."""
        h, w = ref_map.shape[:2]
        x = np.clip(np.round(self.screen[..., 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(self.screen[..., 1]).astype(int), 0, h - 1)
        return ref_map[y, x]


def _silhouette(points: np.ndarray, faces: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    for tri in points[faces]:
        draw.polygon([tuple(p) for p in tri], fill=255)
    return np.asarray(img) > 0


def _view_candidates():
    """(depth axis, screen-x axis, screen-y axis, flip x, flip y), most likely
    first: image-to-3D models usually face the input view along Z, x right, y up.
    Mirror-symmetric silhouettes tie, and ties keep the earlier candidate."""
    for depth in (2, 0, 1):
        a, b = [i for i in range(3) if i != depth]
        for sx, sy in ((a, b), (b, a)):
            for fx, fy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                yield depth, sx, sy, fx, fy


def _to_screen(vertices, view, box, ref_box):
    """Map mesh vertices into reference pixels: the mesh's screen-plane bounding
    box is fitted onto the reference cut-out's bounding box."""
    depth, sx, sy, fx, fy = view
    x = vertices[:, sx] * fx
    y = -vertices[:, sy] * fy  # image rows grow downward
    (x0, y0, x1, y1) = box
    (rx0, ry0, rx1, ry1) = ref_box
    px = rx0 + (x - x0) / max(x1 - x0, 1e-9) * (rx1 - rx0)
    py = ry0 + (y - y0) / max(y1 - y0, 1e-9) * (ry1 - ry0)
    return np.stack([px, py], axis=1), vertices[:, depth]


def find_view(mesh: trimesh.Trimesh, ref_mask: np.ndarray):
    """Best orientation (by silhouette IoU) of the mesh against the cut-out."""
    ys, xs = np.nonzero(ref_mask)
    ref_box = (xs.min(), ys.min(), xs.max(), ys.max())
    # Compare at low resolution: the cut-out's bounding box scaled to ~160 px.
    scale = 160 / max(ref_box[2] - ref_box[0], ref_box[3] - ref_box[1], 1)
    small = (max(8, int((ref_box[2] - ref_box[0]) * scale)), max(8, int((ref_box[3] - ref_box[1]) * scale)))
    ref_small = np.asarray(Image.fromarray(
        ref_mask[ref_box[1]:ref_box[3] + 1, ref_box[0]:ref_box[2] + 1].astype(np.uint8) * 255)
        .resize(small)) > 127
    best = (0.0, None)
    faces = mesh.faces
    if len(faces) > 6000:  # silhouettes don't need every triangle
        faces = faces[np.random.default_rng(0).choice(len(faces), 6000, replace=False)]
    for view in _view_candidates():
        depth, sx, sy, fx, fy = view
        x, y = mesh.vertices[:, sx] * fx, -mesh.vertices[:, sy] * fy
        box = (x.min(), y.min(), x.max(), y.max())
        # Uniform scale here: stretching would let any box match any rectangle.
        mw, mh = max(box[2] - box[0], 1e-9), max(box[3] - box[1], 1e-9)
        k = np.sqrt((small[0] * small[1]) / (mw * mh))
        cx, cy = (small[0] - 1) / 2, (small[1] - 1) / 2
        fit = (cx - mw * k / 2, cy - mh * k / 2, cx + mw * k / 2, cy + mh * k / 2)
        pts, _ = _to_screen(mesh.vertices, view, box, fit)
        sil = _silhouette(pts, faces, small)
        inter, union = (sil & ref_small).sum(), (sil | ref_small).sum()
        iou = inter / max(union, 1)
        if iou > best[0] + 0.01:  # clearly better, not a mirror-symmetric tie
            best = (iou, (view, box))
    return best[0], best[1], ref_box


def _rasterize(tri_xy: np.ndarray, size: tuple[int, int]):
    """Yield (ys, xs, barycentrics) for the pixels each triangle covers."""
    w, h = size
    for i, tri in enumerate(tri_xy):
        x0, y0 = np.floor(tri.min(0)).astype(int)
        x1, y1 = np.ceil(tri.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = tri
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
        l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-4) & (l2 >= -1e-4) & (l3 >= -1e-4)
        if inside.any():
            yield i, (gy[inside] - 0.5).astype(int), (gx[inside] - 0.5).astype(int), \
                np.stack([l1[inside], l2[inside], l3[inside]], axis=1)


def project(mesh: trimesh.Trimesh, texture_size: tuple[int, int],
            reference: Image.Image) -> Projection | None:
    """Map texture texels to reference pixels. None if the mesh has no UVs or
    its silhouette doesn't match the reference (then the bake is kept as is)."""
    visual = mesh.visual
    if not isinstance(visual, trimesh.visual.TextureVisuals) or visual.uv is None:
        return None
    ref = np.asarray(reference.convert("RGBA"))
    ref_mask = ref[..., 3] > 127
    if ref_mask.sum() < 500:
        return None

    iou, found, ref_box = find_view(mesh, ref_mask)
    if found is None or iou < MIN_SILHOUETTE_IOU:
        log.info("Projection skipped: mesh silhouette matches the reference poorly (IoU %.2f)", iou)
        return None
    view, box = found
    screen, depth = _to_screen(mesh.vertices, view, box, ref_box)
    rw, rh = ref_mask.shape[1], ref_mask.shape[0]

    # Depth buffers for the camera on each side (the far side sees the mirror view).
    zfront = np.full((rh, rw), -np.inf, np.float32)
    zback = np.full((rh, rw), np.inf, np.float32)
    faces = mesh.faces
    for i, ys, xs, bary in _rasterize(screen[faces], (rw, rh)):
        z = bary @ depth[faces[i]]
        np.maximum.at(zfront, (ys, xs), z)
        np.minimum.at(zback, (ys, xs), z)
    extent = float(depth.max() - depth.min()) or 1.0
    eps = 0.02 * extent

    tw, th = texture_size
    uv = np.asarray(visual.uv, dtype=np.float64)
    uv_px = np.stack([uv[:, 0] * tw, (1.0 - uv[:, 1]) * th], axis=1)  # texture rows grow down
    normals = mesh.face_normals[:, view[0]]  # component along the view (depth) axis
    out_screen = np.zeros((th, tw, 2), np.float32)
    weight = np.zeros((th, tw), np.float32)
    for i, ys, xs, bary in _rasterize(uv_px[faces], (tw, th)):
        f = faces[i]
        sxy = bary @ screen[f]
        z = bary @ depth[f]
        n = normals[i]
        px = np.clip(sxy[:, 0].astype(int), 0, rw - 1)
        py = np.clip(sxy[:, 1].astype(int), 0, rh - 1)
        if n >= 0:  # faces the camera on the +depth side
            visible = z >= zfront[py, px] - eps
        else:
            visible = z <= zback[py, px] + eps
        w = (abs(n) ** 1.5) * visible * ref_mask[py, px]
        out_screen[ys, xs] = sxy
        weight[ys, xs] = np.maximum(weight[ys, xs], w)
    log.info("Projected the reference onto %.0f%% of the texture (silhouette IoU %.2f)",
             100 * (weight > 0.3).mean(), iou)
    return Projection(screen=out_screen, weight=weight, iou=float(iou))
