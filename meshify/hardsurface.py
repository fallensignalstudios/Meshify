"""A small hard-surface modelling kernel: what an artist does in Blender for a
side-profile prop (weapons, ship parts), automated.

    silhouette -> clean polygon -> parts split by thickness -> chamfered
    extrusions -> raised armour plates with panel grooves -> clean UV atlas

No neural network and no training: the shape comes straight from the
reference silhouette, so proportions are exact, edges are crisp, the
triangle count is small and the UVs are tidy. Depth (thickness) is the one
thing a single side view doesn't show, so it follows simple rules: the body
gets a receiver-like thickness, thin protrusions (barrels, sights) get a
near-square cross-section of their own height.

Coordinates: X along the object's length (0..1), Y up, Z thickness with the
reference view looking at +Z (front). Units are "lengths"; the pipeline
scales to real-world size afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from PIL import Image
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


@dataclass
class HardSurfaceStyle:
    """Shape language. Values are fractions of the object's length."""

    body_depth: float = 0.065  # receiver thickness
    chamfer: float = 0.006  # edge bevel on the body
    plate_height: float = 0.006  # how far armour plates stand proud
    plate_margin: float = 0.012  # plate inset from the body outline
    groove: float = 0.006  # panel-line gap between plates
    panel_pitch: float = 0.16  # spacing of vertical panel lines
    thin_ratio: float = 0.45  # thinner than this x the typical body height = "thin part"
    angular: bool = True  # mitred plate corners (Dominion) vs rounded (Reformation)
    simplify: float = 0.0035  # outline simplification tolerance
    denoise: float = 0.004  # outline features smaller than ~2x this are smoothed away
    appendage_depth: float = 0.72  # grip / magazine / stock drop: thinner than the core


STYLES = {
    "dominion": HardSurfaceStyle(angular=True, plate_height=0.007, panel_pitch=0.15),
    "reformation": HardSurfaceStyle(angular=False, chamfer=0.008, plate_height=0.004,
                                    panel_pitch=0.22, body_depth=0.055),
    "ancient": HardSurfaceStyle(angular=False, chamfer=0.009, plate_height=0.005, panel_pitch=0.2),
    "aurelion": HardSurfaceStyle(angular=False, chamfer=0.009, plate_height=0.006, panel_pitch=0.18),
    "eclipse": HardSurfaceStyle(angular=True, chamfer=0.004, plate_height=0.003, panel_pitch=0.3),
}


@dataclass
class Silhouette:
    polygon: Polygon | MultiPolygon  # in object units: x 0..1 along the length
    to_pixel: tuple[float, float, float, float]  # x0, y0 (pixels), scale (px per unit), height px
    height: float  # object height in units


@dataclass
class Part:
    polygon: Polygon
    z0: float
    z1: float
    chamfer: float
    kind: str  # "body", "thin", "plate"


@dataclass
class _Builder:
    verts: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    uvs: list = field(default_factory=list)

    def add(self, positions, uvs, tris):
        base = len(self.verts)
        self.verts.extend(positions)
        self.uvs.extend(uvs)
        self.faces.extend([(a + base, b + base, c + base) for a, b, c in tris])


# ------------------------------------------------------------- silhouette

def trace_silhouette(mask: np.ndarray, style: HardSurfaceStyle, max_px: int = 480) -> Silhouette:
    """Mask (True = object) -> simplified polygon, longest side = 1 unit."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Empty silhouette")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    k = min(1.0, max_px / max(crop.shape))
    small = np.asarray(Image.fromarray(crop.astype(np.uint8) * 255).resize(
        (max(1, round(crop.shape[1] * k)), max(1, round(crop.shape[0] * k))))) > 127
    h, w = small.shape
    # Union of horizontal pixel runs, row by row (image rows grow down; flip to Y up).
    boxes = []
    for r in range(h):
        row = np.concatenate([[False], small[r], [False]])
        edges = np.flatnonzero(row[1:] != row[:-1])
        for a, b in zip(edges[::2], edges[1::2]):
            boxes.append(box(a, h - r - 1, b, h - r))
    shape = unary_union(boxes)
    length_px = max(w, h)
    shape = affinity.scale(shape, 1 / length_px, 1 / length_px, origin=(0, 0))
    # Drop specks, keep the outline crisp: simplify the pixel staircase into edges.
    shape = shape.simplify(style.simplify, preserve_topology=True).buffer(0)
    # Close small notches, then drop small teeth (rail serrations, AI noise):
    # those read better as texture/normal detail than as jagged geometry.
    n = style.denoise
    shape = (shape.buffer(n, join_style="mitre").buffer(-2 * n, join_style="mitre")
             .buffer(n, join_style="mitre"))
    shape = _keep_large(shape, 0.002)
    scale_px = length_px / k  # original pixels per unit
    return Silhouette(shape, (float(x0), float(y0), scale_px, float(y1 - y0)), h / length_px)


def _keep_large(shape, min_area):
    parts = [shape] if isinstance(shape, Polygon) else list(getattr(shape, "geoms", []))
    parts = [orient(p, 1.0) for p in parts if p.area >= min_area]
    if not parts:
        raise ValueError("Silhouette too small after cleaning")
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _polys(shape) -> list[Polygon]:
    if shape.is_empty:
        return []
    if isinstance(shape, Polygon):
        return [orient(shape, 1.0)]
    return [orient(p, 1.0) for p in getattr(shape, "geoms", []) if isinstance(p, Polygon) and not p.is_empty]


# ----------------------------------------------------------------- parts

def plan_parts(sil: Silhouette, style: HardSurfaceStyle) -> list[Part]:
    """Split the outline into the body and thin protrusions, then add plates.

    A morphological opening (shrink, then grow back) keeps everything at
    least `thin_ratio` x the typical body height thick: that's the body.
    What it removes are the thin protrusions (barrels, rails, sight posts),
    which get a slimmer, near-square cross-section of their own."""
    shape = sil.polygon
    typical = _typical_height(shape)
    r = 0.5 * style.thin_ratio * typical
    join = "mitre" if style.angular else "round"
    body = shape.buffer(-r, join_style=join, mitre_limit=3.0).buffer(r, join_style=join, mitre_limit=3.0)
    body = body.intersection(shape)
    rest = shape.difference(body)
    thin_parts, crumbs = [], []
    for poly in _polys(rest.buffer(-1e-4).buffer(1e-4)):
        minx, miny, maxx, maxy = poly.bounds
        # Real protrusions are long and reasonably big; crumbs from the
        # opening along the body outline go back to the body.
        if poly.area > 2e-4 and max(maxx - minx, maxy - miny) > 2.5 * min(maxx - minx, maxy - miny, r):
            thin_parts.append(poly)
        else:
            crumbs.append(poly)
    if crumbs:
        body = unary_union([body, *crumbs])

    parts: list[Part] = []
    d = style.body_depth
    core, appendages = _split_core(body)
    for poly in _polys(core):
        if poly.area < 2e-4:
            continue
        parts.append(Part(poly, -d / 2, d / 2, style.chamfer, "body"))
        parts.extend(_plates(poly, d / 2, style, _feature_cuts(poly, style)))
    da = d * style.appendage_depth
    for poly in appendages:
        parts.append(Part(poly, -da / 2, da / 2, style.chamfer, "body"))
        parts.extend(_plates(poly, da / 2, style, []))
    for poly in thin_parts:
        minx, miny, maxx, maxy = poly.bounds
        thickness = min(maxy - miny, maxx - minx)
        depth = float(np.clip(0.85 * thickness, 0.01, d * 0.9))  # near-square cross-section
        bevel = min(style.chamfer * 1.6, depth * 0.3)  # big bevel: reads as round-ish tube
        parts.append(Part(poly, -depth / 2, depth / 2, bevel, "thin"))
    return parts


def _split_core(body) -> tuple:
    """Core = the horizontal band the receiver lives in; appendages = what
    hangs below or sits above it (grip, magazine, stock drop, sight block).
    Appendages get a slimmer depth, which gives the layered, assembled look."""
    xs = np.linspace(*body.bounds[0::2], 160)[1:-1]
    tops, bots = [], []
    for x in xs:
        inter = body.intersection(box(x - 5e-4, -1, x + 5e-4, 2))
        if not inter.is_empty and inter.area > 0:
            bots.append(inter.bounds[1])
            tops.append(inter.bounds[3])
    if not tops:
        return body, []
    band = box(-1, float(np.median(bots)), 2, float(np.median(tops)))
    core = body.intersection(band)
    appendages = []
    for poly in _polys(body.difference(band)):
        minx, miny, maxx, maxy = poly.bounds
        if poly.area > 1.5e-3 and min(maxx - minx, maxy - miny) > 0.02:
            appendages.append(poly)
        else:
            core = core.union(poly)
    return core, appendages


def _feature_cuts(poly: Polygon, style: HardSurfaceStyle) -> list[float]:
    """Panel-line positions where the outline has corners (where a designer
    would split panels), topped up to the panel pitch on long runs."""
    c = np.asarray(poly.exterior.coords)[:-1]
    e1 = c - np.roll(c, 1, 0)
    e2 = np.roll(c, -1, 0) - c
    ang = np.abs(np.arctan2(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0], (e1 * e2).sum(1)))
    minx, _, maxx, _ = poly.bounds
    xs = sorted(c[ang > 0.5, 0])
    edge = 0.05
    cuts: list[float] = []
    for x in xs:
        if minx + edge < x < maxx - edge and (not cuts or x - cuts[-1] > 0.06):
            cuts.append(float(x))
    filled: list[float] = []
    for a, b in zip([minx, *cuts], [*cuts, maxx]):
        n = int((b - a) // (1.6 * style.panel_pitch))
        filled += [a + (b - a) * (i + 1) / (n + 1) for i in range(n)]
        if b != maxx:
            filled.append(b)
    return filled


def _typical_height(shape) -> float:
    xs = np.linspace(0.02, 0.98, 120)
    heights = np.array([_column_height(shape, x) for x in xs])
    return float(np.median(heights[heights > 0])) if (heights > 0).any() else 0.1


def _column_height(shape, x, w=1e-3):
    """Filled height of the outline at x (gaps don't count)."""
    inter = shape.intersection(box(x - w / 2, -1, x + w / 2, 2))
    return inter.area / w if not inter.is_empty else 0.0


def _plates(body: Polygon, surface_z: float, style: HardSurfaceStyle,
            cut_xs: list[float]) -> list[Part]:
    """Armour plates: the body inset by a margin, cut into panels by grooves,
    standing proud of both faces."""
    join = "mitre" if style.angular else "round"
    inset = body.buffer(-style.plate_margin, join_style=join, mitre_limit=2.0)
    if inset.is_empty:
        return []
    minx, miny, maxx, maxy = inset.bounds
    cuts = [box(x - style.groove / 2, miny - 1, x + style.groove / 2, maxy + 1) for x in cut_xs]
    panels = inset.difference(unary_union(cuts)) if cuts else inset
    plates = []
    for poly in _polys(panels):
        # Skip slivers: panels need some size to read as armour.
        if poly.area < 2e-4:
            continue
        if not style.angular:
            poly = orient(poly.buffer(-0.002, join_style="round").buffer(0.002, join_style="round"), 1.0)
            if poly.is_empty or not isinstance(poly, Polygon):
                continue
        ch = min(style.plate_height * 0.6, style.chamfer)
        plates.append(Part(poly, surface_z - 1e-4, surface_z + style.plate_height, ch, "plate"))
        plates.append(Part(poly, -surface_z - style.plate_height, -surface_z + 1e-4, ch, "plate"))
    return plates


# ------------------------------------------------------------ extrusion

def _offset_ring(coords: np.ndarray, dist: float) -> np.ndarray:
    """Move each vertex of a CCW ring inward by `dist` along its miter bisector
    (keeps one-to-one vertex correspondence for the chamfer strip)."""
    prev = np.roll(coords, 1, axis=0)
    nxt = np.roll(coords, -1, axis=0)
    e1 = coords - prev
    e2 = nxt - coords
    n1 = np.stack([-e1[:, 1], e1[:, 0]], 1)
    n2 = np.stack([-e2[:, 1], e2[:, 0]], 1)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-12
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12
    bis = n1 + n2
    bl = np.linalg.norm(bis, axis=1, keepdims=True)
    bis = np.where(bl > 1e-9, bis / (bl + 1e-12), n1)
    cos_half = np.clip((bis * n1).sum(1, keepdims=True), 0.35, 1.0)  # limit spikes on sharp corners
    return coords + bis * (dist / cos_half)


def _ring(ring) -> np.ndarray:
    c = np.asarray(ring.coords)[:-1]
    keep = np.r_[True, np.linalg.norm(np.diff(c, axis=0), axis=1) > 1e-7]
    return c[keep]


class UVLayout:
    """Atlas: front faces (top), back faces (middle), walls & bevels (bottom strip).
    Front/back are planar side-view maps, so the reference projects 1:1."""

    FRONT = (0.0, 0.52, 1.0, 0.98)
    BACK = (0.0, 0.06, 1.0, 0.50)
    WALLS = (0.0, 0.0, 1.0, 0.045)

    def __init__(self, height: float, perimeter_total: float):
        self.h = max(height, 1e-6)
        self.perim = max(perimeter_total, 1e-6)
        self.cursor = 0.0
        self.segments: list[tuple[float, float, np.ndarray, np.ndarray]] = []  # s0, s1, p0, p1

    def face(self, xy: np.ndarray, front: bool) -> np.ndarray:
        u0, v0, u1, v1 = self.FRONT if front else self.BACK
        u = u0 + np.clip(xy[:, 0], 0, 1) * (u1 - u0)
        v = v0 + np.clip(xy[:, 1] / self.h, 0, 1) * (v1 - v0)
        return np.stack([u, v], 1)

    def wall(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        u0, v0, u1, v1 = self.WALLS
        return np.stack([u0 + (s / self.perim) * (u1 - u0), v0 + t * (v1 - v0)], 1)


def _extrude(part: Part, b: _Builder, layout: UVLayout, cap_back: bool = True):
    poly = part.polygon
    c = part.chamfer
    # Chamfer must not exceed half the thinnest feature; shrink it if needed.
    c = min(c, (part.z1 - part.z0) * 0.45)
    rings = [_ring(poly.exterior)] + [_ring(r) for r in poly.interiors]
    rings = [r for r in rings if len(r) >= 3]
    insets = [_offset_ring(r, c) for r in rings]
    # Caps: triangulate the inset outline (front at z1, back at z0).
    try:
        cap_poly = Polygon(insets[0], insets[1:])
        if not cap_poly.is_valid:
            cap_poly = cap_poly.buffer(0)
        caps = _polys(cap_poly)
    except Exception:
        caps = []
    for cp in caps:
        v2, tris = trimesh.creation.triangulate_polygon(cp, engine="earcut")
        if len(tris) == 0:
            continue
        front = np.column_stack([v2, np.full(len(v2), part.z1)])
        b.add(front, layout.face(v2, True), [tuple(t) for t in tris])
        if cap_back:
            back = np.column_stack([v2, np.full(len(v2), part.z0)])
            b.add(back, layout.face(v2, False), [(t[0], t[2], t[1]) for t in tris])
    # Walls: front chamfer, side band, back chamfer.
    zs = [part.z1, part.z1 - c, part.z0 + c, part.z0]
    for outer, inner in zip(rings, insets):
        n = len(outer)
        seg = np.linalg.norm(np.roll(outer, -1, 0) - outer, axis=1)
        s = layout.cursor + np.r_[0, np.cumsum(seg)]
        layout.cursor = s[-1]
        layout.segments += [(s[i], s[i + 1], outer[i], outer[(i + 1) % n]) for i in range(n)]
        loops = [inner, outer, outer, inner]
        for li in range(3):
            pa, pb = loops[li], loops[li + 1]
            za, zb = zs[li], zs[li + 1]
            for i in range(n):
                j = (i + 1) % n
                quad = np.array([[*pa[i], za], [*pa[j], za], [*pb[j], zb], [*pb[i], zb]])
                tvals = np.array([li, li, li + 1, li + 1]) / 3.0
                svals = np.array([s[i], s[i + 1], s[i + 1], s[i]])
                # Outward winding for CCW exterior (CW holes come out right too).
                b.add(quad, layout.wall(svals, tvals), [(0, 3, 2), (0, 2, 1)])


def build(sil: Silhouette, style: HardSurfaceStyle) -> tuple[trimesh.Trimesh, list[Part]]:
    parts = plan_parts(sil, style)
    perim = sum(p.polygon.length for p in parts)
    layout = UVLayout(sil.height, perim)
    b = _Builder()
    for part in parts:
        _extrude(part, b, layout)
    mesh = trimesh.Trimesh(np.array(b.verts, float), np.array(b.faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(b.uvs, float))
    mesh.merge_vertices()  # weld shared corners; UV seams stay split
    mesh.metadata["clean_topology"] = True
    mesh.metadata["wall_segments"] = (layout.perim, layout.segments)
    return mesh, parts


# ---------------------------------------------------------------- texture

def paint_from_reference(mesh: trimesh.Trimesh, sil: Silhouette, reference: np.ndarray,
                         size: int) -> Image.Image:
    """Base texture: front/back regions take the reference colour at each
    point (planar side view); walls take the colour just inside the outline
    at the same point, so edges match the faces they wrap around."""
    rgba = reference.astype(np.float32) / 255
    rgb, alpha = rgba[..., :3], rgba[..., 3] > 0.5
    x0, y0, scale, h_px = sil.to_pixel
    tex = np.zeros((size, size, 3), np.float32)
    layout = UVLayout(sil.height, 1.0)

    def ref_lookup(x, y):
        px = np.clip(np.round(x0 + x * scale).astype(int), 0, rgb.shape[1] - 1)
        py = np.clip(np.round(y0 + h_px - y * scale).astype(int), 0, rgb.shape[0] - 1)
        return px, py

    # Fill colours where the silhouette has holes in alpha (anti-aliased edges):
    # nearest object pixel.
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~alpha, return_distances=False, return_indices=True)
    filled = rgb[idx[0], idx[1]]

    for region, front in ((UVLayout.FRONT, True), (UVLayout.BACK, False)):
        u0, v0, u1, v1 = region
        r0, r1 = int((1 - v1) * size), int((1 - v0) * size)
        c0, c1 = int(u0 * size), int(u1 * size)
        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        vv = 1 - (rows + 0.5) / size
        uu = (cols + 0.5) / size
        x = (uu - u0) / (u1 - u0)
        y = (vv - v0) / (v1 - v0) * layout.h
        X, Y = np.meshgrid(x, y)
        px, py = ref_lookup(X, Y)
        tex[r0:r1, c0:c1] = filled[py, px]

    # Walls: each texel column of the wall strip is a point along some
    # outline; it takes the colour of the reference there.
    perim, segments = mesh.metadata.get("wall_segments", (1.0, []))
    if segments:
        u0, v0, u1, v1 = UVLayout.WALLS
        s0 = np.array([g[0] for g in segments])
        s1 = np.array([g[1] for g in segments])
        p0 = np.array([g[2] for g in segments])
        p1 = np.array([g[3] for g in segments])
        cols = np.arange(int(u0 * size), int(u1 * size))
        sv = ((cols + 0.5) / size - u0) / (u1 - u0) * perim
        k = np.clip(np.searchsorted(s1, sv), 0, len(s1) - 1)
        t = np.clip((sv - s0[k]) / np.maximum(s1[k] - s0[k], 1e-9), 0, 1)[:, None]
        pts = p0[k] * (1 - t) + p1[k] * t
        px, py = ref_lookup(pts[:, 0], pts[:, 1])
        r0, r1 = int((1 - v1) * size), int(np.ceil((1 - v0) * size))
        tex[r0:r1, cols] = filled[py, px][None]
    return Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8))
