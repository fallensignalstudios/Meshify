import numpy as np
import trimesh
from PIL import Image, ImageDraw

from meshify.pipeline import projection

# A thin slab (long in X, tall in Y, thin in Z), like a weapon seen side-on.
EXT = np.array([2.0, 0.6, 0.12])
# Atlas layout: front face -> top half, back face -> bottom half, edges -> a thin strip.
FACES = {  # name: (axis, sign, uv rect (u0, v0, u1, v1))
    "front": (2, +1, (0.0, 0.52, 1.0, 1.0)),
    "back": (2, -1, (0.0, 0.04, 1.0, 0.48)),
    "top": (1, +1, (0.0, 0.0, 0.25, 0.03)),
    "bottom": (1, -1, (0.25, 0.0, 0.5, 0.03)),
    "left": (0, -1, (0.5, 0.0, 0.75, 0.03)),
    "right": (0, +1, (0.75, 0.0, 1.0, 0.03)),
}


def slab() -> trimesh.Trimesh:
    verts, faces, uvs = [], [], []
    for axis, sign, (u0, v0, u1, v1) in FACES.values():
        a, b = [i for i in range(3) if i != axis]
        corners = []
        for ca, cb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.zeros(3)
            p[axis] = sign * EXT[axis] / 2
            p[a], p[b] = ca * EXT[a] / 2, cb * EXT[b] / 2
            corners.append(p)
        base = len(verts)
        verts += corners
        # UV: map the face's (a, b) extent onto the rect.
        for p in corners:
            ta = (p[a] / EXT[a]) + 0.5
            tb = (p[b] / EXT[b]) + 0.5
            uvs.append((u0 + ta * (u1 - u0), v0 + tb * (v1 - v0)))
        tri = [(0, 1, 2), (0, 2, 3)]
        n = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        if np.sign(n[axis]) != sign:
            tri = [(0, 2, 1), (0, 3, 2)]
        faces += [[base + i for i in t] for t in tri]
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(uvs))
    return mesh


def reference() -> Image.Image:
    """Side view: red (left) to blue (right) gradient, with a bright amber strip."""
    img = Image.new("RGBA", (1000, 400), (0, 0, 0, 0))
    arr = np.zeros((400, 1000, 4), np.uint8)
    x0, x1, y0, y1 = 100, 900, 80, 320  # 800 x 240 => aspect 2.0 / 0.6
    t = np.linspace(0, 1, x1 - x0)
    arr[y0:y1, x0:x1, 0] = (220 * (1 - t)).astype(np.uint8)
    arr[y0:y1, x0:x1, 2] = (220 * t).astype(np.uint8)
    arr[y0:y1, x0:x1, 3] = 255
    arr[190:210, 300:700, :3] = (255, 200, 80)  # glow strip
    return Image.fromarray(arr, "RGBA")


def test_view_is_found_from_the_silhouette():
    iou, found, _ = projection.find_view(slab(), np.asarray(reference())[..., 3] > 127)
    assert iou > 0.9 and found[0][0] == 2  # looking along the thin (Z) axis


def test_front_and_back_texels_take_reference_colours():
    ref = reference()
    proj = projection.project(slab(), (512, 512), ref)
    assert proj is not None and proj.iou > 0.9
    colours = proj.sample(np.asarray(ref)[..., :3]).astype(int)

    def texel(u, v):  # u,v in the atlas -> (row, col)
        return int((1 - v) * 511), int(u * 511)

    for v in (0.76, 0.26):  # front (top half) and back (bottom half)
        left, right = texel(0.1, v), texel(0.9, v)
        assert proj.weight[left] > 0.9 and proj.weight[right] > 0.9
        assert colours[left][0] > colours[left][2]   # reddish on the left...
        assert colours[right][2] > colours[right][0]  # ...bluish on the right
    # Edges are seen side-on: they keep the baked texture.
    assert proj.weight[texel(0.1, 0.015)] < 0.1


def test_glow_mask_projects_onto_the_strip():
    ref = reference()
    glow = np.zeros((400, 1000), np.float32)
    glow[190:210, 300:700] = 1.0
    proj = projection.project(slab(), (512, 512), ref)
    tex_glow = proj.sample(glow) * (proj.weight > 0.5)
    assert 0.002 < tex_glow.mean() < 0.05
    row, col = int((1 - 0.76) * 511), int(0.5 * 511)  # front-face centre: inside the strip
    assert tex_glow[row, col] == 1.0


def test_mismatched_mesh_is_not_projected():
    ball = trimesh.creation.icosphere(subdivisions=3)
    ball.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(0).random((len(ball.vertices), 2)))
    long_thin = Image.new("RGBA", (1000, 200), (0, 0, 0, 0))
    ImageDraw.Draw(long_thin).rectangle([50, 90, 950, 110], fill=(200, 100, 50, 255))
    assert projection.project(ball, (256, 256), long_thin) is None


def test_projection_restores_glow_that_the_bake_painted_over():
    """The real failure: the texture painter turns the amber channel into dull
    paint. Projection finds the glow on the reference and puts it back."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from helpers import silhouette_mesh
    from meshify.pipeline import images, materials
    from meshify.styles import resolve

    ref = Image.open("meshify/styles/references/dominion/hero_weapon/officer_rifle_target.png").convert("RGB")
    cutout = images.remove_background(ref)
    mesh = silhouette_mesh(cutout)
    proj = projection.project(mesh, (768, 768), cutout)
    assert proj is not None and proj.iou > 0.8

    # Fake bake: reference colours by UV, darker, glow painted over in dull orange.
    rgba = np.asarray(cutout.convert("RGBA"), dtype=np.float32) / 255
    tex = proj.sample(rgba[..., :3]) * (proj.weight[..., None] > 0.05)
    hue_hot = (tex.max(-1) > 0.75) & (tex[..., 0] > 1.4 * tex[..., 2])
    tex[hue_hot] = (0.42, 0.22, 0.10)
    baked = Image.fromarray((np.clip(0.6 * tex, 0, 1) * 255).astype(np.uint8))

    style = resolve("sovereign_call", "dominion", "weapon", "maintained").material
    plain = materials.build_pbr_maps(baked, 0.6, style, reference=cutout)
    projected = materials.build_pbr_maps(baked, 0.6, style, reference=cutout, projection=proj)
    assert projected.emissive_mask is not None
    assert projected.stats["emissive_coverage"] > plain.stats["emissive_coverage"]
    assert projected.stats["projected"] > 0.2
