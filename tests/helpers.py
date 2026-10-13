"""Test helpers: a mesh made by extruding a cut-out's silhouette."""
import numpy as np
import trimesh
from PIL import Image


def silhouette_mesh(cutout: Image.Image, cells: int = 120, thickness: float = 0.08) -> trimesh.Trimesh:
    """Front and back faces of the silhouette, with a UV atlas: front in the top
    half, back (mirrored) in the bottom half, empty space around both."""
    a = np.asarray(cutout.convert("RGBA"))[..., 3] > 127
    ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    step = max(h, w) / cells
    gh, gw = int(h / step), int(w / step)
    grid = np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize((gw, gh))) > 127
    verts, faces, uvs = [], [], []
    for front, z, v0 in ((True, thickness / 2, 0.5), (False, -thickness / 2, 0.0)):
        idx = {}

        def vid(i, j):
            if (i, j) not in idx:
                idx[(i, j)] = len(verts)
                verts.append((j / gw * w / max(h, w), -(i / gh) * h / max(h, w), z))
                uvs.append((j / gw if front else 1 - j / gw, v0 + 0.49 * (1 - i / gh)))
            return idx[(i, j)]

        for i in range(gh):
            for j in range(gw):
                if grid[i, j]:
                    q = [vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)]
                    faces += ([(q[0], q[3], q[2]), (q[0], q[2], q[1])] if front
                              else [(q[0], q[1], q[2]), (q[0], q[2], q[3])])
    mesh = trimesh.Trimesh(np.array(verts, float), np.array(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(uvs))
    return mesh
