import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from meshify.config import GenerationRequest
from meshify.pipeline import images, materials
from meshify.pipeline.run import generate


@pytest.fixture
def studio_photo(tmp_path):
    """A brown 'object' with a shading gradient on a flat grey background."""
    img = Image.new("RGB", (400, 400), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    for i in range(160):
        shade = 60 + i
        draw.rectangle([120 + i // 2, 80, 280, 320 - i // 4], fill=(shade, int(shade * 0.7), 40))
    path = tmp_path / "photo.png"
    img.save(path)
    return path


def test_realism_prompt_is_added_but_respects_explicit_style():
    assert "photorealistic" in images.build_prompt("oak barrel", realism=True)
    assert images.build_prompt("stylized cartoon barrel", realism=True) == "stylized cartoon barrel"
    assert images.build_prompt("oak barrel", realism=False) == "oak barrel"


def test_background_removal_and_framing(studio_photo):
    cut = images.remove_background(Image.open(studio_photo))
    alpha = np.asarray(cut)[..., 3]
    assert alpha[0, 0] == 0 and alpha[200, 200] == 255
    framed = images.frame_object(cut)
    assert framed.size == (518, 518)


def test_delight_flattens_shading():
    x = np.linspace(0.2, 0.9, 256, dtype=np.float32)
    rgb = np.dstack([np.tile(x, (256, 1))] * 3) * np.array([1.0, 0.7, 0.4], dtype=np.float32)
    flat = materials.delight(rgb, 1.0)
    assert flat[..., 0].std() < rgb[..., 0].std() * 0.6


def test_pbr_maps_shapes_and_orm_layout():
    tex = Image.fromarray((np.random.default_rng(0).random((128, 128, 3)) * 255).astype(np.uint8))
    maps = materials.build_pbr_maps(tex, 0.5)
    orm = np.asarray(maps.orm)
    assert orm.shape == (128, 128, 3)
    assert orm[..., 2].max() == 0  # metallic channel defaults to dielectric
    assert np.array_equal(np.asarray(maps.roughness), orm[..., 1])
    gl, dx = np.asarray(maps.normal_gl).astype(int), np.asarray(maps.normal_dx).astype(int)
    assert np.abs(gl[..., 1] + dx[..., 1] - 255).max() <= 1  # green channel flipped for Unreal


def test_end_to_end_with_dummy_backend(studio_photo, tmp_path):
    req = GenerationRequest(image_path=str(studio_photo), backend="dummy", texture_size=512,
                            real_world_size_m=1.2, asset_name="Test Barrel")
    result = generate(req, output_root=tmp_path / "out")

    for key in ["glb", "obj", "albedo", "normal_dx", "orm", "reference", "cutout"]:
        assert (result.output_dir / result.files[key]).is_file(), key
    assert result.files["glb"] == "SM_Test_Barrel.glb"

    scene = trimesh.load(result.output_dir / result.files["glb"])
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    assert mesh.extents[1] == pytest.approx(1.2, rel=1e-3)  # real-world height
    assert mesh.bounds[0][1] == pytest.approx(0.0, abs=1e-6)  # sits on the ground
    material = mesh.visual.material
    assert material.normalTexture is not None
    assert material.metallicRoughnessTexture is not None


def test_request_validation():
    with pytest.raises(ValueError):
        GenerationRequest().validate()
    with pytest.raises(ValueError):
        GenerationRequest(prompt="x", texture_size=300).validate()
