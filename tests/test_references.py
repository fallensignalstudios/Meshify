import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from meshify import server, styles
from meshify.config import GenerationRequest
from meshify.pipeline import images
from meshify.pipeline.run import generate
from meshify.styles import add_reference, delete_reference, list_library, resolve


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "refs"
    monkeypatch.setenv("MESHIFY_REFERENCE_DIR", str(root))
    styles.load_packs.cache_clear()
    yield root
    styles.load_packs.cache_clear()


def _png(color=(140, 98, 57)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, "PNG")
    return buf.getvalue()


def test_type_references_come_before_faction_references(library):
    add_reference("sovereign_call", "dominion", None, "faction_look.png", _png())
    add_reference("sovereign_call", "dominion", "weapon", "rifle_a.png", _png())
    add_reference("sovereign_call", "dominion", "weapon", "rifle_b.jpg", _png())
    weapon = resolve("sovereign_call", "dominion", "weapon", None).references
    assert [p.name for p in weapon] == ["rifle_a.png", "rifle_b.jpg", "faction_look.png"]
    # Other asset types (and the generic type) get only the faction-wide refs.
    assert [p.name for p in resolve(None, "dominion", "armor", None).references] == ["faction_look.png"]
    assert [p.name for p in resolve(None, "dominion", None, None).references] == ["faction_look.png"]
    assert resolve(None, "reformation", "weapon", None).references == []


def test_references_are_capped(library):
    for i in range(6):
        add_reference("sovereign_call", "eclipse", "weapon", f"r{i}.png", _png())
    assert len(resolve(None, "eclipse", "weapon", None).references) == styles.MAX_STYLE_REFERENCES


def test_library_names_are_safe(library):
    path = add_reference("sovereign_call", "dominion", "weapon", "../../evil name!.png", _png())
    assert path.parent == library / "dominion" / "weapon" and path.name == "evil_name_.png"
    again = add_reference("sovereign_call", "dominion", "weapon", "evil name!.png", _png())
    assert again.name == "evil_name__2.png"  # never overwrites
    with pytest.raises(ValueError):
        add_reference("sovereign_call", "../etc", None, "x.png", _png())
    with pytest.raises(ValueError):
        add_reference("sovereign_call", "dominion", "weapon", "notes.txt", b"hi")
    with pytest.raises(ValueError):
        delete_reference("sovereign_call", "dominion", "weapon", "../../../meshify.json")
    delete_reference("sovereign_call", "dominion", "weapon", "evil_name_.png")
    assert not path.exists()


def test_list_library_reports_scope(library):
    add_reference("sovereign_call", "ancient", None, "ivory.png", _png())
    add_reference("sovereign_call", "ancient", "armor", "helm.png", _png())
    scopes = {r["name"]: r["scope"] for r in list_library("sovereign_call", "ancient", "armor")}
    assert scopes == {"helm.png": "type", "ivory.png": "faction"}


def test_library_references_flow_into_text_jobs(library, tmp_path, monkeypatch):
    add_reference("sovereign_call", "dominion", "weapon", "rifle.png", _png())
    add_reference("sovereign_call", "dominion", None, "look.png", _png())
    seen = {}

    def fake_t2i(prompt, seed, realism, style=None, style_images=None, style_strength=0.6):
        seen["n"] = len(style_images or [])
        img = Image.new("RGB", (400, 300), (128, 128, 128))
        ImageDraw.Draw(img).rectangle([80, 80, 320, 220], fill=(140, 98, 57))
        return img

    monkeypatch.setattr(images, "text_to_image", fake_t2i)
    req = GenerationRequest(prompt="rifle", backend="dummy", faction="dominion",
                            category="weapon", texture_size=512)
    result = generate(req, output_root=tmp_path / "out")
    assert seen["n"] == 2 and result.stats["style_references"] == 2


def test_reference_api_roundtrip(library):
    client = TestClient(server.app)
    r = client.post("/api/references", data={"pack": "sovereign_call", "faction": "dominion",
                                             "category": "weapon"},
                    files=[("files", ("a.png", _png(), "image/png")),
                           ("files", ("b.png", _png((90, 20, 20)), "image/png"))])
    assert r.status_code == 200, r.text
    assert [x["name"] for x in r.json()] == ["a.png", "b.png"]
    img = client.get(r.json()[0]["url"])
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    # Arbitrary files can't be fetched through the reference endpoint.
    assert client.get("/api/references/file", params={"path": "/etc/passwd"}).status_code == 404
    left = client.delete("/api/references", params={"pack": "sovereign_call", "faction": "dominion",
                                                    "category": "weapon", "name": "a.png"})
    assert [x["name"] for x in left.json()] == ["b.png"]
    bad = client.post("/api/references", data={"pack": "sovereign_call", "faction": "nope"},
                      files=[("files", ("a.png", _png(), "image/png"))])
    assert bad.status_code == 400


def test_job_accepts_several_one_off_style_images(library, tmp_path, monkeypatch):
    import meshify.pipeline.run as run

    captured = {}
    original = run.generate

    def fake_generate(req, **kw):
        captured["paths"] = list(req.style_image_paths)
        return original(req, output_root=tmp_path, **kw)

    monkeypatch.setattr(server, "generate", fake_generate)
    client = TestClient(server.app)
    photo = io.BytesIO()
    img = Image.new("RGB", (200, 200), (128, 128, 128))
    img.paste((150, 90, 40), (60, 40, 140, 170))
    img.save(photo, "PNG")
    r = client.post("/api/jobs", data={"backend": "dummy", "texture_size": "512"},
                    files=[("image", ("p.png", photo.getvalue(), "image/png")),
                           ("style_images", ("s1.png", _png(), "image/png")),
                           ("style_images", ("s2.png", _png(), "image/png"))])
    assert r.status_code == 200, r.text
    import time
    for _ in range(100):
        if client.get(f"/api/jobs/{r.json()['id']}").json()["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert len(captured["paths"]) == 2


def test_a_slot_is_kept_for_faction_wide_materials(library):
    for i in range(5):
        add_reference("sovereign_call", "dominion", "weapon", f"rifle{i}.png", _png())
    add_reference("sovereign_call", "dominion", None, "materials.png", _png())
    refs = [p.name for p in resolve(None, "dominion", "weapon", None).references]
    assert len(refs) == styles.MAX_STYLE_REFERENCES and refs[-1] == "materials.png"


def test_bundled_sovereign_call_references(monkeypatch):
    # The repo ships Dominion and Reformation references cut from the style boards.
    monkeypatch.delenv("MESHIFY_REFERENCE_DIR", raising=False)
    styles.load_packs.cache_clear()
    dom = [p.name for p in resolve("sovereign_call", "dominion", "weapon", None).references]
    assert "materials.png" in dom and len(dom) == styles.MAX_STYLE_REFERENCES
    ref = resolve("sovereign_call", "reformation", "armor", None).references
    assert [p.name for p in ref][-1] == "materials.png" and len(ref) == styles.MAX_STYLE_REFERENCES
    for p in ref + resolve("sovereign_call", "dominion", "hero_weapon", None).references:
        assert Image.open(p).size[0] >= 60


def _maps_on(rel, faction):
    from meshify.pipeline import materials
    import numpy as np

    im = Image.open(f"meshify/styles/references/{rel}").convert("RGB")
    alpha = np.asarray(images.remove_background(im))[..., 3] > 128
    maps = materials.build_pbr_maps(im, 0.0, resolve("sovereign_call", faction, None, None).material)
    return np.asarray(im).astype(int), alpha, maps


def test_real_dominion_art_bronze_is_metal_oxblood_is_enamel_amber_glows():
    import numpy as np

    rgb, alpha, maps = _maps_on("dominion/hero_weapon/officer_rifle_target.png", "dominion")
    metal = np.asarray(maps.metallic) / 255
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    oxblood = (r > g * 1.5) & (r > b * 1.5) & (r > 50) & (r < 200) & alpha
    bronze = (r > g * 1.1) & (g > b * 1.15) & (r > 90) & alpha
    assert metal[bronze].mean() > 0.6
    assert metal[oxblood].mean() < 0.2
    assert maps.emissive_mask is not None  # the amber light strips


def test_real_reformation_white_armour_is_not_mistaken_for_glow():
    import numpy as np

    _, alpha, maps = _maps_on("reformation/armor/trooper.png", "reformation")
    glow = np.asarray(maps.emissive_mask) / 255 > 0.5 if maps.emissive_mask else np.zeros(1)
    assert glow[alpha].mean() < 0.05
