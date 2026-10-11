import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from meshify import server


def test_upload_job_runs_and_serves_files(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)
    import meshify.pipeline.run as run
    monkeypatch.setattr(run, "OUTPUT_ROOT", tmp_path)
    original = run.generate
    monkeypatch.setattr(server, "generate",
                        lambda req, **kw: original(req, output_root=tmp_path, **kw))

    client = TestClient(server.app)
    assert any(b["name"] == "dummy" and b["available"] for b in client.get("/api/backends").json())

    img = Image.new("RGB", (200, 200), (120, 120, 120))
    img.paste((150, 90, 40), (60, 40, 140, 170))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = client.post("/api/jobs", data={"backend": "dummy", "texture_size": "512",
                                       "asset_name": "crate"},
                    files={"image": ("crate.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    glb = client.get(f"/api/jobs/{job_id}/files/{job['files']['glb']}")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    assert client.get(f"/api/jobs/{job_id}/files/../../etc/passwd").status_code == 404


def test_rejects_empty_request():
    client = TestClient(server.app)
    assert client.post("/api/jobs", data={"backend": "dummy"}).status_code == 400
