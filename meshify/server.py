"""Local web server: a job queue around the pipeline plus the browser UI.

Jobs run one at a time on a single worker thread, because the models want
the whole GPU.
"""

from __future__ import annotations

import logging
import queue
import shutil
import tempfile
import threading
import traceback
import uuid
from urllib.parse import quote
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .backends import list_backends
from .gpu import blackwell_torch_problem, gpu_info, low_vram
from .config import DEFAULT_BACKEND, OUTPUT_ROOT, GenerationRequest
from .pipeline.run import generate
from .styles import add_reference, delete_reference, list_library, list_packs, load_packs

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Meshify")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
_jobs: dict[str, dict] = {}
_queue: "queue.Queue[tuple[str, GenerationRequest]]" = queue.Queue()


def _worker() -> None:
    while True:
        job_id, req = _queue.get()
        job = _jobs[job_id]
        job["status"] = "running"

        def progress(stage: str, frac: float) -> None:
            job["stage"], job["progress"] = stage, round(frac, 3)

        try:
            result = generate(req, progress=progress, job_id=job_id)
            job.update(status="done", files=result.files, stats=result.stats)
        except Exception as exc:
            log.error("Job %s failed:\n%s", job_id, traceback.format_exc())
            job.update(status="error", error=str(exc))
        finally:
            if req.extra.get("upload_dir"):
                shutil.rmtree(req.extra["upload_dir"], ignore_errors=True)
            _queue.task_done()


threading.Thread(target=_worker, daemon=True, name="meshify-worker").start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/backends")
def backends() -> list[dict]:
    return list_backends()


@app.get("/api/system")
def system() -> dict:
    return {"gpu": gpu_info(), "low_vram": low_vram(), "warning": blackwell_torch_problem()}


@app.get("/api/styles")
def styles() -> list[dict]:
    return list_packs()


@app.get("/api/references")
def references(pack: str, faction: str, category: str = "") -> list[dict]:
    """Style references a job with this faction + asset type would use."""
    try:
        refs = list_library(pack, faction, category or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    for ref in refs:
        ref["url"] = f"/api/references/file?path={quote(ref.pop('path'))}"
    return refs


@app.get("/api/references/file")
def reference_file(path: str) -> FileResponse:
    target = Path(path).resolve()
    # Only serve images that some faction/type actually uses as a reference.
    allowed = {Path(r["path"]).resolve() for r in _all_references()}
    if target not in allowed:
        raise HTTPException(404, "Not a style reference")
    return FileResponse(target)


def _all_references() -> list[dict]:
    out = []
    for pack in list_packs():
        for faction in pack["factions"]:
            for category in [None, *pack["categories"]]:
                out += list_library(pack["id"], faction, category)
    return out


@app.post("/api/references")
async def upload_references(pack: str = Form(...), faction: str = Form(...),
                            category: str = Form(""),
                            files: list[UploadFile] = File(...)) -> list[dict]:
    try:
        for upload in files:
            add_reference(pack, faction, category or None, upload.filename or "reference.png",
                          await upload.read())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    load_packs.cache_clear()
    return references(pack, faction, category)


@app.delete("/api/references")
def remove_reference(pack: str, faction: str, name: str, category: str = "") -> list[dict]:
    try:
        delete_reference(pack, faction, category or None, name)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    load_packs.cache_clear()
    return references(pack, faction, category)


@app.post("/api/jobs")
async def create_job(
    prompt: str = Form(""),
    image: UploadFile | None = File(None),
    backend: str = Form(DEFAULT_BACKEND),
    seed: int = Form(42),
    realism: bool = Form(True),
    delight_strength: float = Form(0.6),
    texture_size: int = Form(0),
    target_faces: str = Form(""),
    real_world_size_m: float = Form(0.0),
    size_axis: str = Form(""),
    style_pack: str = Form(""),
    faction: str = Form(""),
    category: str = Form(""),
    condition: str = Form(""),
    style_images: list[UploadFile] = File(default_factory=list),
    style_strength: float = Form(0.6),
    asset_name: str = Form("asset"),
) -> dict:
    image_path = None
    tmp = Path(tempfile.mkdtemp(prefix="meshify-upload-"))
    if image is not None and image.filename:
        image_path = tmp / ("input" + Path(image.filename).suffix.lower())
        image_path.write_bytes(await image.read())
    style_paths = []
    for i, upload in enumerate(style_images or []):
        if upload.filename:
            path = tmp / f"style_{i}{Path(upload.filename).suffix.lower()}"
            path.write_bytes(await upload.read())
            style_paths.append(str(path))

    try:
        req = GenerationRequest(
            prompt=prompt.strip() or None,
            image_path=str(image_path) if image_path else None,
            backend=backend,
            seed=seed,
            realism=realism,
            delight_strength=delight_strength,
            texture_size=texture_size or None,
            # "" = use the preset; "0" = full density.
            target_faces=int(target_faces) if target_faces.strip() else None,
            real_world_size_m=real_world_size_m or None,
            size_axis=size_axis or None,
            style_pack=style_pack or None,
            faction=faction or None,
            category=category or None,
            condition=condition or None,
            style_image_paths=style_paths,
            style_strength=style_strength,
            asset_name=asset_name,
        )
        req.validate()
    except (ValueError, TypeError) as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(400, str(exc))
    req.extra["upload_dir"] = str(tmp)

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"id": job_id, "status": "queued", "stage": "Queued", "progress": 0.0,
                     "request": {"prompt": req.prompt, "backend": req.backend,
                                 "asset_name": req.asset_name, "faction": req.faction,
                                 "category": req.category, "condition": req.condition}}
    _queue.put((job_id, req))
    return _jobs[job_id]


@app.get("/api/jobs")
def jobs() -> list[dict]:
    return list(reversed(_jobs.values()))


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "No such job")
    return _jobs[job_id]


@app.get("/api/jobs/{job_id}/files/{file_path:path}")
def job_file(job_id: str, file_path: str) -> FileResponse:
    if job_id not in _jobs:
        raise HTTPException(404, "No such job")
    root = (OUTPUT_ROOT / job_id).resolve()
    target = (root / file_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(target, filename=target.name)
