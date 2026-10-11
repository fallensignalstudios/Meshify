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
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .backends import list_backends
from .config import DEFAULT_BACKEND, OUTPUT_ROOT, GenerationRequest
from .pipeline.run import generate

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
            if req.image_path and Path(req.image_path).parent.name.startswith("meshify-upload-"):
                shutil.rmtree(Path(req.image_path).parent, ignore_errors=True)
            _queue.task_done()


threading.Thread(target=_worker, daemon=True, name="meshify-worker").start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/backends")
def backends() -> list[dict]:
    return list_backends()


@app.post("/api/jobs")
async def create_job(
    prompt: str = Form(""),
    image: UploadFile | None = File(None),
    backend: str = Form(DEFAULT_BACKEND),
    seed: int = Form(42),
    realism: bool = Form(True),
    delight_strength: float = Form(0.6),
    texture_size: int = Form(2048),
    target_faces: int = Form(50_000),
    real_world_height_m: float = Form(0.0),
    asset_name: str = Form("asset"),
) -> dict:
    image_path = None
    if image is not None and image.filename:
        tmp = Path(tempfile.mkdtemp(prefix="meshify-upload-"))
        image_path = tmp / ("input" + Path(image.filename).suffix.lower())
        image_path.write_bytes(await image.read())

    req = GenerationRequest(
        prompt=prompt.strip() or None,
        image_path=str(image_path) if image_path else None,
        backend=backend,
        seed=seed,
        realism=realism,
        delight_strength=delight_strength,
        texture_size=texture_size,
        target_faces=target_faces or None,
        real_world_height_m=real_world_height_m or None,
        asset_name=asset_name,
    )
    try:
        req.validate()
    except ValueError as exc:
        if image_path:
            shutil.rmtree(image_path.parent, ignore_errors=True)
        raise HTTPException(400, str(exc))

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"id": job_id, "status": "queued", "stage": "Queued", "progress": 0.0,
                     "request": {"prompt": req.prompt, "backend": req.backend,
                                 "asset_name": req.asset_name}}
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
