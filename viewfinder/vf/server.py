"""Job server for the rented GPU box: upload photos, run the pipeline, download results.

    uvicorn vf.server:app --host 127.0.0.1 --port 8000

Bind to localhost and reach it through an SSH tunnel (see README). If you must expose it,
set VF_TOKEN and send it as the X-VF-Token header or ?token=... query parameter.

Jobs run one at a time (one GPU) in a background worker.
"""

from __future__ import annotations

import json
import os
import queue
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

HERE = Path(__file__).resolve().parent
JOBS = Path(os.environ.get("VF_JOBS", Path(os.environ.get("VF_WORK", "/workspace")) / "vf_jobs"))
TOKEN = os.environ.get("VF_TOKEN", "")
# The command that runs one job; overridable for testing.
PIPELINE_CMD = os.environ.get("VF_PIPELINE_CMD", f"{shlex.quote(sys.executable)} -m vf.pipeline")
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
MAX_PHOTOS = 200

work_q: "queue.Queue[str]" = queue.Queue()


def auth(request: Request):
    if not TOKEN:
        return
    given = request.headers.get("x-vf-token") or request.query_params.get("token", "")
    if not secrets.compare_digest(given, TOKEN):
        raise HTTPException(401, "bad or missing token")


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HTTPException(404, "no such job")
    d = JOBS / job_id
    if not d.is_dir():
        raise HTTPException(404, "no such job")
    return d


def read_status(d: Path) -> dict:
    try:
        st = json.loads((d / "status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        st = {"state": "queued", "stage": None, "done": [], "outputs": []}
    opts = json.loads((d / "job.json").read_text())
    return {"id": d.name, "name": opts.get("name"), "created": opts.get("created"), **st}


def worker():
    while True:
        job_id = work_q.get()
        d = JOBS / job_id
        with open(d / "log.txt", "a") as log:
            proc = subprocess.run(
                f"{PIPELINE_CMD} {shlex.quote(str(d))}", shell=True, cwd=HERE.parent,
                stdout=log, stderr=subprocess.STDOUT,
            )
        st = read_status(d)
        if proc.returncode != 0 and st.get("state") != "error":
            (d / "status.json").write_text(json.dumps(
                {**st, "state": "error", "message": f"pipeline exited with {proc.returncode}"}))
        work_q.task_done()


@asynccontextmanager
async def lifespan(_app):
    JOBS.mkdir(parents=True, exist_ok=True)
    # Re-queue jobs that never finished (e.g. the pod restarted).
    for d in sorted(JOBS.iterdir(), key=lambda p: p.stat().st_mtime):
        if (d / "job.json").exists() and read_status(d)["state"] in ("queued", "running"):
            work_q.put(d.name)
    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title="viewfinder-open", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text()


@app.post("/api/jobs", dependencies=[Depends(auth)])
async def create_job(photos: list[UploadFile] = File(...), options: str = Form("{}")):
    try:
        opts = json.loads(options)
        assert isinstance(opts, dict)
    except Exception:
        raise HTTPException(400, "options must be a JSON object")
    if not 1 <= len(photos) <= MAX_PHOTOS:
        raise HTTPException(400, f"send 1-{MAX_PHOTOS} photos")
    exts = [Path(f.filename or "").suffix.lower() for f in photos]
    bad = sorted({e for e in exts if e not in ALLOWED})
    if bad:
        raise HTTPException(400, f"unsupported file type(s): {bad}")
    job_id = secrets.token_hex(6)
    d = JOBS / job_id
    (d / "input").mkdir(parents=True)
    for i, (f, ext) in enumerate(zip(photos, exts)):
        (d / "input" / f"{i:04d}{ext}").write_bytes(await f.read())
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(opts.get("name") or "scene"))[:40] or "scene"
    opts.update(name=name, created=time.time())
    (d / "job.json").write_text(json.dumps(opts, indent=1))
    work_q.put(job_id)
    return {"id": job_id, "queue_position": work_q.qsize()}


@app.get("/api/jobs", dependencies=[Depends(auth)])
def list_jobs():
    ds = [d for d in JOBS.iterdir() if (d / "job.json").exists()]
    return sorted((read_status(d) for d in ds), key=lambda s: s.get("created") or 0, reverse=True)


@app.get("/api/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str):
    d = job_dir(job_id)
    st = read_status(d)
    log = d / "log.txt"
    st["log_tail"] = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
    return st


@app.get("/api/jobs/{job_id}/files/{rel:path}", dependencies=[Depends(auth)])
def get_file(job_id: str, rel: str):
    d = job_dir(job_id)
    st = read_status(d)
    if rel not in st.get("outputs", []):  # only files the pipeline published
        raise HTTPException(404, "no such output")
    path = (d / rel).resolve()
    if d.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "no such output")
    return FileResponse(path, filename=path.name if path.suffix in (".zip", ".spz", ".sog", ".ply", ".glb") else None)
