import importlib
import io
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).parent


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VF_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setenv("VF_TOKEN", "s3cret")
    monkeypatch.setenv("VF_PIPELINE_CMD", f"{sys.executable} {HERE / 'fake_pipeline.py'}")
    import vf.server as server
    importlib.reload(server)
    with TestClient(server.app) as c:
        yield c


def wait_done(c, job_id, headers):
    for _ in range(100):
        st = c.get(f"/api/jobs/{job_id}", headers=headers).json()
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise AssertionError("job never finished")


H = {"X-VF-Token": "s3cret"}
PHOTO = ("photos", ("a.jpg", io.BytesIO(b"\xff\xd8fake"), "image/jpeg"))


def test_token_required(client):
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"X-VF-Token": "nope"}).status_code == 401
    assert client.get("/api/jobs?token=s3cret").status_code == 200
    assert client.get("/").status_code == 200  # the page itself is public, the API is not


def test_job_runs_and_only_published_files_are_served(client):
    r = client.post("/api/jobs", headers=H, files=[PHOTO, PHOTO], data={"options": json.dumps({"name": "my room!"})})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    st = wait_done(client, job_id, H)
    assert st["state"] == "done" and st["name"] == "my_room_"
    assert client.get(f"/api/jobs/{job_id}/files/output/scene.html", headers=H).text == "<html>splat</html>"
    assert client.get(f"/api/jobs/{job_id}/files/secret.txt", headers=H).status_code == 404
    assert client.get(f"/api/jobs/{job_id}/files/../../etc/passwd", headers=H).status_code == 404
    assert client.get("/api/jobs/zzz/files/x", headers=H).status_code == 404
    assert [j["id"] for j in client.get("/api/jobs", headers=H).json()] == [job_id]


def test_failed_pipeline_is_reported(client):
    r = client.post("/api/jobs", headers=H, files=[PHOTO], data={"options": json.dumps({"fail": True})})
    st = wait_done(client, r.json()["id"], H)
    assert st["state"] == "error"
    assert "boom" in "\n".join(st["log_tail"])


def test_rejects_bad_uploads(client):
    bad = ("photos", ("x.exe", io.BytesIO(b"MZ"), "application/octet-stream"))
    assert client.post("/api/jobs", headers=H, files=[bad]).status_code == 400
    assert client.post("/api/jobs", headers=H, files=[PHOTO], data={"options": "[1]"}).status_code == 400
