"""Upload validation — exercised through the real HTTP layer."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "hardware" in body
    assert "models" in body


def test_valid_png_upload_creates_session(small_png_bytes):
    res = client.post(
        "/api/upload",
        files={"file": ("photo.png", small_png_bytes, "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"]
    assert body["width"] == 4
    assert body["height"] == 4
    assert body["format"] == "png"


def test_extension_does_not_override_magic_bytes(text_file_bytes):
    """A text file renamed to .png must be rejected."""
    res = client.post(
        "/api/upload",
        files={"file": ("photo.png", text_file_bytes, "image/png")},
    )
    assert res.status_code == 400
    assert "Unsupported or unrecognized" in res.json()["error"]


def test_oversized_upload_rejected(small_png_bytes, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 64)
    res = client.post(
        "/api/upload",
        files={"file": ("photo.png", small_png_bytes, "image/png")},
    )
    assert res.status_code == 413
    assert "upload limit" in res.json()["error"]


def test_cleanup_endpoint_is_idempotent(small_png_bytes):
    res = client.post(
        "/api/upload",
        files={"file": ("photo.png", small_png_bytes, "image/png")},
    )
    sid = res.json()["session_id"]
    first = client.post(f"/api/cleanup/{sid}")
    assert first.status_code == 200
    assert first.json()["existed"] is True
    second = client.post(f"/api/cleanup/{sid}")
    assert second.status_code == 200
    assert second.json()["existed"] is False


def test_process_endpoint_rejects_unknown_session():
    res = client.get("/api/process/nonexistent-session-123456")
    assert res.status_code == 200  # SSE stream that immediately errors
    assert '"type": "error"' in res.text
