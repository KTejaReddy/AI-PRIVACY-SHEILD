"""API endpoints.

    GET  /api/health              -> service + hardware + model status
    GET  /api/models              -> model registry detail
    POST /api/upload              -> validate + stage an upload, returns session_id
    GET  /api/process/{id}        -> SSE stream of pipeline stages + result
    POST /api/cleanup/{id}        -> delete the session's temp files (idempotent)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import threading

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from ..cleanup.manager import get_store
from ..config import settings
from ..models.face_models import get_registry
from ..processing.pipeline import run_pipeline_locked
from ..utils.imaging import load_and_normalize

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict:
    registry = get_registry()
    return {
        "status": "ok",
        "service": "AI Privacy Shield local processing engine",
        "version": "1.0.0",
        "hardware": {
            "device": registry.device,
            "cuda": registry.device == "cuda",
            "gpu_name": _gpu_name(registry.device),
            "note": "GPU acceleration: available" if registry.device == "cuda" else "CPU mode: active",
        },
        "models": registry.describe(),
        "ocr": _ocr_status(),
    }


@router.get("/models")
async def models() -> dict:
    registry = get_registry()
    return registry.describe()


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if len(raw) > settings.MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"error": f"Image exceeds the {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."},
            status_code=413,
        )
    try:
        record = load_and_normalize(raw, len(raw))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected upload validation failure")
        return JSONResponse({"error": "The uploaded file could not be processed."}, status_code=400)

    store = get_store()
    session_id = store.create()
    ext = _extension_for(record.source_format)
    store.save_upload(session_id, raw, extension=ext)

    logger.info("Upload accepted for session %s (%s, %dx%d)", session_id, record.source_format, record.width, record.height)
    return JSONResponse(
        {
            "session_id": session_id,
            "width": record.width,
            "height": record.height,
            "format": record.source_format,
            "size_bytes": record.source_size_bytes,
        },
        status_code=200,
    )


@router.get("/process/{session_id}")
async def process(session_id: str) -> EventSourceResponse:
    store = get_store()
    if not store.exists(session_id):
        return _sse_single({"type": "error", "message": "Session not found or already cleaned up."})

    queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    def emit(ev: dict) -> None:
        try:
            queue.put_nowait(ev)
        except asyncio.QueueFull:
            logger.warning("Event queue full; dropping event.")

    def run() -> None:
        try:
            run_pipeline_locked(session_id, emit)
        except Exception:  # noqa: BLE001 - pipeline already emitted a friendly error
            logger.exception("Processing thread ended with an error.")

    thread = threading.Thread(target=run, daemon=True, name=f"pipeline-{session_id}")
    thread.start()

    async def event_gen():
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if not thread.is_alive() and queue.empty():
                    yield _sse({"type": "error", "message": "Processing stopped unexpectedly."})
                    return
                continue
            yield _sse(ev)
            if ev.get("type") in ("done", "error"):
                # keep draining briefly so no event is lost before close
                while not queue.empty():
                    yield _sse(queue.get_nowait())
                return

    return EventSourceResponse(
        event_gen(),
        ping=15,
        ping_message_factory=lambda: ServerSentEvent(data=json.dumps({"type": "ping"})),
    )


@router.post("/cleanup/{session_id}")
async def cleanup(session_id: str) -> JSONResponse:
    store = get_store()
    existed = store.delete(session_id)
    return JSONResponse({"status": "cleaned", "existed": existed})


# ---------------------------------------------------------------------------


def _extension_for(fmt: str) -> str:
    return {
        "jpeg": ".jpg",
        "png": ".png",
        "webp": ".webp",
        "gif": ".gif",
        "bmp": ".bmp",
        "tiff": ".tiff",
    }.get(fmt, ".bin")


def _gpu_name(device: str) -> str | None:
    if device != "cuda":
        return None
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None


def _ocr_status() -> dict:
    from ..vision.sensitive import ocr_status  # noqa: PLC0415

    return ocr_status()


def _sse(ev: dict) -> ServerSentEvent:
    return ServerSentEvent(data=json.dumps(_json_safe(ev)))


def _json_safe(value):
    """Recursively replace non-finite floats with None (strict JSON).

    Python's ``json.dumps`` happily emits bare ``Infinity``/``NaN`` tokens that
    ``JSON.parse`` in the browser rejects. Guard every SSE payload so a metric
    can never break the result event.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _sse_single(ev: dict) -> EventSourceResponse:
    async def gen():
        yield _sse(ev)

    return EventSourceResponse(gen())
