"""AI Privacy Shield — local processing backend.

Run from the ``backend`` directory:

    .venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Everything runs locally on this machine. User images are held only in
per-session temporary directories and deleted after processing.
"""
from __future__ import annotations

import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .cleanup.manager import get_store
from .config import ensure_dirs, settings

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("aips")

app = FastAPI(
    title="AI Privacy Shield — local processing engine",
    description=(
        "Local-first privacy image protection: face detection, multi-model adversarial "
        "protection, robustness testing, metadata sanitization, automatic cleanup."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

_stop_janitor = threading.Event()


@app.on_event("startup")
async def startup() -> None:
    ensure_dirs()

    # background janitor for stale temp sessions
    janitor = threading.Thread(
        target=get_store().janitor_loop,
        args=(_stop_janitor,),
        daemon=True,
        name="janitor",
    )
    janitor.start()
    logger.info("Janitor started (TTL %ds).", settings.SESSION_TTL_SECONDS)

    # preload models in the background so the first request is fast
    def _preload() -> None:
        try:
            from .models.face_models import load_models  # noqa: PLC0415

            registry = load_models()
            logger.info(
                "Model preload complete: %s",
                ", ".join(f"{m.info.id}={m.info.loaded}" for m in registry.optimization_models),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Model preload failed (will retry lazily on first request).")

    threading.Thread(target=_preload, daemon=True, name="model-preload").start()


@app.on_event("shutdown")
async def shutdown() -> None:
    _stop_janitor.set()


@app.get("/")
async def root() -> dict:
    return {
        "service": "AI Privacy Shield — local processing engine",
        "status": "running",
        "privacy": "Images are not permanently stored. Temporary files are deleted after processing.",
    }
