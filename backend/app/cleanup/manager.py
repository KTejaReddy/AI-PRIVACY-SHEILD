"""Temporary session storage + automatic cleanup.

Uploads live in per-session temp directories under ``backend/.tmp/sessions``.
They are never placed in a database. The pipeline deletes the directory when it
finishes; ``/api/cleanup`` deletes on demand; a janitor thread removes stale
directories left behind by crashes or aborted connections.
"""
from __future__ import annotations

import logging
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from ..config import TMP_DIR, settings

logger = logging.getLogger(__name__)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_SESSION_ROOT = TMP_DIR / "sessions"


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or _SESSION_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_id(session_id: str) -> None:
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError("Invalid session id.")

    def _resolve(self, session_id: str) -> Path:
        self._validate_id(session_id)
        path = (self.root / session_id).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Invalid session path.")
        return path

    # ------------------------------------------------------------------
    def create(self) -> str:
        """Create a new session directory, return its id."""
        session_id = secrets.token_urlsafe(18)
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=False)
        logger.debug("Session %s created", session_id)
        return session_id

    def save_upload(self, session_id: str, raw: bytes, extension: str = ".bin") -> Path:
        """Persist the raw upload inside the session dir (never outside it)."""
        path = self._resolve(session_id) / f"original{extension}"
        with self._lock:
            path.write_bytes(raw)
        return path

    def upload_path(self, session_id: str) -> Path:
        path = self._resolve(session_id)
        for candidate in path.glob("original.*"):
            return candidate
        raise FileNotFoundError("Uploaded file is missing.")

    def exists(self, session_id: str) -> bool:
        try:
            return self._resolve(session_id).is_dir()
        except ValueError:
            return False

    def delete(self, session_id: str) -> bool:
        """Delete a session directory. Idempotent. Returns True if it existed."""
        try:
            path = self._resolve(session_id)
        except ValueError:
            return False
        if not path.is_dir():
            return False
        with self._lock:
            shutil.rmtree(path, ignore_errors=True)
        logger.info("Session %s cleaned up (temporary files removed).", session_id)
        return True

    # ------------------------------------------------------------------
    def cleanup_stale(self, max_age_seconds: int | None = None) -> int:
        """Delete sessions older than the TTL. Returns the count removed."""
        ttl = max_age_seconds or settings.SESSION_TTL_SECONDS
        now = time.time()
        removed = 0
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                mtime = child.stat().st_mtime
                if now - mtime > ttl:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

    def janitor_loop(self, stop_event: threading.Event | None = None) -> None:
        """Background thread that removes stale sessions periodically."""
        while True:
            try:
                if stop_event is not None and stop_event.is_set():
                    break
                removed = self.cleanup_stale()
                if removed:
                    logger.info("Janitor removed %d stale session(s).", removed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Janitor error: %s", exc)
            time.sleep(settings.JANITOR_INTERVAL_SECONDS)


_store: SessionStore | None = None
_store_lock = threading.Lock()


def get_store() -> SessionStore:
    global _store  # noqa: PLW0603
    with _store_lock:
        if _store is None:
            _store = SessionStore()
        return _store
