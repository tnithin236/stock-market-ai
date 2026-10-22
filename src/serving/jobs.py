"""Tiny in-memory registry for background training jobs (single-process; resets on restart)."""
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from src.serving.errors import TrainingInProgressError

ACTIVE = ("queued", "running")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, ticker: str) -> Dict[str, Any]:
        with self._lock:
            if any(j["ticker"] == ticker and j["status"] in ACTIVE for j in self._jobs.values()):
                raise TrainingInProgressError(f"A training job for {ticker} is already running.")
            job = {"job_id": uuid.uuid4().hex[:12], "ticker": ticker, "status": "queued",
                   "started_at": None, "finished_at": None, "error": None, "result": None}
            self._jobs[job["job_id"]] = job
            return dict(job)

    def run(self, job_id: str, fn: Callable[[], Dict[str, Any]]) -> None:
        """Execute `fn` and record the outcome. Never raises (it runs in a background thread)."""
        with self._lock:
            job = self._jobs[job_id]
            job["status"], job["started_at"] = "running", _now()
        try:
            result = fn()
            with self._lock:
                job["status"], job["result"] = "succeeded", result
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller via status
            with self._lock:
                job["status"], job["error"] = "failed", f"{type(exc).__name__}: {str(exc)[:400]}"
        finally:
            with self._lock:
                job["finished_at"] = _now()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None
