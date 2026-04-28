from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from service.schemas import utc_now
from service.settings import ServiceSettings


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _processes() -> list[tuple[int, list[str]]]:
    processes: list[tuple[int, list[str]]] = []
    proc_root = Path("/proc")

    if not proc_root.exists():
        return processes

    for cmdline_path in proc_root.glob("[0-9]*/cmdline"):
        try:
            pid = int(cmdline_path.parent.name)
            raw_cmdline = cmdline_path.read_bytes()
        except (OSError, ValueError):
            continue

        if not raw_cmdline:
            continue

        command = [part.decode("utf-8", errors="ignore") for part in raw_cmdline.split(b"\0") if part]
        if command:
            processes.append((pid, command))

    return processes


def _terminate_process(pid: int, *, timeout: float = 4.0) -> None:
    if pid <= 0 or pid == os.getpid():
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)

    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _command_for_pid(pid: int) -> list[str]:
    try:
        raw_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="ignore") for part in raw_cmdline.split(b"\0") if part]


class DroneWorkerProcess:
    def __init__(
        self,
        *,
        settings: ServiceSettings,
        pipeline: dict[str, object],
    ) -> None:
        self.settings = settings
        self._pipeline = dict(pipeline)
        self.drone_id = str(pipeline.get("droneId") or "").strip()
        self.worker_dir = settings.runtime_dir / "workers" / self.drone_id
        self.pipeline_path = self.worker_dir / "pipeline.json"
        self.runtime_path = self.worker_dir / "runtime.json"
        self.processed_frame_path = self.worker_dir / "processed.jpg"
        self.raw_frame_path = self.worker_dir / "raw.jpg"
        self.pid_path = self.worker_dir / "worker.pid"
        self.log_path = self.worker_dir / "worker.log"
        self._process: subprocess.Popen[bytes] | None = None
        self._started_at = ""

    def start(self) -> None:
        if self.is_running():
            return

        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self._stop_duplicate_processes()
        self._write_pipeline()
        self._clear_transient_files()

        log_stream = self.log_path.open("ab", buffering=0)
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "service.worker_main",
                    "--drone-id",
                    self.drone_id,
                    "--worker-dir",
                    str(self.worker_dir),
                ],
                cwd=str(self.settings.repo_dir),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_stream.close()
        self._started_at = utc_now()
        self.pid_path.write_text(f"{self._process.pid}\n", encoding="utf-8")
        self._write_runtime_overlay(
            {
                "status": "starting",
                "message": "dedicated worker process starting",
                "workerProcessPid": self._process.pid,
                "workerProcessRunning": True,
                "workerProcessMode": "process",
                "workerProcessStartedAt": self._started_at,
                "workerProcessLogPath": str(self.log_path),
            }
        )

    def stop(self, timeout: float = 5.0) -> None:
        pid = self.pid
        process = self._process

        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        elif pid:
            _terminate_process(pid, timeout=timeout)

        self._process = None
        self._write_runtime_overlay(
            {
                "status": "stopped",
                "message": "dedicated worker process stopped",
                "sourceOpened": False,
                "workerProcessPid": None,
                "workerProcessRunning": False,
                "workerProcessMode": "process",
            }
        )

    def update_pipeline(self, pipeline: dict[str, object]) -> None:
        self._pipeline = dict(pipeline)
        self._write_pipeline()
        if not self.is_running():
            self.start()

    def runtime_snapshot(self) -> dict[str, object]:
        runtime = self._runtime_with_defaults()
        pid = self.pid
        running = self.is_running()

        runtime.update(
            {
                "workerProcessPid": pid if running else None,
                "workerProcessRunning": running,
                "workerProcessMode": "process",
                "workerProcessStartedAt": runtime.get("workerProcessStartedAt") or self._started_at,
                "workerProcessLogPath": str(self.log_path),
            }
        )

        if not running and runtime.get("status") not in {"stopped", "disabled"}:
            runtime.update(
                {
                    "status": "crashed",
                    "message": "dedicated worker process is not running",
                    "sourceOpened": False,
                    "processedStreamReady": False,
                    "processedPublisherPid": None,
                }
            )

        return runtime

    def latest_processed_frame(self) -> bytes | None:
        return _read_bytes(self.processed_frame_path)

    def latest_raw_frame(self) -> bytes | None:
        return _read_bytes(self.raw_frame_path)

    @property
    def pid(self) -> int | None:
        if self._process and self._process.poll() is None:
            return self._process.pid
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def is_running(self) -> bool:
        if self._process and self._process.poll() is None:
            return True
        pid = self.pid
        return bool(pid and _is_pid_running(pid) and self._pid_matches_worker(pid))

    def _write_pipeline(self) -> None:
        _write_json_atomic(self.pipeline_path, self._pipeline)

    def _write_runtime_overlay(self, overlay: dict[str, object]) -> None:
        runtime = self._runtime_with_defaults()
        runtime.update(overlay)
        runtime["workerProcessUpdatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_json_atomic(self.runtime_path, runtime)

    def _runtime_with_defaults(self) -> dict[str, object]:
        runtime = _read_json(self.runtime_path)
        defaults = {
            "status": "configured",
            "message": "worker process not started",
            "sourceOpened": False,
            "sourceType": "unknown",
            "sourceUrl": str(self._pipeline.get("rtspUrl") or ""),
            "captureBackend": "",
            "capturePid": None,
            "activeRtspConnections": 0,
            "maxRtspConnections": 1,
            "singleIngestHealthy": True,
            "sourceSessionId": "",
            "sourceOpenCount": 0,
            "sourceReconnectCount": 0,
            "lastSourceOpenAt": "",
            "lastSourceCloseAt": "",
            "lastSourceError": "",
            "frameWidth": None,
            "frameHeight": None,
            "sourceFps": None,
            "processingFps": 0.0,
            "framesProcessed": 0,
            "detectionsTotal": 0,
            "avgInferenceMs": None,
            "lastFrameAt": "",
            "lastDetectionAt": "",
            "currentEvent": None,
            "lastRecording": None,
            "latestFrameAvailable": False,
            "latestFrameContentType": "image/jpeg",
            "latestProcessedFrameAvailable": self.processed_frame_path.exists(),
            "latestProcessedFrameContentType": "image/jpeg",
            "latestRawFrameAvailable": self.raw_frame_path.exists(),
            "latestRawFrameContentType": "image/jpeg",
            "processedStreamReady": False,
            "processedStreamUrl": self.settings.processed_publish_url(self.drone_id),
            "processedPublisherPid": None,
            "processedStreamRevision": 0,
            "processedStreamStartedAt": "",
            "modelId": str(self._pipeline.get("currentModelId") or ""),
            "modelName": "",
        }
        return {**defaults, **runtime}

    def _clear_transient_files(self) -> None:
        for path in (self.processed_frame_path, self.raw_frame_path):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue

    def _stop_duplicate_processes(self) -> None:
        worker_dir_token = str(self.worker_dir)
        drone_token = self.drone_id
        for pid, command in _processes():
            if pid == os.getpid():
                continue
            if "service.worker_main" not in command:
                continue
            command_text = "\0".join(command)
            if worker_dir_token not in command_text and f"--drone-id\0{drone_token}" not in command_text:
                continue
            _terminate_process(pid)

    def _pid_matches_worker(self, pid: int) -> bool:
        command = _command_for_pid(pid)
        if "service.worker_main" not in command:
            return False
        command_text = "\0".join(command)
        return str(self.worker_dir) in command_text or f"--drone-id\0{self.drone_id}" in command_text


def terminate_drone_worker_process(
    settings: ServiceSettings,
    drone_id: str,
    *,
    timeout: float = 5.0,
) -> None:
    safe_drone_id = str(drone_id or "").strip()
    if not safe_drone_id:
        return

    worker_dir = settings.runtime_dir / "workers" / safe_drone_id
    worker_dir_token = str(worker_dir)
    pid_path = worker_dir / "worker.pid"

    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pid = 0

    if pid:
        command_text = "\0".join(_command_for_pid(pid))
        if "service.worker_main" in command_text and (
            worker_dir_token in command_text or f"--drone-id\0{safe_drone_id}" in command_text
        ):
            _terminate_process(pid, timeout=timeout)

    for candidate_pid, command in _processes():
        if candidate_pid == os.getpid():
            continue
        if "service.worker_main" not in command:
            continue
        command_text = "\0".join(command)
        if worker_dir_token not in command_text and f"--drone-id\0{safe_drone_id}" not in command_text:
            continue
        _terminate_process(candidate_pid, timeout=timeout)
