from __future__ import annotations

from datetime import datetime
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
from service.processed_publisher import sanitize_stream_url


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        tmp_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


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


def _safe_int(value: object, fallback = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: object, fallback = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _timestamp_to_epoch(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


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
        self.detections_path = self.worker_dir / "detections.latest.json"
        self.buffer_dir = self.worker_dir / "buffer"
        self.pid_path = self.worker_dir / "worker.pid"
        self.log_path = self.worker_dir / "worker.log"
        self._process: subprocess.Popen[bytes] | None = None
        self._started_at = ""

    def start(self) -> None:
        running_pid = self.pid if self.is_running() else None
        if running_pid:
            self._stop_duplicate_processes(keep_pid=running_pid)
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
        running_pid = self.pid if self.is_running() else None
        if running_pid:
            self._stop_duplicate_processes(keep_pid=running_pid)
        else:
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
        if not self.settings.legacy_mjpeg_enabled:
            return None
        return self._read_fresh_frame(
            self.processed_frame_path,
            available_key="latestProcessedFrameAvailable",
        )

    def latest_raw_frame(self) -> bytes | None:
        if not self.settings.legacy_mjpeg_enabled:
            return None
        return self._read_fresh_frame(
            self.raw_frame_path,
            available_key="latestRawFrameAvailable",
        )

    def latest_detections(self) -> dict[str, object]:
        if not self.is_running():
            return self._empty_detections()

        runtime = self._runtime_with_defaults()
        if not runtime.get("sourceOpened") or str(runtime.get("status") or "") not in {"running", "starting"}:
            return self._empty_detections(runtime=runtime)

        payload = _read_json(self.detections_path)
        if not payload:
            return self._empty_detections(runtime=runtime)

        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        frame_at = str(payload.get("frameAt") or "")
        last_frame_at = str(runtime.get("lastFrameAt") or "")
        stale_after = max(
            self.settings.rtsp_read_timeout_seconds * 3.0,
            self.settings.reconnect_delay_seconds * 4.0,
            12.0,
        )
        frame_epoch = _timestamp_to_epoch(frame_at or last_frame_at)
        if frame_epoch and time.time() - frame_epoch > stale_after:
            return self._empty_detections(runtime=runtime)

        return {
            "droneId": self.drone_id,
            "frameAt": frame_at,
            "modelId": str(payload.get("modelId") or runtime.get("modelId") or ""),
            "modelName": str(payload.get("modelName") or runtime.get("modelName") or ""),
            "sensorType": str(payload.get("sensorType") or self._pipeline.get("sensorType") or "unknown"),
            "frameWidth": payload.get("frameWidth"),
            "frameHeight": payload.get("frameHeight"),
            "items": [item for item in items if isinstance(item, dict)],
        }

    def buffered_frames(
        self,
        *,
        variant: str,
        delay_seconds: float,
        after_seq: int = 0,
        limit: int = 4,
    ) -> list[tuple[int, bytes]]:
        if not self.settings.legacy_mjpeg_enabled:
            return []
        if not self.is_running():
            return []

        runtime = self._runtime_with_defaults()
        if not runtime.get("sourceOpened") or str(runtime.get("status") or "") not in {"running", "starting"}:
            return []

        stale_after = max(
            self.settings.rtsp_read_timeout_seconds * 3.0,
            self.settings.reconnect_delay_seconds * 4.0,
            12.0,
        )
        last_frame_epoch = _timestamp_to_epoch(runtime.get("lastFrameAt"))
        if last_frame_epoch and time.time() - last_frame_epoch > stale_after:
            return []

        manifest = _read_json(self.buffer_dir / f"{variant}.json")
        raw_items = manifest.get("items") if isinstance(manifest.get("items"), list) else []
        cutoff = time.time() - max(0.0, delay_seconds)
        entries = [
            item
            for item in raw_items
            if isinstance(item, dict)
            and _safe_float(item.get("createdAt")) <= cutoff
        ]
        entries.sort(key=lambda item: _safe_int(item.get("seq")))

        if after_seq and entries and _safe_int(entries[-1].get("seq")) <= after_seq:
            after_seq = 0

        frames: list[tuple[int, bytes]] = []
        for item in entries:
            sequence = _safe_int(item.get("seq"))
            if after_seq and sequence <= after_seq:
                continue

            relative_path = str(item.get("file") or "")
            if not relative_path or relative_path.startswith("/") or ".." in Path(relative_path).parts:
                continue

            frame = _read_bytes(self.buffer_dir / relative_path)
            if not frame:
                continue

            frames.append((sequence, frame))
            if len(frames) >= max(1, limit):
                break

        return frames

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
        runtime["workerProcessUpdatedAt"] = utc_now()
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
            "framesCaptured": 0,
            "framesDroppedBeforeInference": 0,
            "captureQueueCapacity": 1,
            "captureQueueDepth": 0,
            "lastCapturedFrameAt": "",
            "lastInferenceFrameAgeMs": None,
            "avgInferenceFrameAgeMs": None,
            "maxInferenceFrameAgeMs": None,
            "framesProcessed": 0,
            "detectionsTotal": 0,
            "avgInferenceMs": None,
            "lastFrameAt": "",
            "lastDetectionAt": "",
            "currentEvent": None,
            "lastRecording": None,
            "latestFrameAvailable": False,
            "latestFrameContentType": "image/jpeg",
            "latestProcessedFrameAvailable": (
                self.settings.legacy_mjpeg_enabled and self.processed_frame_path.exists()
            ),
            "latestProcessedFrameContentType": "image/jpeg",
            "latestRawFrameAvailable": (
                self.settings.legacy_mjpeg_enabled and self.raw_frame_path.exists()
            ),
            "latestRawFrameContentType": "image/jpeg",
            "processedStreamReady": False,
            "processedStreamUrl": (
                f"/v1/pipelines/{self.drone_id}/stream.mjpg"
                if self.settings.legacy_mjpeg_enabled
                else ""
            ),
            "processedPublisherPid": None,
            "processedPublisherTransport": self.settings.processed_publish_transport,
            "processedPublisherUrl": sanitize_stream_url(
                self.settings.processed_publish_url(
                    self.drone_id,
                    str(self._pipeline.get("rtspUrl") or ""),
                )
            ),
            "processedPublisherError": "",
            "processedPublisherLastFailureReason": "",
            "processedPublisherLastFailureAt": "",
            "processedPublisherFailureCount": 0,
            "processedPublisherRestartCount": 0,
            "processedPublisherFramesWritten": 0,
            "processedPublisherDroppedFrames": 0,
            "processedPublisherStable": False,
            "processedPublisherUptimeSeconds": 0.0,
            "legacyMjpegEnabled": self.settings.legacy_mjpeg_enabled,
            "processedStreamRevision": 0,
            "processedStreamStartedAt": "",
            "modelId": str(self._pipeline.get("currentModelId") or ""),
            "modelName": "",
        }
        return {**defaults, **runtime}

    def _clear_transient_files(self) -> None:
        for path in (self.processed_frame_path, self.raw_frame_path, self.detections_path):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue

    def _empty_detections(self, *, runtime: dict[str, object] | None = None) -> dict[str, object]:
        runtime = runtime or self._runtime_with_defaults()
        return {
            "droneId": self.drone_id,
            "frameAt": str(runtime.get("lastFrameAt") or ""),
            "modelId": str(runtime.get("modelId") or self._pipeline.get("currentModelId") or ""),
            "modelName": str(runtime.get("modelName") or ""),
            "sensorType": str(self._pipeline.get("sensorType") or "unknown"),
            "frameWidth": runtime.get("frameWidth"),
            "frameHeight": runtime.get("frameHeight"),
            "items": [],
        }

    def _read_fresh_frame(self, path: Path, *, available_key: str) -> bytes | None:
        if not self.is_running():
            return None

        runtime = self._runtime_with_defaults()
        if not runtime.get(available_key):
            return None

        if not runtime.get("sourceOpened"):
            return None

        status = str(runtime.get("status") or "")
        if status not in {"running", "starting"}:
            return None

        stale_after = max(
            self.settings.rtsp_read_timeout_seconds * 3.0,
            self.settings.reconnect_delay_seconds * 4.0,
            12.0,
        )
        last_frame_epoch = _timestamp_to_epoch(runtime.get("lastFrameAt"))

        if last_frame_epoch and time.time() - last_frame_epoch > stale_after:
            return None

        try:
            frame_mtime = path.stat().st_mtime
        except OSError:
            return None

        if not last_frame_epoch and time.time() - frame_mtime > stale_after:
            return None

        return _read_bytes(path)

    def _stop_duplicate_processes(self, *, keep_pid: int | None = None) -> None:
        worker_dir_token = str(self.worker_dir)
        drone_token = self.drone_id
        for pid, command in _processes():
            if pid == os.getpid() or (keep_pid is not None and pid == keep_pid):
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
