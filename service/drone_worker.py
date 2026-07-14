from __future__ import annotations

from collections import deque
import hashlib
import os
import select
import signal
import threading
import time
from pathlib import Path
from queue import Empty
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from service.event_detector import EventDetector
from service.latest_frame_queue import CapturedFrame, LatestFrameQueue
from service.model_registry import ModelRegistry
from service.nvr_store import NvrStore
from service.recorder import EventRecorder
from service.schemas import utc_now
from service.processed_publisher import (
    FfmpegFramePublisher,
    PUBLISHER_PROCESS_MARKER,
    sanitize_stream_url,
)
from service.second_stage_classifier import SecondStageClassifier, SecondStageSummary
from service.settings import ServiceSettings
from service.runtime_log import safe_console_log


_PYRONE_WATERMARK_IMAGE: np.ndarray | None = None
_PYRONE_WATERMARK_LOOKED_UP = False


def _load_pyrone_watermark_image() -> np.ndarray | None:
    global _PYRONE_WATERMARK_IMAGE, _PYRONE_WATERMARK_LOOKED_UP

    if _PYRONE_WATERMARK_LOOKED_UP:
        return _PYRONE_WATERMARK_IMAGE

    _PYRONE_WATERMARK_LOOKED_UP = True
    service_dir = Path(__file__).resolve().parent
    candidates = [
        service_dir / "img" / "Pyrone_2.png",
        service_dir.parent / "img" / "Pyrone_2.png",
        service_dir.parent.parent / "img" / "Pyrone_2.png",
    ]

    for candidate in candidates:
        image = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)

        if image is not None:
            _PYRONE_WATERMARK_IMAGE = image
            return _PYRONE_WATERMARK_IMAGE

    return None


def _ffmpeg_processes() -> list[tuple[int, list[str]]]:
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

        if not command or "ffmpeg" not in Path(command[0]).name.lower():
            continue

        processes.append((pid, command))

    return processes


def _terminate_ffmpeg_process(pid: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return

    deadline = time.monotonic() + 1.5
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


def _terminate_marked_publisher_processes(
    output_url: str,
    *,
    exclude_pid: int | None = None,
) -> None:
    if not output_url:
        return

    for pid, command in _ffmpeg_processes():
        if exclude_pid and pid == exclude_pid:
            continue
        if output_url not in command:
            continue
        if PUBLISHER_PROCESS_MARKER not in command or "rawvideo" not in command:
            continue
        _terminate_ffmpeg_process(pid)


def cleanup_orphaned_processed_publishers(active_output_urls: set[str]) -> None:
    for pid, command in _ffmpeg_processes():
        output_url = next(
            (
                part
                for part in command
                if "/processed/" in part and part.startswith(("rtsp://", "rtsps://", "rtmp://", "rtmps://"))
            ),
            "",
        )

        if not output_url or output_url in active_output_urls:
            continue

        if (
            PUBLISHER_PROCESS_MARKER not in command
            or "rawvideo" not in command
            or ("rtsp" not in command and "flv" not in command)
        ):
            continue

        _terminate_ffmpeg_process(pid)


class _FfmpegY4mReader:
    def __init__(
        self,
        *,
        ffmpeg_path: str,
        source: str,
        read_timeout_seconds: float = 5.0,
    ) -> None:
        self.source = source
        self._process: subprocess.Popen[bytes] | None = None
        self._read_timeout_seconds = max(0.5, float(read_timeout_seconds))
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._frame_size = 0
        self._header_ready = False
        self._duplicate_frame_count = 0
        self._last_frame_digest: bytes | None = None
        self._last_unique_frame_at = time.monotonic()
        self._startup_deadline = time.monotonic() + max(self._read_timeout_seconds * 3.0, 12.0)
        self._command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "+genpts+discardcorrupt",
            "-analyzeduration",
            "1000000",
            "-probesize",
            "1000000",
            "-rtbufsize",
            "100M",
            "-max_delay",
            "500000",
            "-flags",
            "low_delay",
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            source,
            "-an",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "yuv4mpegpipe",
            "pipe:1",
        ]
        self._process = subprocess.Popen(
            self._command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            start_new_session=True,
        )

    def isOpened(self) -> bool:  # noqa: N802
        return bool(self._process and self._process.stdout and self._process.poll() is None)

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    def read(self):
        if not self._process or not self._process.stdout:
            return False, None

        if not self._header_ready and not self._read_y4m_header():
            return False, None

        frame_header = self._read_line()
        if not frame_header or not frame_header.startswith(b"FRAME"):
            return False, None

        payload = self._read_exact(self._frame_size)
        if payload is None:
            return False, None

        self._track_duplicate_frame(payload)
        try:
            yuv = np.frombuffer(payload, dtype=np.uint8).reshape((self._height * 3 // 2, self._width))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        except (ValueError, cv2.error):
            return False, None

        return True, frame

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def _read_y4m_header(self) -> bool:
        header = self._read_line()
        if not header or not header.startswith(b"YUV4MPEG2 "):
            return False

        width = 0
        height = 0
        fps = 0.0

        for raw_token in header.split()[1:]:
            token = raw_token.decode("ascii", errors="ignore")
            if token.startswith("W"):
                try:
                    width = int(token[1:])
                except ValueError:
                    width = 0
            elif token.startswith("H"):
                try:
                    height = int(token[1:])
                except ValueError:
                    height = 0
            elif token.startswith("F") and ":" in token:
                try:
                    numerator, denominator = token[1:].split(":", 1)
                    den = float(denominator)
                    fps = float(numerator) / den if den else 0.0
                except ValueError:
                    fps = 0.0

        if width <= 0 or height <= 0:
            return False

        self._width = width
        self._height = height
        # Algunos streams RTSP exponen el timebase (por ejemplo 90000/1) en el
        # campo F de Y4M. Eso no es FPS y no debe contaminar scheduling/health.
        self._fps = fps if 0.1 <= fps <= 240.0 else 0.0
        self._frame_size = width * height * 3 // 2
        self._header_ready = True
        self._startup_deadline = 0.0
        return True

    def _read_line(self) -> bytes | None:
        line = bytearray()

        while True:
            chunk = self._read_chunk(1)
            if chunk is None:
                return None
            if chunk == b"\n":
                return bytes(line).rstrip(b"\r")
            line.extend(chunk)
            if len(line) > 4096:
                return None

    def _read_exact(self, size: int) -> bytes | None:
        payload = bytearray()
        remaining = size

        while remaining > 0:
            chunk = self._read_chunk(min(remaining, 256 * 1024))
            if chunk is None:
                return None
            payload.extend(chunk)
            remaining -= len(chunk)

        return bytes(payload)

    def _read_chunk(self, size: int) -> bytes | None:
        if not self._process or not self._process.stdout:
            return None

        stdout = self._process.stdout
        while True:
            try:
                ready, _, _ = select.select([stdout], [], [], self._read_timeout_seconds)
            except (OSError, ValueError):
                return None
            if not ready:
                if time.monotonic() < self._startup_deadline:
                    continue
                return None

            try:
                chunk = stdout.read(size)
            except (OSError, ValueError):
                return None
            if not chunk:
                return None
            return bytes(chunk)

    def _track_duplicate_frame(self, frame_bytes: bytes) -> None:
        digest = hashlib.blake2s(frame_bytes, digest_size=8).digest()
        now = time.monotonic()

        if digest != self._last_frame_digest:
            self._last_frame_digest = digest
            self._last_unique_frame_at = now
            self._duplicate_frame_count = 0
            return

        self._duplicate_frame_count += 1

    def release(self) -> None:
        process = self._process
        self._process = None
        if not process:
            return
        try:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
                process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait(timeout=3)
        finally:
            if process.stdout:
                process.stdout.close()


class DroneWorker:
    def __init__(
        self,
        *,
        settings: ServiceSettings,
        model_registry: ModelRegistry,
        store: NvrStore,
        pipeline: dict[str, object],
        on_recording_saved=None,
    ) -> None:
        self.settings = settings
        self.model_registry = model_registry
        self.store = store
        self.on_recording_saved = on_recording_saved
        self._lock = threading.Lock()
        self._pipeline = dict(pipeline)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_processed_frame_jpeg: bytes | None = None
        self._latest_raw_frame_jpeg: bytes | None = None
        self._processed_frame_sequence = 0
        self._raw_frame_sequence = 0
        self._processed_publisher: FfmpegFramePublisher | None = None
        self._processed_publisher_signature: tuple[object, ...] | None = None
        self._processed_publisher_next_retry_at = 0.0
        self._processed_publisher_retry_delay = self.settings.processed_publish_retry_seconds
        self._processed_publisher_start_count = 0
        self._source_retry_delay = self.settings.reconnect_delay_seconds
        self._source_retry_count = 0
        self._model_retry_delay = self.settings.model_retry_seconds
        self._model_retry_next_at = 0.0
        self._model_retry_count = 0
        self._model_registry_revision = 0
        self._model_loaded_signature: tuple[object, ...] | None = None
        self._model_id = ""
        self._model_name = ""
        self._latest_detections: dict[str, object] = self._empty_detections()
        self._active_capture = None
        resolved_device = self._resolve_device()
        self._runtime = {
            "status": "configured",
            "message": "",
            "sourceOpened": False,
            "sourceType": "unknown",
            "sourceUrl": "",
            "captureBackend": "",
            "capturePid": None,
            "activeRtspConnections": 0,
            "maxRtspConnections": 1,
            "singleIngestHealthy": True,
            "sourceSessionId": "",
            "sourceOpenCount": 0,
            "sourceReconnectCount": 0,
            "sourceRetryCount": 0,
            "sourceRetryDelaySeconds": 0.0,
            "sourceStable": False,
            "sourceStallCount": 0,
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
            "latestProcessedFrameAvailable": False,
            "latestProcessedFrameContentType": "image/jpeg",
            "latestRawFrameAvailable": False,
            "latestRawFrameContentType": "image/jpeg",
            "processedStreamReady": False,
            "processedStreamUrl": (
                f"/v1/pipelines/{str(pipeline.get('droneId') or '')}/stream.mjpg"
                if self.settings.legacy_mjpeg_enabled
                else ""
            ),
            "processedPublisherPid": None,
            "processedPublisherTransport": self.settings.processed_publish_transport,
            "processedPublisherUrl": "",
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
            "device": resolved_device,
            "modelRetryCount": 0,
            "modelRetryDelaySeconds": 0.0,
            "modelRegistryRevision": self.model_registry.revision,
            "modelCatalogError": self.model_registry.last_reload_error,
            "secondStageEnabled": self.settings.second_stage_enabled and bool(pipeline.get("secondStageEnabled", True)),
            "secondStageApplied": False,
            "secondStageAvailable": self.settings.second_stage_model_path.exists(),
            "secondStageModelPath": str(self.settings.second_stage_model_path),
            "secondStageConfLow": self.settings.second_stage_conf_low,
            "secondStageConfHigh": self.settings.second_stage_conf_high,
            "secondStageCropSize": self.settings.second_stage_crop_size,
            "secondStageSentToClassifier": 0,
            "secondStageRejectedByClassifier": 0,
            "secondStageDroppedLow": 0,
            "secondStageKeptAfterClassifier": 0,
            "secondStageLastMs": 0.0,
            "secondStageLastReason": "",
        }
        self._model: YOLO | None = None
        self._second_stage = SecondStageClassifier(settings=self.settings, device=str(resolved_device))
        self._detector = EventDetector(
            min_positive_frames=self.settings.event_min_positive_frames,
            confirmation_window_frames=self.settings.event_confirmation_window_frames,
            post_event_seconds=self.settings.post_event_seconds,
            cooldown_seconds=self.settings.event_cooldown_seconds,
        )
        self._recorder = EventRecorder(self.settings, self.store)
        self._fps_window: deque[float] = deque(maxlen=30)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"drone-worker-{self.pipeline_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        self._release_active_capture()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        recording = self._recorder.close()
        if recording:
            self._store_recording(recording)
        self._stop_processed_publisher()
        self._clear_preview()
        self._set_runtime(status="stopped", message="worker stopped", sourceOpened=False, currentEvent=None)

    def update_pipeline(self, pipeline: dict[str, object]) -> None:
        with self._lock:
            self._pipeline = dict(pipeline)
        self._wake_event.set()

    def latest_frame(self) -> bytes | None:
        return self.latest_processed_frame()

    def latest_processed_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_processed_frame_jpeg

    def latest_processed_frame_snapshot(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._processed_frame_sequence, self._latest_processed_frame_jpeg

    def latest_raw_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_raw_frame_jpeg

    def latest_raw_frame_snapshot(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._raw_frame_sequence, self._latest_raw_frame_jpeg

    def latest_detections(self) -> dict[str, object]:
        with self._lock:
            return {
                **self._latest_detections,
                "items": [
                    dict(item)
                    for item in self._latest_detections.get("items", [])
                    if isinstance(item, dict)
                ],
            }

    def runtime_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                **self._runtime,
                "currentEvent": dict(self._runtime["currentEvent"]) if self._runtime["currentEvent"] else None,
                "lastRecording": dict(self._runtime["lastRecording"]) if self._runtime["lastRecording"] else None,
                "modelId": self._model_id or str(self._pipeline.get("currentModelId") or ""),
                "modelName": self._model_name or "",
            }

    @property
    def pipeline_id(self) -> str:
        return str(self._pipeline.get("droneId") or "")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_forever()
            except Exception as error:
                pipeline = self._pipeline_snapshot()
                source = str(pipeline.get("rtspUrl") or "")
                detail = str(error)
                if source:
                    detail = detail.replace(source, self._sanitize_source_url(source))
                reason = f"unexpected worker error: {detail}"
                self._release_active_capture()
                self._clear_preview()
                self._set_runtime(
                    status="waiting_source",
                    message=reason,
                    sourceOpened=False,
                    activeRtspConnections=0,
                    singleIngestHealthy=True,
                    lastSourceError=reason,
                )
                self._wait_for_source_retry(reason)

    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            pipeline = self._pipeline_snapshot()
            min_frame_interval = 1.0 / max(
                0.5,
                float(pipeline.get("processingFps") or self.settings.processing_fps),
            )
            if not pipeline.get("analyticsEnabled", True):
                self._stop_processed_publisher()
                self._clear_preview()
                self._set_runtime(
                    status="disabled",
                    message="analytics disabled",
                    sourceOpened=False,
                    activeRtspConnections=0,
                    singleIngestHealthy=True,
                )
                self._wait_interruptibly(0.5)
                continue

            source = str(pipeline.get("rtspUrl") or "").strip()
            if not source:
                self._stop_processed_publisher()
                self._clear_preview()
                self._set_runtime(
                    status="waiting_source",
                    message="missing source URL",
                    sourceOpened=False,
                    activeRtspConnections=0,
                    singleIngestHealthy=True,
                )
                self._wait_for_source_retry("missing source URL")
                continue

            source_path = Path(source)
            if source_path.exists() and (
                source_path.is_dir() or source_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ):
                self._mark_source_open(
                    source=source,
                    source_type="image",
                    capture_backend="image-loader",
                    capture_pid=None,
                    source_fps=0.0,
                )
                self._process_image_source(source_path)
                self._mark_source_closed(last_error="")
                if not self._stop_event.is_set():
                    self._wait_for_source_retry("image source ended")
                continue

            try:
                capture = self._open_capture(source)
            except Exception as error:
                safe_error = f"could not create source capture: {error}"
                self._stop_processed_publisher()
                self._clear_preview()
                self._set_runtime(
                    status="waiting_source",
                    message=safe_error,
                    sourceOpened=False,
                    activeRtspConnections=0,
                    singleIngestHealthy=True,
                    lastSourceError=safe_error,
                )
                self._wait_for_source_retry(safe_error)
                continue
            if not capture.isOpened():
                self._stop_processed_publisher()
                self._clear_preview()
                self._set_runtime(
                    status="waiting_source",
                    message=f"could not open source {source}",
                    sourceOpened=False,
                    activeRtspConnections=0,
                    singleIngestHealthy=True,
                    lastSourceError=f"could not open source {self._sanitize_source_url(source)}",
                )
                capture.release()
                self._wait_for_source_retry(
                    f"could not open source {self._sanitize_source_url(source)}"
                )
                continue

            source_fps = self._capture_fps(capture)
            self._set_active_capture(capture)
            self._mark_source_open(
                source=source,
                source_type="rtsp" if source.strip().lower().startswith("rtsp://") else "video",
                capture_backend=self._capture_backend_name(capture),
                capture_pid=self._capture_pid(capture),
                source_fps=source_fps,
            )
            try:
                self._run_capture_session(
                    capture,
                    source_fps=source_fps,
                    initial_frame_interval=min_frame_interval,
                )
            finally:
                self._clear_active_capture(capture)
                capture.release()
            self._mark_source_closed(last_error="")
            if not self._stop_event.is_set():
                runtime = self.runtime_snapshot()
                retry_reason = str(runtime.get("lastSourceError") or "source stream ended")
                self._wait_for_source_retry(retry_reason)

    def _process_image_source(self, source_path: Path) -> None:
        image_paths = self._resolve_image_paths(source_path)
        if not image_paths:
            self._set_runtime(status="waiting_source", message=f"no images found in {source_path}", sourceOpened=False)
            return

        self._set_runtime(status="starting", message=f"image source ready: {source_path}", sourceOpened=True)
        while not self._stop_event.is_set():
            for image_path in image_paths:
                if self._stop_event.is_set():
                    return
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                pipeline = self._pipeline_snapshot()
                if not pipeline.get("analyticsEnabled", True):
                    return
                pipeline_fps = float(pipeline.get("processingFps") or self.settings.processing_fps)
                captured_at = utc_now()
                self._record_captured_frame(captured_at=captured_at, queue_depth=0, dropped=0)
                try:
                    self._process_frame(
                        frame,
                        source_fps=pipeline_fps,
                        pipeline=pipeline,
                        frame_age_ms=0.0,
                        captured_at=captured_at,
                    )
                except Exception as error:
                    self._handle_frame_processing_error(error)
                time.sleep(1.0 / max(0.5, pipeline_fps))

    def _run_capture_session(
        self,
        capture,
        *,
        source_fps: float,
        initial_frame_interval: float,
    ) -> None:
        """Keep ingest current while inference consumes only the newest available frame."""

        frame_queue: LatestFrameQueue[np.ndarray] = LatestFrameQueue()
        session_stop = threading.Event()
        capture_failed = threading.Event()
        session_started_at = time.monotonic()
        last_capture_progress_at = [session_started_at]
        first_frame_at = [0.0]
        failure_reason = [""]
        source_stable = False
        capture_stall_timeout = max(
            float(getattr(self.settings, "rtsp_read_timeout_seconds", 5.0)),
            float(
                getattr(
                    self.settings,
                    "capture_stall_timeout_seconds",
                    max(float(getattr(self.settings, "rtsp_read_timeout_seconds", 5.0)) * 2.0, 10.0),
                )
            ),
        )
        source_stable_reset_seconds = max(
            1.0,
            float(getattr(self.settings, "source_stable_reset_seconds", 10.0)),
        )

        def capture_frames() -> None:
            sequence = 0
            while not self._stop_event.is_set() and not session_stop.is_set():
                try:
                    ok, frame = capture.read()
                except Exception as error:
                    if not failure_reason[0]:
                        failure_reason[0] = f"source frame read error: {error}"
                    capture_failed.set()
                    break
                if not ok or frame is None:
                    if not failure_reason[0]:
                        failure_reason[0] = "source frame read failed"
                    capture_failed.set()
                    break

                sequence += 1
                captured_monotonic = time.monotonic()
                last_capture_progress_at[0] = captured_monotonic
                if not first_frame_at[0]:
                    first_frame_at[0] = captured_monotonic
                captured_at = utc_now()
                dropped = frame_queue.put_latest(
                    CapturedFrame(
                        sequence=sequence,
                        value=frame,
                        captured_monotonic=captured_monotonic,
                        captured_at=captured_at,
                    )
                )
                self._record_captured_frame(
                    captured_at=captured_at,
                    queue_depth=frame_queue.qsize(),
                    dropped=dropped,
                )

        reader = threading.Thread(
            target=capture_frames,
            name=f"drone-capture-{self.pipeline_id}",
            daemon=True,
        )
        reader.start()
        next_inference_at = 0.0
        frame_interval = initial_frame_interval
        resolved_source_fps = max(0.0, float(source_fps or 0.0))

        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if next_inference_at and now < next_inference_at:
                    if capture_failed.is_set() and frame_queue.empty():
                        break
                    self._stop_event.wait(min(next_inference_at - now, 0.05))
                    continue

                try:
                    captured = frame_queue.get(timeout=0.25)
                except Empty:
                    if capture_failed.is_set():
                        break
                    stalled_for = time.monotonic() - last_capture_progress_at[0]
                    if stalled_for >= capture_stall_timeout:
                        failure_reason[0] = (
                            "source frame capture stalled for "
                            f"{stalled_for:.1f}s"
                        )
                        capture_failed.set()
                        with self._lock:
                            self._runtime["sourceStallCount"] = (
                                int(self._runtime.get("sourceStallCount") or 0) + 1
                            )
                        # _FfmpegY4mReader.release() terminates the subprocess
                        # and unblocks its stdout reader.  The finally block
                        # below remains the authoritative cleanup path.
                        capture.release()
                        break
                    continue

                captured, superseded = frame_queue.drain_to_latest(captured)
                if superseded:
                    self._record_dropped_before_inference(superseded)
                if captured is None:
                    continue

                if (
                    not source_stable
                    and first_frame_at[0]
                    and time.monotonic() - first_frame_at[0]
                    >= source_stable_reset_seconds
                ):
                    self._reset_source_retry_backoff()
                    source_stable = True

                if resolved_source_fps <= 0.0:
                    detected_source_fps = self._capture_fps(capture)
                    if detected_source_fps > 0.0:
                        resolved_source_fps = detected_source_fps
                        self._set_runtime(sourceFps=round(resolved_source_fps, 2))

                pipeline = self._pipeline_snapshot()
                if not pipeline.get("analyticsEnabled", True):
                    break
                frame_interval = 1.0 / max(
                    0.5,
                    float(pipeline.get("processingFps") or self.settings.processing_fps),
                )
                inference_started_at = time.monotonic()
                frame_age_ms = max(
                    0.0,
                    (inference_started_at - captured.captured_monotonic) * 1000.0,
                )
                self._set_runtime(captureQueueDepth=frame_queue.qsize())
                try:
                    self._process_frame(
                        captured.value,
                        source_fps=resolved_source_fps,
                        pipeline=pipeline,
                        frame_age_ms=frame_age_ms,
                        captured_at=captured.captured_at,
                    )
                except Exception as error:
                    self._handle_frame_processing_error(error)
                next_inference_at = inference_started_at + frame_interval
        finally:
            session_stop.set()
            capture.release()
            reader.join(timeout=max(1.0, self.settings.rtsp_read_timeout_seconds + 0.5))
            frame_queue.clear()
            self._set_runtime(captureQueueDepth=0)

        if capture_failed.is_set() and not self._stop_event.is_set():
            normalized_error = failure_reason[0] or "source frame read failed"
            self._stop_processed_publisher()
            self._clear_preview()
            self._set_runtime(
                status="waiting_source",
                message=normalized_error,
                sourceOpened=False,
                activeRtspConnections=0,
                singleIngestHealthy=True,
                lastSourceError=normalized_error,
            )

    def _record_captured_frame(self, *, captured_at: str, queue_depth: int, dropped: int) -> None:
        with self._lock:
            self._runtime["framesCaptured"] = int(self._runtime.get("framesCaptured") or 0) + 1
            self._runtime["framesDroppedBeforeInference"] = (
                int(self._runtime.get("framesDroppedBeforeInference") or 0) + int(dropped)
            )
            self._runtime["captureQueueDepth"] = min(max(int(queue_depth), 0), 1)
            self._runtime["lastCapturedFrameAt"] = captured_at

    def _record_dropped_before_inference(self, dropped: int) -> None:
        with self._lock:
            self._runtime["framesDroppedBeforeInference"] = (
                int(self._runtime.get("framesDroppedBeforeInference") or 0) + int(dropped)
            )

    def _resolve_image_paths(self, source_path: Path) -> list[Path]:
        if source_path.is_dir():
            return sorted(
                path
                for path in source_path.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        if source_path.is_file():
            return [source_path]
        return []

    def _prepare_frame(self, frame):
        if frame is None:
            return frame
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= self.settings.max_frame_width:
            return frame
        scale = self.settings.max_frame_width / float(frame_width)
        resized = cv2.resize(
            frame,
            (int(frame_width * scale), int(frame_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        return resized

    def _open_capture(self, source: str):
        normalized = source.strip().lower()
        if normalized.startswith("rtsp://"):
            ffmpeg_reader = _FfmpegY4mReader(
                ffmpeg_path=self.settings.ffmpeg_path,
                source=source,
                read_timeout_seconds=self.settings.rtsp_read_timeout_seconds,
            )
            if ffmpeg_reader.isOpened():
                return ffmpeg_reader
            ffmpeg_reader.release()

            capture = cv2.VideoCapture(source)
            if capture.isOpened():
                return capture
            capture.release()
            return ffmpeg_reader
        return cv2.VideoCapture(source)

    @staticmethod
    def _capture_backend_name(capture) -> str:
        if isinstance(capture, _FfmpegY4mReader):
            return "ffmpeg-y4m-reader"
        return "opencv-videocapture"

    @staticmethod
    def _capture_pid(capture) -> int | None:
        if isinstance(capture, _FfmpegY4mReader):
            return capture.pid
        return None

    @staticmethod
    def _capture_fps(capture) -> float:
        getter = getattr(capture, "get", None)
        if not callable(getter):
            return 0.0
        try:
            return float(getter(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            return 0.0

    def _wait_interruptibly(self, delay_seconds: float) -> bool:
        """Wait for retry delay, but wake immediately on stop or pipeline update."""

        awakened = self._wake_event.wait(max(0.0, float(delay_seconds)))
        if awakened and not self._stop_event.is_set():
            self._wake_event.clear()
        return awakened

    def _wait_for_source_retry(self, reason: str) -> None:
        if self._stop_event.is_set():
            return
        retry_max = max(
            self.settings.reconnect_delay_seconds,
            float(getattr(self.settings, "reconnect_delay_max_seconds", 30.0)),
        )
        delay = min(
            max(self._source_retry_delay, self.settings.reconnect_delay_seconds),
            retry_max,
        )
        self._source_retry_count += 1
        self._set_runtime(
            status="waiting_source",
            message=f"{reason}; retrying source in {delay:.1f}s",
            sourceOpened=False,
            activeRtspConnections=0,
            sourceRetryCount=self._source_retry_count,
            sourceRetryDelaySeconds=round(delay, 2),
            sourceStable=False,
        )
        self._source_retry_delay = min(
            delay * 2.0,
            retry_max,
        )
        self._wait_interruptibly(delay)

    def _reset_source_retry_backoff(self) -> None:
        self._source_retry_delay = self.settings.reconnect_delay_seconds
        self._set_runtime(
            sourceRetryDelaySeconds=0.0,
            sourceStable=True,
        )

    def _process_frame(
        self,
        frame,
        *,
        source_fps: float,
        pipeline: dict[str, object],
        frame_age_ms: float,
        captured_at: str,
    ) -> None:
        frame = self._prepare_frame(frame)
        self._update_raw_preview(frame)
        frame_at = captured_at or utc_now()

        model_info = self._ensure_model(pipeline)
        if not model_info:
            self._clear_processed_preview()
            return

        frame_height, frame_width = frame.shape[:2]
        effective_fps = max(float(pipeline.get("processingFps") or self.settings.processing_fps), 1.0)
        self._recorder.configure(frame_width=frame_width, frame_height=frame_height, fps=effective_fps)

        inference_started = perf_counter()
        confidence_threshold = float(pipeline.get("confidenceThreshold") or self.settings.inference_confidence)
        detector_confidence_threshold = self._second_stage.detector_confidence_threshold(
            confidence_threshold,
            model_info=model_info,
            pipeline=pipeline,
        )
        second_stage_summary = SecondStageSummary(
            enabled=self.settings.second_stage_enabled,
            applied=False,
            available=self.settings.second_stage_model_path.exists(),
            reason="no detections",
        )
        if self._uses_skyline_composite_preprocessor(model_info):
            inference_result = self._predict_skyline_composite(
                frame,
                confidence_threshold=detector_confidence_threshold,
                model_info=model_info,
                pipeline=pipeline,
            )
            result = None
            detection_items = inference_result["items"]
            detection_items, second_stage_summary = self._second_stage.refine_items(
                frame,
                detection_items,
                model_info=model_info,
                pipeline=pipeline,
            )
            detection_count = len(detection_items)
            max_confidence = max(
                (float(item.get("confidence") or 0.0) for item in detection_items),
                default=0.0,
            )
        else:
            predict_options = {
                "source": frame,
                "conf": detector_confidence_threshold,
                "imgsz": int(model_info.get("imageSize") or self.settings.inference_image_size),
                "device": self._runtime["device"],
                "verbose": False,
            }
            if model_info.get("iouThreshold"):
                predict_options["iou"] = float(model_info["iouThreshold"])
            result = self._model.predict(**predict_options)[0]
            result, second_stage_summary = self._second_stage.refine_result(
                frame,
                result,
                model_info=model_info,
                pipeline=pipeline,
            )
            detection_count = 0 if result.boxes is None else len(result.boxes)
            max_confidence = max(result.boxes.conf.tolist()) if detection_count and result.boxes is not None else 0.0
            detection_items = None
        inference_ms = (perf_counter() - inference_started) * 1000.0

        if detection_items is not None:
            structured_detections = self._build_structured_detections_from_items(
                detection_items,
                frame_at=frame_at,
                frame_width=frame_width,
                frame_height=frame_height,
                model_info=model_info,
                pipeline=pipeline,
            )
        else:
            structured_detections = self._build_structured_detections(
                result,
                frame_at=frame_at,
                frame_width=frame_width,
                frame_height=frame_height,
                model_info=model_info,
                pipeline=pipeline,
            )
        positive = detection_count > 0
        event = self._detector.update(
            positive=positive,
            detection_count=detection_count,
            max_confidence=max_confidence,
        )

        if detection_items is not None:
            annotated = frame.copy()
            if detection_count:
                self._draw_detection_items(
                    annotated,
                    detection_items,
                    model_info=model_info,
                    pipeline=pipeline,
                )
        elif detection_count:
            self._apply_display_labels(result, model_info=model_info, pipeline=pipeline)
            annotated = result.plot()
        else:
            annotated = frame.copy()
        self._overlay_status(
            annotated,
            model_name=str(model_info.get("name") or model_info.get("id") or ""),
            detection_count=detection_count,
            max_confidence=max_confidence,
            sensor_type=str(pipeline.get("sensorType") or "unknown"),
        )

        if pipeline.get("recordOnEvent", True):
            recording = self._recorder.ingest(
                annotated,
                timestamp=utc_now(),
                event=event,
                metadata={
                    "droneId": pipeline["droneId"],
                    "eventType": "smoke_detection",
                    "modelId": pipeline["currentModelId"],
                    "sensorType": pipeline.get("sensorType", "unknown"),
                    "recordingSegmentMode": pipeline.get("recordingSegmentMode"),
                    "recordingSegmentMinutes": pipeline.get("recordingSegmentMinutes"),
                    "recordingSegmentMaxMb": pipeline.get("recordingSegmentMaxMb"),
                },
                fps=effective_fps,
            )
            if recording:
                self._store_recording(recording)

        self._update_detections(structured_detections)
        self._update_preview(annotated)
        self._publish_processed_frame(
            annotated,
            fps=effective_fps,
            source_url=str(pipeline.get("rtspUrl") or ""),
        )
        self._update_runtime_after_frame(
            frame_at=frame_at,
            frame_width=frame_width,
            frame_height=frame_height,
            source_fps=source_fps,
            inference_ms=inference_ms,
            detection_count=detection_count,
            max_confidence=max_confidence,
            event=event,
            second_stage_summary=second_stage_summary,
            frame_age_ms=frame_age_ms,
        )

    def _empty_detections(self) -> dict[str, object]:
        return {
            "droneId": self.pipeline_id,
            "frameAt": "",
            "modelId": self._model_id,
            "modelName": self._model_name,
            "sensorType": str(self._pipeline.get("sensorType") or "unknown"),
            "frameWidth": None,
            "frameHeight": None,
            "items": [],
        }

    def _build_structured_detections(
        self,
        result,
        *,
        frame_at: str,
        frame_width: int,
        frame_height: int,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> dict[str, object]:
        items: list[dict[str, object]] = []
        boxes = getattr(result, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
            confidences = boxes.conf.detach().cpu().tolist() if getattr(boxes, "conf", None) is not None else []
            classes = boxes.cls.detach().cpu().tolist() if getattr(boxes, "cls", None) is not None else []
            names = getattr(result, "names", {}) or {}
            mask_polygons = self._mask_polygons_for_result(result)

            for index, bbox_values in enumerate(xyxy):
                if len(bbox_values) < 4:
                    continue

                x1, y1, x2, y2 = [float(value) for value in bbox_values[:4]]
                x1 = min(max(x1, 0.0), float(frame_width))
                x2 = min(max(x2, 0.0), float(frame_width))
                y1 = min(max(y1, 0.0), float(frame_height))
                y2 = min(max(y2, 0.0), float(frame_height))
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)

                if width <= 0.0 or height <= 0.0:
                    continue

                class_index = int(classes[index]) if index < len(classes) else 0
                class_name = self._display_class_name(
                    str(names.get(class_index, class_index)),
                    model_info=model_info,
                    pipeline=pipeline,
                )
                confidence = float(confidences[index]) if index < len(confidences) else 0.0
                target_pixel = self._target_pixel_for_detection(
                    bbox=(x1, y1, x2, y2),
                    polygon=mask_polygons[index] if index < len(mask_polygons) else [],
                )

                items.append(
                    {
                        "detectionId": self._detection_id(frame_at, index),
                        "index": index,
                        "classIndex": class_index,
                        "className": class_name,
                        "confidence": round(confidence, 6),
                        "bbox": {
                            "x": round(x1, 2),
                            "y": round(y1, 2),
                            "width": round(width, 2),
                            "height": round(height, 2),
                        },
                        "targetPixel": target_pixel,
                    }
                )

        return {
            "droneId": self.pipeline_id,
            "frameAt": frame_at,
            "modelId": str(model_info.get("id") or self._model_id),
            "modelName": str(model_info.get("name") or self._model_name),
            "sensorType": str(pipeline.get("sensorType") or "unknown"),
            "frameWidth": frame_width,
            "frameHeight": frame_height,
            "items": items,
        }

    def _build_structured_detections_from_items(
        self,
        items: list[dict[str, object]],
        *,
        frame_at: str,
        frame_width: int,
        frame_height: int,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> dict[str, object]:
        structured_items: list[dict[str, object]] = []

        for index, item in enumerate(items):
            bbox = item.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            x1 = min(max(x1, 0.0), float(frame_width))
            x2 = min(max(x2, 0.0), float(frame_width))
            y1 = min(max(y1, 0.0), float(frame_height))
            y2 = min(max(y2, 0.0), float(frame_height))
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            if width <= 0.0 or height <= 0.0:
                continue

            class_index = int(item.get("classIndex") or 0)
            class_name = self._display_class_name(
                str(item.get("className") or class_index),
                model_info=model_info,
                pipeline=pipeline,
            )
            confidence = float(item.get("confidence") or 0.0)
            target_pixel = self._target_pixel_for_detection(bbox=(x1, y1, x2, y2), polygon=[])

            structured_item = {
                "detectionId": self._detection_id(frame_at, index),
                "index": index,
                "classIndex": class_index,
                "className": class_name,
                "confidence": round(confidence, 6),
                "bbox": {
                    "x": round(x1, 2),
                    "y": round(y1, 2),
                    "width": round(width, 2),
                    "height": round(height, 2),
                },
                "targetPixel": target_pixel,
            }
            if item.get("sourceRegion"):
                structured_item["sourceRegion"] = str(item["sourceRegion"])
            structured_items.append(structured_item)

        return {
            "droneId": self.pipeline_id,
            "frameAt": frame_at,
            "modelId": str(model_info.get("id") or self._model_id),
            "modelName": str(model_info.get("name") or self._model_name),
            "sensorType": str(pipeline.get("sensorType") or "unknown"),
            "frameWidth": frame_width,
            "frameHeight": frame_height,
            "preprocessor": str(model_info.get("preprocessor") or "none"),
            "items": structured_items,
        }

    @staticmethod
    def _uses_skyline_composite_preprocessor(model_info: dict[str, object]) -> bool:
        return str(model_info.get("preprocessor") or "").strip().lower() == "skyline_composite"

    def _predict_skyline_composite(
        self,
        frame,
        *,
        confidence_threshold: float,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> dict[str, object]:
        composite, mapping = self._build_skyline_composite(frame, model_info=model_info)
        predict_options = {
            "source": composite,
            "conf": confidence_threshold,
            "imgsz": int(model_info.get("imageSize") or self.settings.inference_image_size),
            "device": self._runtime["device"],
            "verbose": False,
        }
        if model_info.get("iouThreshold"):
            predict_options["iou"] = float(model_info["iouThreshold"])

        result = self._model.predict(**predict_options)[0]
        items = self._detection_items_from_composite_result(
            result,
            mapping=mapping,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            model_info=model_info,
            pipeline=pipeline,
        )
        nms_iou = float(model_info.get("compositeNmsIou") or 0.5)
        return {
            "items": self._nms_detection_items(items, iou_threshold=nms_iou),
            "composite": composite,
        }

    def _build_skyline_composite(self, frame, *, model_info: dict[str, object]):
        frame_height, frame_width = frame.shape[:2]
        target_size = int(model_info.get("imageSize") or self.settings.inference_image_size or 640)
        target_size = max(320, target_size)
        global_width = target_size
        global_height = max(1, int(round(frame_height * (global_width / frame_width))))

        if global_height >= target_size:
            scale = target_size / frame_height
            resized_width = max(1, int(round(frame_width * scale)))
            global_view = cv2.resize(frame, (resized_width, target_size), interpolation=cv2.INTER_AREA)
            composite = np.zeros((target_size, target_size, 3), dtype=frame.dtype)
            offset_x = max(0, (target_size - resized_width) // 2)
            composite[:, offset_x : offset_x + resized_width] = global_view
            return composite, {
                "targetSize": target_size,
                "roiHeight": 0,
                "global": {
                    "scale": scale,
                    "offsetX": offset_x,
                    "offsetY": 0,
                    "width": resized_width,
                    "height": target_size,
                },
                "roi": None,
            }

        roi_height = target_size - global_height
        global_view = cv2.resize(frame, (global_width, global_height), interpolation=cv2.INTER_AREA)
        intermediate_width = int(model_info.get("compositeIntermediateWidth") or target_size * 2)
        intermediate_width = max(target_size, intermediate_width)
        intermediate_scale = intermediate_width / frame_width
        intermediate_height = max(1, int(round(frame_height * intermediate_scale)))
        intermediate = cv2.resize(frame, (intermediate_width, intermediate_height), interpolation=cv2.INTER_LINEAR)

        crop_width = min(target_size, intermediate_width)
        crop_height = min(roi_height, intermediate_height)
        skyline_y = self._estimate_skyline_row(intermediate)
        if skyline_y is None:
            crop_top = max(0, (intermediate_height - crop_height) // 2)
        else:
            crop_top = int(round(skyline_y - crop_height * 0.25))
            crop_top = min(max(crop_top, 0), max(0, intermediate_height - crop_height))
        crop_left = max(0, (intermediate_width - crop_width) // 2)

        roi = intermediate[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]
        if roi.shape[0] != roi_height or roi.shape[1] != target_size:
            roi = cv2.resize(roi, (target_size, roi_height), interpolation=cv2.INTER_LINEAR)

        composite = np.vstack([roi, global_view])
        if composite.shape[0] != target_size or composite.shape[1] != target_size:
            composite = cv2.resize(composite, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        return composite, {
            "targetSize": target_size,
            "roiHeight": roi_height,
            "global": {
                "scale": global_width / frame_width,
                "offsetX": 0,
                "offsetY": roi_height,
                "width": global_width,
                "height": global_height,
            },
            "roi": {
                "scale": intermediate_scale,
                "cropLeft": crop_left,
                "cropTop": crop_top,
                "sourceWidth": crop_width,
                "sourceHeight": crop_height,
                "displayWidth": target_size,
                "displayHeight": roi_height,
                "displayScaleX": crop_width / target_size,
                "displayScaleY": crop_height / roi_height,
                "skylineY": skyline_y,
            },
        }

    @staticmethod
    def _estimate_skyline_row(frame) -> int | None:
        height, width = frame.shape[:2]
        if height < 20 or width < 20:
            return None

        small_width = max(64, int(width * 0.25))
        small_height = max(36, int(height * 0.25))
        small = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        bright_low_texture = (value > 95) & (saturation < 95)
        blue_sky = (hue > 85) & (hue < 135) & (saturation > 25) & (value > 65)
        sky_mask = bright_low_texture | blue_sky
        row_ratios = sky_mask.mean(axis=1)

        sky_rows = np.where(row_ratios > 0.42)[0]
        if len(sky_rows) < max(3, int(small_height * 0.06)):
            return None

        first_sky = int(sky_rows[0])
        for row in range(first_sky, small_height):
            if row_ratios[row] < 0.18:
                return int(round(row / small_height * height))

        return int(round(int(sky_rows[-1]) / small_height * height))

    def _detection_items_from_composite_result(
        self,
        result,
        *,
        mapping: dict[str, object],
        frame_width: int,
        frame_height: int,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> list[dict[str, object]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) <= 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
        confidences = boxes.conf.detach().cpu().tolist() if getattr(boxes, "conf", None) is not None else []
        classes = boxes.cls.detach().cpu().tolist() if getattr(boxes, "cls", None) is not None else []
        names = getattr(result, "names", {}) or {}

        items: list[dict[str, object]] = []
        for index, bbox_values in enumerate(xyxy):
            if len(bbox_values) < 4:
                continue

            mapped = self._map_composite_bbox_to_frame(
                tuple(float(value) for value in bbox_values[:4]),
                mapping=mapping,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if not mapped:
                continue

            x1, y1, x2, y2, source_region = mapped
            if x2 <= x1 or y2 <= y1:
                continue

            class_index = int(classes[index]) if index < len(classes) else 0
            raw_class_name = str(names.get(class_index, class_index))
            class_name = self._display_class_name(
                raw_class_name,
                model_info=model_info,
                pipeline=pipeline,
            )
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            items.append(
                {
                    "classIndex": class_index,
                    "className": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "sourceRegion": source_region,
                }
            )

        return items

    @staticmethod
    def _map_composite_bbox_to_frame(
        bbox: tuple[float, float, float, float],
        *,
        mapping: dict[str, object],
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float, float, str] | None:
        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2.0
        roi_height = float(mapping.get("roiHeight") or 0)
        region = "roi" if roi_height > 0 and center_y < roi_height else "global"

        if region == "roi":
            roi = mapping.get("roi")
            if not isinstance(roi, dict):
                return None
            display_scale_x = float(roi.get("displayScaleX") or 1.0)
            display_scale_y = float(roi.get("displayScaleY") or 1.0)
            crop_left = float(roi.get("cropLeft") or 0.0)
            crop_top = float(roi.get("cropTop") or 0.0)
            scale = float(roi.get("scale") or 1.0)

            mapped_x1 = (crop_left + x1 * display_scale_x) / scale
            mapped_x2 = (crop_left + x2 * display_scale_x) / scale
            mapped_y1 = (crop_top + y1 * display_scale_y) / scale
            mapped_y2 = (crop_top + y2 * display_scale_y) / scale
        else:
            global_mapping = mapping.get("global")
            if not isinstance(global_mapping, dict):
                return None
            scale = float(global_mapping.get("scale") or 1.0)
            offset_x = float(global_mapping.get("offsetX") or 0.0)
            offset_y = float(global_mapping.get("offsetY") or 0.0)
            width = float(global_mapping.get("width") or 0.0)
            height = float(global_mapping.get("height") or 0.0)

            clipped_x1 = min(max(x1, offset_x), offset_x + width)
            clipped_x2 = min(max(x2, offset_x), offset_x + width)
            clipped_y1 = min(max(y1, offset_y), offset_y + height)
            clipped_y2 = min(max(y2, offset_y), offset_y + height)
            mapped_x1 = (clipped_x1 - offset_x) / scale
            mapped_x2 = (clipped_x2 - offset_x) / scale
            mapped_y1 = (clipped_y1 - offset_y) / scale
            mapped_y2 = (clipped_y2 - offset_y) / scale

        mapped_x1 = min(max(mapped_x1, 0.0), float(frame_width))
        mapped_x2 = min(max(mapped_x2, 0.0), float(frame_width))
        mapped_y1 = min(max(mapped_y1, 0.0), float(frame_height))
        mapped_y2 = min(max(mapped_y2, 0.0), float(frame_height))

        return mapped_x1, mapped_y1, mapped_x2, mapped_y2, region

    @classmethod
    def _nms_detection_items(
        cls,
        items: list[dict[str, object]],
        *,
        iou_threshold: float,
    ) -> list[dict[str, object]]:
        threshold = min(max(float(iou_threshold), 0.05), 0.95)
        ordered = sorted(items, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
        kept: list[dict[str, object]] = []

        for item in ordered:
            bbox = item.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) < 4:
                continue
            should_keep = True
            for kept_item in kept:
                if int(item.get("classIndex") or 0) != int(kept_item.get("classIndex") or 0):
                    continue
                kept_bbox = kept_item.get("bbox")
                if not isinstance(kept_bbox, tuple) or len(kept_bbox) < 4:
                    continue
                if cls._bbox_iou(bbox, kept_bbox) > threshold:
                    should_keep = False
                    break
            if should_keep:
                kept.append(item)

        return kept

    @staticmethod
    def _bbox_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in first[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in second[:4]]
        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)
        intersection_width = max(0.0, intersection_x2 - intersection_x1)
        intersection_height = max(0.0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = first_area + second_area - intersection_area

        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area

    @classmethod
    def _draw_detection_items(
        cls,
        frame,
        items: list[dict[str, object]],
        *,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> None:
        for item in items:
            bbox = item.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
            class_name = cls._display_class_name(
                str(item.get("className") or "detection"),
                model_info=model_info,
                pipeline=pipeline,
            )
            confidence = float(item.get("confidence") or 0.0)
            label = f"{class_name} {confidence:.2f}"
            color = (0, 128, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y = max(y1, label_size[1] + baseline + 4)
            cv2.rectangle(
                frame,
                (x1, label_y - label_size[1] - baseline - 4),
                (x1 + label_size[0] + 8, label_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    @classmethod
    def _apply_display_labels(
        cls,
        result,
        *,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> None:
        names = getattr(result, "names", None)
        if not isinstance(names, dict):
            return

        display_names = {
            key: cls._display_class_name(str(value), model_info=model_info, pipeline=pipeline)
            for key, value in names.items()
        }
        try:
            result.names = display_names
        except AttributeError:
            names.update(display_names)

    @classmethod
    def _display_class_name(
        cls,
        class_name: str,
        *,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> str:
        if class_name.strip().lower() == "smoke" and cls._uses_thermal_model(
            model_info=model_info,
            pipeline=pipeline,
        ):
            return "fire"
        return class_name

    @staticmethod
    def _uses_thermal_model(
        *,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> bool:
        sensor_type = str(pipeline.get("sensorType") or "").strip().lower()
        model_id = str(model_info.get("id") or pipeline.get("currentModelId") or "").strip().lower()
        model_name = str(model_info.get("name") or "").strip().lower()

        return (
            sensor_type in {"thermal", "ir", "infrared"}
            or "thermal" in model_id
            or "thermal" in model_name
        )

    @staticmethod
    def _mask_polygons_for_result(result) -> list[list[tuple[float, float]]]:
        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xy", None)

        if not polygons:
            return []

        out: list[list[tuple[float, float]]] = []
        for polygon in polygons:
            try:
                out.append([(float(point[0]), float(point[1])) for point in polygon])
            except (TypeError, ValueError, IndexError):
                out.append([])
        return out

    @staticmethod
    def _target_pixel_for_detection(
        *,
        bbox: tuple[float, float, float, float],
        polygon: list[tuple[float, float]],
    ) -> dict[str, object]:
        x1, y1, x2, y2 = bbox

        if polygon:
            max_y = max(point[1] for point in polygon)
            bottom_points = [point for point in polygon if point[1] >= max_y - 2.0]
            if bottom_points:
                x = sum(point[0] for point in bottom_points) / len(bottom_points)
                return {
                    "x": round(x, 2),
                    "y": round(max_y, 2),
                    "method": "mask-bottom",
                }

        return {
            "x": round((x1 + x2) / 2.0, 2),
            "y": round((y1 + y2) / 2.0, 2),
            "method": "bbox-center",
        }

    def _detection_id(self, frame_at: str, index: int) -> str:
        seed = f"{self.pipeline_id}|{frame_at}|{self._model_id}|{index}".encode("utf-8")
        digest = hashlib.blake2s(seed, digest_size=8).hexdigest()
        return f"{self.pipeline_id}-{digest}"

    def _update_detections(self, detections: dict[str, object]) -> None:
        with self._lock:
            self._latest_detections = detections

    def _ensure_model(self, pipeline: dict[str, object]) -> dict[str, object] | None:
        requested_model_id = str(pipeline.get("currentModelId") or "").strip()
        desired_model_id = self.model_registry.canonical_id(requested_model_id)
        model_info = self.model_registry.get(desired_model_id)
        registry_revision = self.model_registry.revision
        catalog_error = self.model_registry.last_reload_error
        model_signature = self._model_info_signature(model_info)

        if (
            self._model
            and self._model_id == desired_model_id
            and model_info
            and self._model_loaded_signature == model_signature
        ):
            self._set_runtime(
                modelRegistryRevision=registry_revision,
                modelCatalogError=catalog_error,
            )
            return model_info

        now = time.monotonic()
        if now < self._model_retry_next_at:
            self._set_runtime(
                modelRegistryRevision=registry_revision,
                modelCatalogError=catalog_error,
            )
            return None

        if not model_info:
            self._discard_loaded_model()
            self._schedule_model_retry(
                f"model {requested_model_id or desired_model_id} not found",
                now=now,
                registry_revision=registry_revision,
                catalog_error=catalog_error,
            )
            return None
        if not model_info.get("weightsPresent"):
            self._discard_loaded_model()
            self._schedule_model_retry(
                f"weights missing for {desired_model_id}",
                now=now,
                registry_revision=registry_revision,
                catalog_error=catalog_error,
            )
            return None
        try:
            loaded_model = YOLO(str(model_info["weightsPath"]))
            self._model = loaded_model
            self._model_id = str(model_info["id"])
            self._model_name = str(model_info.get("name") or self._model_id)
            self._model_loaded_signature = model_signature
            self._model_registry_revision = registry_revision
            self._reset_model_retry_backoff()
            self._set_runtime(
                status="running",
                message=f"model loaded: {self._model_name}",
                modelRegistryRevision=registry_revision,
                modelCatalogError=catalog_error,
            )
            return model_info
        except Exception as error:
            self._discard_loaded_model()
            self._schedule_model_retry(
                f"could not load model {desired_model_id}: {error}",
                now=now,
                registry_revision=registry_revision,
                catalog_error=catalog_error,
            )
            return None

    @staticmethod
    def _model_info_signature(model_info: dict[str, object] | None) -> tuple[object, ...] | None:
        if not model_info:
            return None
        weights_path = Path(str(model_info.get("weightsPath") or ""))
        try:
            weights_stat = weights_path.stat()
            weights_signature: tuple[int, int] = (
                int(weights_stat.st_mtime_ns),
                int(weights_stat.st_size),
            )
        except OSError:
            weights_signature = (0, 0)
        return (
            str(model_info.get("id") or ""),
            str(weights_path),
            weights_signature,
            model_info.get("imageSize"),
            model_info.get("iouThreshold"),
            model_info.get("preprocessor"),
            model_info.get("compositeIntermediateWidth"),
            model_info.get("compositeNmsIou"),
        )

    def _schedule_model_retry(
        self,
        reason: str,
        *,
        now: float,
        registry_revision: int,
        catalog_error: str,
    ) -> None:
        retry_base = float(getattr(self.settings, "model_retry_seconds", 2.0))
        retry_max = max(
            retry_base,
            float(getattr(self.settings, "model_retry_max_seconds", 30.0)),
        )
        delay = min(
            max(self._model_retry_delay, retry_base),
            retry_max,
        )
        self._model_retry_count += 1
        self._model_retry_next_at = now + delay
        self._model_retry_delay = min(
            delay * 2.0,
            retry_max,
        )
        self._set_runtime(
            status="model_error",
            message=f"{reason}; retrying model in {delay:.1f}s",
            modelRetryCount=self._model_retry_count,
            modelRetryDelaySeconds=round(delay, 2),
            modelRegistryRevision=registry_revision,
            modelCatalogError=catalog_error,
        )

    def _reset_model_retry_backoff(self) -> None:
        self._model_retry_delay = float(getattr(self.settings, "model_retry_seconds", 2.0))
        self._model_retry_next_at = 0.0
        self._set_runtime(modelRetryDelaySeconds=0.0)

    def _discard_loaded_model(self) -> None:
        self._model = None
        self._model_id = ""
        self._model_name = ""
        self._model_loaded_signature = None

    def _handle_frame_processing_error(self, error: Exception) -> None:
        self._discard_loaded_model()
        self._clear_processed_preview()
        self._schedule_model_retry(
            f"frame processing failed: {error}",
            now=time.monotonic(),
            registry_revision=self.model_registry.revision,
            catalog_error=self.model_registry.last_reload_error,
        )

    def _encode_preview(self, frame) -> bytes | None:
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.preview_jpeg_quality)],
        )
        return encoded.tobytes() if success else None

    def _update_raw_preview(self, frame) -> None:
        if not self.settings.legacy_mjpeg_enabled:
            with self._lock:
                if self._latest_raw_frame_jpeg is not None:
                    self._latest_raw_frame_jpeg = None
                    self._raw_frame_sequence += 1
                self._runtime["latestRawFrameAvailable"] = False
            return
        encoded = self._encode_preview(frame)
        with self._lock:
            self._latest_raw_frame_jpeg = encoded
            self._raw_frame_sequence += 1
            self._runtime["latestRawFrameAvailable"] = encoded is not None
            self._runtime["latestRawFrameContentType"] = "image/jpeg"

    def _update_preview(self, frame) -> None:
        if not self.settings.legacy_mjpeg_enabled:
            with self._lock:
                if self._latest_processed_frame_jpeg is not None:
                    self._latest_processed_frame_jpeg = None
                    self._processed_frame_sequence += 1
                self._runtime["latestFrameAvailable"] = False
                self._runtime["latestProcessedFrameAvailable"] = False
            return
        encoded = self._encode_preview(frame)
        with self._lock:
            self._latest_processed_frame_jpeg = encoded
            self._processed_frame_sequence += 1
            processed_available = encoded is not None
            self._runtime["latestFrameAvailable"] = processed_available
            self._runtime["latestFrameContentType"] = "image/jpeg"
            self._runtime["latestProcessedFrameAvailable"] = processed_available
            self._runtime["latestProcessedFrameContentType"] = "image/jpeg"

    def _publish_processed_frame(self, frame, *, fps: float, source_url: str) -> None:
        fallback_url = self._legacy_mjpeg_fallback_url()
        fallback_status = (
            "legacy MJPEG fallback active"
            if fallback_url
            else "legacy MJPEG is disabled"
        )
        if not self.settings.processed_rtsp_enabled:
            self._stop_processed_publisher()
            self._set_runtime(
                processedPublisherError=f"publisher disabled by configuration; {fallback_status}"
            )
            return

        output_url = self.settings.processed_publish_url(self.pipeline_id, source_url)
        safe_output_url = sanitize_stream_url(output_url)
        if not output_url:
            self._stop_processed_publisher()
            self._set_runtime(
                processedPublisherUrl="",
                processedPublisherError=(
                    "publisher URL is not configured; set PYRONE_PROCESSED_RTMP_URL_TEMPLATE; "
                    f"{fallback_status}"
                ),
            )
            return

        height, width = frame.shape[:2]
        publish_fps = min(
            max(float(fps or 1.0), 1.0),
            self.settings.processed_publish_max_fps,
        )
        signature = (
            self.settings.processed_publish_transport,
            output_url,
            int(width),
            int(height),
            round(publish_fps, 3),
        )
        now = time.monotonic()
        publisher = self._processed_publisher

        if publisher and self._processed_publisher_signature != signature:
            self._stop_processed_publisher(reset_retry=True)
            publisher = None

        if publisher and not publisher.is_running():
            error = publisher.last_error or "processed publisher stopped"
            self._record_processed_publisher_failure(error)
            self._stop_processed_publisher(reset_retry=False)
            self._schedule_publisher_retry(now)
            self._set_runtime(processedPublisherError=error)
            publisher = None

        if publisher is None:
            if now < self._processed_publisher_next_retry_at:
                self._set_runtime(
                    processedStreamReady=False,
                    processedStreamUrl=fallback_url,
                    processedPublisherUrl=safe_output_url,
                )
                return

            try:
                _terminate_marked_publisher_processes(output_url)
                publisher = FfmpegFramePublisher(
                    ffmpeg_path=self.settings.ffmpeg_path,
                    output_url=output_url,
                    transport=self.settings.processed_publish_transport,
                    width=width,
                    height=height,
                    fps=publish_fps,
                    bitrate=self.settings.processed_rtsp_bitrate,
                    bufsize=self.settings.processed_rtsp_bufsize,
                    preset=self.settings.processed_rtsp_preset,
                    write_timeout_seconds=self.settings.processed_rtsp_write_timeout_seconds,
                    startup_timeout_seconds=(
                        self.settings.processed_publish_startup_timeout_seconds
                    ),
                    ready_after_seconds=self.settings.processed_publish_ready_after_seconds,
                    ready_freshness_seconds=(
                        self.settings.processed_publish_ready_freshness_seconds
                    ),
                    stable_reset_seconds=(
                        self.settings.processed_publish_stable_reset_seconds
                    ),
                    stale_frame_seconds=self.settings.processed_publish_stale_seconds,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                safe_error = str(error).replace(output_url, safe_output_url)
                self._record_processed_publisher_failure(safe_error)
                self._schedule_publisher_retry(now)
                self._set_runtime(
                    processedStreamReady=False,
                    processedStreamUrl=fallback_url,
                    processedPublisherPid=None,
                    processedPublisherUrl=safe_output_url,
                    processedPublisherError=safe_error,
                )
                return

            self._processed_publisher = publisher
            self._processed_publisher_signature = signature
            self._processed_publisher_start_count += 1
            self._processed_publisher_next_retry_at = 0.0

        payload = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        accepted = publisher.write(payload)
        if not accepted:
            error = publisher.last_error or "processed publisher rejected a frame"
            self._record_processed_publisher_failure(error)
            self._stop_processed_publisher(reset_retry=False)
            self._schedule_publisher_retry(now)
            self._set_runtime(processedPublisherError=error)
            return

        if not publisher.is_running():
            error = publisher.last_error or "processed publisher stopped during startup"
            self._record_processed_publisher_failure(error)
            self._stop_processed_publisher(reset_retry=False)
            self._schedule_publisher_retry(now)
            self._set_runtime(processedPublisherError=error)
            return

        ready = publisher.ready
        stable = publisher.stable

        self._reset_publisher_backoff_if_stable(stable)

        with self._lock:
            was_ready = bool(self._runtime.get("processedStreamReady"))
            self._runtime.update(
                {
                    "processedStreamReady": ready,
                    "processedStreamUrl": safe_output_url if ready else fallback_url,
                    "processedPublisherPid": publisher.pid,
                    "processedPublisherTransport": self.settings.processed_publish_transport,
                    "processedPublisherUrl": safe_output_url,
                    "processedPublisherError": publisher.last_error,
                    "processedPublisherRestartCount": max(
                        0,
                        self._processed_publisher_start_count - 1,
                    ),
                    "processedPublisherFramesWritten": publisher.frames_written,
                    "processedPublisherDroppedFrames": publisher.frames_dropped,
                    "processedPublisherStable": stable,
                    "processedPublisherUptimeSeconds": round(
                        publisher.uptime_seconds,
                        2,
                    ),
                }
            )
            if ready and not was_ready:
                self._runtime["processedStreamRevision"] = (
                    int(self._runtime.get("processedStreamRevision") or 0) + 1
                )
                self._runtime["processedStreamStartedAt"] = utc_now()

    def _schedule_publisher_retry(self, now: float) -> None:
        self._processed_publisher_next_retry_at = now + self._processed_publisher_retry_delay
        self._processed_publisher_retry_delay = min(
            self._processed_publisher_retry_delay * 2.0,
            self.settings.processed_publish_retry_max_seconds,
        )

    def _reset_publisher_backoff_if_stable(self, stable: bool) -> None:
        if not stable:
            return
        self._processed_publisher_retry_delay = self.settings.processed_publish_retry_seconds
        self._processed_publisher_next_retry_at = 0.0

    def _record_processed_publisher_failure(self, reason: str) -> None:
        normalized_reason = " ".join(str(reason or "processed publisher failed").split())[-1000:]
        failed_at = utc_now()
        with self._lock:
            failure_count = int(self._runtime.get("processedPublisherFailureCount") or 0) + 1
            self._runtime.update(
                {
                    "processedPublisherLastFailureReason": normalized_reason,
                    "processedPublisherLastFailureAt": failed_at,
                    "processedPublisherFailureCount": failure_count,
                }
            )
        safe_console_log(
            f"[drone-worker] processed publisher failure "
            f"drone={self.pipeline_id} count={failure_count} at={failed_at}: "
            f"{normalized_reason}"
        )

    def _stop_processed_publisher(self, *, reset_retry: bool = True) -> None:
        publisher = self._processed_publisher
        self._processed_publisher = None
        self._processed_publisher_signature = None
        if publisher:
            publisher.release()
        if reset_retry:
            self._processed_publisher_next_retry_at = 0.0
            self._processed_publisher_retry_delay = self.settings.processed_publish_retry_seconds
        self._set_runtime(
            processedStreamReady=False,
            processedPublisherPid=None,
            processedPublisherStable=False,
            processedStreamUrl=self._legacy_mjpeg_fallback_url(),
        )

    def _legacy_mjpeg_fallback_url(self) -> str:
        if not self.settings.legacy_mjpeg_enabled:
            return ""
        return f"/v1/pipelines/{self.pipeline_id}/stream.mjpg"

    def _clear_processed_preview(self) -> None:
        self._stop_processed_publisher()
        with self._lock:
            if self._latest_processed_frame_jpeg is not None:
                self._latest_processed_frame_jpeg = None
                self._processed_frame_sequence += 1
            self._runtime["latestFrameAvailable"] = False
            self._runtime["latestProcessedFrameAvailable"] = False
            self._latest_detections = self._empty_detections()

    def _clear_preview(self) -> None:
        self._stop_processed_publisher()
        with self._lock:
            if self._latest_processed_frame_jpeg is not None:
                self._latest_processed_frame_jpeg = None
                self._processed_frame_sequence += 1
            if self._latest_raw_frame_jpeg is not None:
                self._latest_raw_frame_jpeg = None
                self._raw_frame_sequence += 1
            self._runtime["latestFrameAvailable"] = False
            self._runtime["latestProcessedFrameAvailable"] = False
            self._runtime["latestRawFrameAvailable"] = False
            self._latest_detections = self._empty_detections()

    def _update_runtime_after_frame(
        self,
        *,
        frame_at: str,
        frame_width: int,
        frame_height: int,
        source_fps: float,
        inference_ms: float,
        detection_count: int,
        max_confidence: float,
        event: EventUpdate,
        second_stage_summary: SecondStageSummary,
        frame_age_ms: float,
    ) -> None:
        self._fps_window.append(time.monotonic())
        processing_fps = 0.0
        if len(self._fps_window) >= 2:
            elapsed = self._fps_window[-1] - self._fps_window[0]
            if elapsed > 0:
                processing_fps = (len(self._fps_window) - 1) / elapsed

        with self._lock:
            frames_processed = int(self._runtime["framesProcessed"]) + 1
            detections_total = int(self._runtime["detectionsTotal"]) + int(detection_count)
            previous_avg = self._runtime["avgInferenceMs"]
            avg_inference_ms = (
                inference_ms
                if previous_avg is None
                else ((float(previous_avg) * (frames_processed - 1)) + inference_ms) / frames_processed
            )
            previous_age_avg = self._runtime.get("avgInferenceFrameAgeMs")
            avg_frame_age_ms = (
                frame_age_ms
                if previous_age_avg is None
                else (
                    (float(previous_age_avg) * (frames_processed - 1)) + frame_age_ms
                )
                / frames_processed
            )
            max_frame_age_ms = max(
                float(self._runtime.get("maxInferenceFrameAgeMs") or 0.0),
                frame_age_ms,
            )
            self._runtime.update(
                {
                    "status": "running",
                    "message": (
                        f"{detection_count} detections"
                        if detection_count
                        else f"idle, model {self._model_name or self._model_id}"
                    ),
                    "sourceOpened": True,
                    "singleIngestHealthy": int(self._runtime.get("activeRtspConnections") or 0) <= 1,
                    "frameWidth": frame_width,
                    "frameHeight": frame_height,
                    "sourceFps": round(source_fps, 2) if source_fps else None,
                    "processingFps": round(processing_fps, 2),
                    "framesProcessed": frames_processed,
                    "detectionsTotal": detections_total,
                    "avgInferenceMs": round(avg_inference_ms, 2),
                    "lastInferenceFrameAgeMs": round(frame_age_ms, 2),
                    "avgInferenceFrameAgeMs": round(avg_frame_age_ms, 2),
                    "maxInferenceFrameAgeMs": round(max_frame_age_ms, 2),
                    "lastFrameAt": frame_at,
                    "currentEvent": event.active_event,
                    "secondStageEnabled": second_stage_summary.enabled,
                    "secondStageApplied": second_stage_summary.applied,
                    "secondStageAvailable": second_stage_summary.available,
                    "secondStageSentToClassifier": second_stage_summary.sent_to_classifier,
                    "secondStageRejectedByClassifier": second_stage_summary.rejected_by_classifier,
                    "secondStageDroppedLow": second_stage_summary.dropped_low,
                    "secondStageKeptAfterClassifier": second_stage_summary.kept_after_classifier,
                    "secondStageLastMs": round(float(second_stage_summary.elapsed_ms), 2),
                    "secondStageLastReason": second_stage_summary.reason,
                }
            )
            if detection_count:
                self._runtime["lastDetectionAt"] = frame_at
                self._runtime["lastDetectionCount"] = int(detection_count)
                self._runtime["lastDetectionConfidence"] = round(float(max_confidence), 4)

    def _store_recording(self, recording: dict[str, object]) -> None:
        with self._lock:
            self._runtime["lastRecording"] = dict(recording)
        if callable(self.on_recording_saved):
            try:
                self.on_recording_saved(recording)
            except Exception:
                pass

    def _resolve_device(self) -> str:
        if self.settings.inference_device != "auto":
            return self.settings.inference_device
        return "0" if torch.cuda.is_available() else "cpu"

    def _pipeline_snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._pipeline)

    def _set_active_capture(self, capture) -> None:
        with self._lock:
            self._active_capture = capture

    def _clear_active_capture(self, capture) -> None:
        with self._lock:
            if self._active_capture is capture:
                self._active_capture = None

    def _release_active_capture(self) -> None:
        with self._lock:
            capture = self._active_capture
            self._active_capture = None
        if capture:
            capture.release()

    def _set_runtime(self, **updates) -> None:
        with self._lock:
            self._runtime.update(updates)

    @staticmethod
    def _sanitize_source_url(source: str) -> str:
        value = str(source or "").strip()
        if not value.lower().startswith("rtsp://"):
            return value
        scheme, _, rest = value.partition("://")
        if "@" not in rest:
            return value
        credentials, _, host = rest.partition("@")
        if ":" not in credentials:
            return f"{scheme}://{credentials}@{host}"
        username, _, _password = credentials.partition(":")
        return f"{scheme}://{username}:******@{host}"

    def _mark_source_open(
        self,
        *,
        source: str,
        source_type: str,
        capture_backend: str,
        capture_pid: int | None,
        source_fps: float,
    ) -> None:
        with self._lock:
            source_open_count = int(self._runtime.get("sourceOpenCount") or 0) + 1
            active_rtsp_connections = 1 if source_type == "rtsp" else 0
            self._runtime.update(
                {
                    "status": "starting",
                    "message": "source opened",
                    "sourceOpened": True,
                    "sourceType": source_type,
                    "sourceUrl": self._sanitize_source_url(source),
                    "captureBackend": capture_backend,
                    "capturePid": capture_pid,
                    "sourceFps": source_fps,
                    "activeRtspConnections": active_rtsp_connections,
                    "maxRtspConnections": 1,
                    "singleIngestHealthy": active_rtsp_connections <= 1,
                    "sourceSessionId": f"{self.pipeline_id}-source-{source_open_count}",
                    "sourceOpenCount": source_open_count,
                    "sourceReconnectCount": max(0, source_open_count - 1),
                    "sourceStable": False,
                    "lastSourceOpenAt": utc_now(),
                    "lastSourceError": "",
                }
            )

    def _mark_source_closed(self, *, last_error: str) -> None:
        with self._lock:
            self._runtime.update(
                {
                    "sourceOpened": False,
                    "activeRtspConnections": 0,
                    "singleIngestHealthy": True,
                    "capturePid": None,
                    "lastSourceCloseAt": utc_now(),
                    "lastSourceError": str(last_error or self._runtime.get("lastSourceError") or ""),
                }
            )

    @staticmethod
    def _overlay_status(frame, *, model_name: str, detection_count: int, max_confidence: float, sensor_type: str) -> None:
        _ = model_name
        watermark = _load_pyrone_watermark_image()
        if watermark is not None and frame.size:
            frame_height, frame_width = frame.shape[:2]
            size = max(72, min(140, int(min(frame_width, frame_height) * 0.18)))
            resized = cv2.resize(watermark, (size, size), interpolation=cv2.INTER_AREA)
            logo_x = 14
            logo_y = 10
            max_width = max(0, frame_width - logo_x)
            max_height = max(0, frame_height - logo_y)
            resized = resized[:max_height, :max_width]

            if resized.size:
                if resized.ndim == 2:
                    logo_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                    alpha = np.ones(resized.shape[:2], dtype=np.float32)
                else:
                    logo_bgr = resized[:, :, :3]
                    alpha = np.ones(resized.shape[:2], dtype=np.float32)

                if resized.ndim == 3 and resized.shape[2] == 4:
                    alpha = resized[:, :, 3].astype(np.float32) / 255.0

                alpha = (alpha * 0.16)[:, :, None]
                roi = frame[logo_y : logo_y + resized.shape[0], logo_x : logo_x + resized.shape[1]]
                blended = (logo_bgr.astype(np.float32) * alpha) + (roi.astype(np.float32) * (1.0 - alpha))
                frame[logo_y : logo_y + resized.shape[0], logo_x : logo_x + resized.shape[1]] = blended.astype(np.uint8)

        lines = [
            f"Sensor: {sensor_type or 'unknown'}",
            (
                f"Detections: {detection_count}  Max conf: {max_confidence:.2f}"
                if detection_count
                else "Detections: 0"
            ),
        ]
        for index, text in enumerate(lines):
            y = 30 + (index * 26)
            cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (18, 18, 18), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
