from __future__ import annotations

from collections import deque
import hashlib
import queue
import select
import threading
import time
from pathlib import Path
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from service.event_detector import EventDetector
from service.model_registry import ModelRegistry
from service.nvr_store import NvrStore
from service.recorder import EventRecorder
from service.schemas import utc_now
from service.settings import ServiceSettings


class _FfmpegMjpegReader:
    def __init__(
        self,
        *,
        ffmpeg_path: str,
        source: str,
        read_timeout_seconds: float = 5.0,
    ) -> None:
        self.source = source
        self._process: subprocess.Popen[bytes] | None = None
        self._buffer = bytearray()
        self._read_timeout_seconds = max(0.5, float(read_timeout_seconds))
        self._duplicate_frame_count = 0
        self._last_frame_digest: bytes | None = None
        self._last_unique_frame_at = time.monotonic()
        self._stale_frame_timeout_seconds = max(self._read_timeout_seconds * 2.0, 6.0)
        self._stale_frame_count = 30
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
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            source,
            "-an",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "5",
            "pipe:1",
        ]
        self._process = subprocess.Popen(
            self._command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def isOpened(self) -> bool:  # noqa: N802
        return bool(self._process and self._process.stdout and self._process.poll() is None)

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    def read(self):
        if not self._process or not self._process.stdout:
            return False, None

        stdout = self._process.stdout
        start_marker = b"\xff\xd8"
        end_marker = b"\xff\xd9"

        while True:
            start_index = self._buffer.find(start_marker)
            if start_index != -1:
                end_index = self._buffer.find(end_marker, start_index + 2)
                if end_index != -1:
                    jpeg_bytes = bytes(self._buffer[start_index : end_index + 2])
                    del self._buffer[: end_index + 2]
                    if self._is_stale_duplicate_frame(jpeg_bytes):
                        return False, None
                    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    return (frame is not None), frame

            try:
                ready, _, _ = select.select([stdout], [], [], self._read_timeout_seconds)
            except (OSError, ValueError):
                return False, None
            if not ready:
                if time.monotonic() < self._startup_deadline:
                    continue
                # ffmpeg can hang with an open socket and stop producing bytes.
                # Return False so the worker reconnects the source automatically.
                return False, None

            chunk = stdout.read(4 * 1024)
            if not chunk:
                return False, None
            self._buffer.extend(chunk)
            if self._startup_deadline:
                self._startup_deadline = 0.0
            if len(self._buffer) > 8 * 1024 * 1024:
                del self._buffer[: 4 * 1024 * 1024]

    def _is_stale_duplicate_frame(self, jpeg_bytes: bytes) -> bool:
        digest = hashlib.blake2s(jpeg_bytes, digest_size=8).digest()
        now = time.monotonic()

        if digest != self._last_frame_digest:
            self._last_frame_digest = digest
            self._last_unique_frame_at = now
            self._duplicate_frame_count = 0
            return False

        self._duplicate_frame_count += 1
        return (
            self._duplicate_frame_count >= self._stale_frame_count
            and now - self._last_unique_frame_at >= self._stale_frame_timeout_seconds
        )

    def release(self) -> None:
        process = self._process
        self._process = None
        if not process:
            return
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        finally:
            if process.stdout:
                process.stdout.close()


class _RtspFramePublisher:
    def __init__(
        self,
        *,
        ffmpeg_path: str,
        output_url: str,
        width: int,
        height: int,
        fps: float,
        bitrate: str,
        bufsize: str,
        preset: str,
        write_timeout_seconds: float,
    ) -> None:
        self.output_url = output_url
        self.width = int(width)
        self.height = int(height)
        self.fps = max(float(fps or 1.0), 1.0)
        self.write_timeout_seconds = max(1.0, float(write_timeout_seconds or 3.0))
        self._process: subprocess.Popen[bytes] | None = None
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._failed = False
        self._last_error = ""
        self._write_started_at = 0.0
        self._thread: threading.Thread | None = None
        gop = max(2, int(round(self.fps * 2)))
        self._command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.2f}",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-b:v",
            bitrate,
            "-maxrate",
            bitrate,
            "-bufsize",
            bufsize,
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            output_url,
        ]
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"pyrone-rtsp-publisher-{self._process.pid}",
            daemon=True,
        )
        self._thread.start()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    def is_running(self) -> bool:
        process = self._process
        if not process or not process.stdin or process.poll() is not None:
            return False

        with self._lock:
            if self._failed:
                return False

            write_started_at = self._write_started_at

        if write_started_at and time.monotonic() - write_started_at > self.write_timeout_seconds:
            self.release()
            return False

        return True

    def write(self, frame) -> bool:
        if not self.is_running() or not self._process or not self._process.stdin:
            return False
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            return False

        try:
            payload = np.ascontiguousarray(frame).tobytes()
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(payload)
            return True
        except (BrokenPipeError, OSError, ValueError, queue.Full):
            self.release()
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            process = self._process
            if not process or not process.stdin or process.poll() is not None:
                self._mark_failed("processed stream process stopped")
                return

            try:
                with self._lock:
                    self._write_started_at = time.monotonic()
                process.stdin.write(payload)
            except (BrokenPipeError, OSError, ValueError) as error:
                self._mark_failed(str(error))
                return
            finally:
                with self._lock:
                    self._write_started_at = 0.0

    def _mark_failed(self, error: str) -> None:
        with self._lock:
            self._failed = True
            self._last_error = error

    def release(self) -> None:
        process = self._process
        self._process = None
        self._stop_event.set()

        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if not process:
            return
        try:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        finally:
            thread = self._thread
            if thread and thread is not threading.current_thread():
                thread.join(timeout=1.0)


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
        self._thread: threading.Thread | None = None
        self._latest_processed_frame_jpeg: bytes | None = None
        self._latest_raw_frame_jpeg: bytes | None = None
        self._processed_publisher: _RtspFramePublisher | None = None
        self._processed_publisher_signature: tuple[str, int, int, int] | None = None
        self._active_capture = None
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
            "latestProcessedFrameAvailable": False,
            "latestProcessedFrameContentType": "image/jpeg",
            "latestRawFrameAvailable": False,
            "latestRawFrameContentType": "image/jpeg",
            "processedStreamReady": False,
            "processedStreamUrl": self.settings.processed_stream_url(str(pipeline.get("droneId") or "")),
            "processedPublisherPid": None,
            "device": self._resolve_device(),
        }
        self._model_id = ""
        self._model_name = ""
        self._model: YOLO | None = None
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

    def latest_frame(self) -> bytes | None:
        return self.latest_processed_frame()

    def latest_processed_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_processed_frame_jpeg

    def latest_raw_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_raw_frame_jpeg

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
                time.sleep(0.5)
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
                time.sleep(self.settings.reconnect_delay_seconds)
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
                    time.sleep(self.settings.reconnect_delay_seconds)
                continue

            capture = self._open_capture(source)
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
                time.sleep(self.settings.reconnect_delay_seconds)
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
            last_processed_at = 0.0

            try:
                while not self._stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        self._stop_processed_publisher()
                        self._clear_preview()
                        self._set_runtime(
                            status="waiting_source",
                            message="source frame read failed",
                            sourceOpened=False,
                            activeRtspConnections=0,
                            singleIngestHealthy=True,
                            lastSourceError="source frame read failed",
                        )
                        break

                    now = time.monotonic()
                    if last_processed_at and now - last_processed_at < min_frame_interval:
                        continue
                    last_processed_at = now

                    pipeline = self._pipeline_snapshot()
                    if not pipeline.get("analyticsEnabled", True):
                        break
                    min_frame_interval = 1.0 / max(
                        0.5,
                        float(pipeline.get("processingFps") or self.settings.processing_fps),
                    )

                    self._process_frame(frame, source_fps=source_fps, pipeline=pipeline)
            finally:
                self._clear_active_capture(capture)
                capture.release()
            self._mark_source_closed(last_error="")
            if not self._stop_event.is_set():
                time.sleep(self.settings.reconnect_delay_seconds)

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
                self._process_frame(frame, source_fps=pipeline_fps, pipeline=pipeline)
                time.sleep(1.0 / max(0.5, pipeline_fps))

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
            ffmpeg_reader = _FfmpegMjpegReader(
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
        if isinstance(capture, _FfmpegMjpegReader):
            return "ffmpeg-mjpeg-reader"
        return "opencv-videocapture"

    @staticmethod
    def _capture_pid(capture) -> int | None:
        if isinstance(capture, _FfmpegMjpegReader):
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

    def _process_frame(self, frame, *, source_fps: float, pipeline: dict[str, object]) -> None:
        frame = self._prepare_frame(frame)
        self._update_raw_preview(frame)

        model_info = self._ensure_model(pipeline)
        if not model_info:
            self._stop_processed_publisher()
            self._clear_processed_preview()
            time.sleep(self.settings.reconnect_delay_seconds)
            return

        frame_height, frame_width = frame.shape[:2]
        effective_fps = max(float(pipeline.get("processingFps") or self.settings.processing_fps), 1.0)
        self._recorder.configure(frame_width=frame_width, frame_height=frame_height, fps=effective_fps)

        inference_started = perf_counter()
        confidence_threshold = float(pipeline.get("confidenceThreshold") or self.settings.inference_confidence)
        result = self._model.predict(
            source=frame,
            conf=confidence_threshold,
            imgsz=int(model_info.get("imageSize") or self.settings.inference_image_size),
            device=self._runtime["device"],
            verbose=False,
        )[0]
        inference_ms = (perf_counter() - inference_started) * 1000.0

        detection_count = 0 if result.boxes is None else len(result.boxes)
        max_confidence = max(result.boxes.conf.tolist()) if detection_count and result.boxes is not None else 0.0
        positive = detection_count > 0
        event = self._detector.update(
            positive=positive,
            detection_count=detection_count,
            max_confidence=max_confidence,
        )

        annotated = result.plot() if detection_count else frame.copy()
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
                },
                fps=effective_fps,
            )
            if recording:
                self._store_recording(recording)

        self._update_preview(annotated)
        self._publish_processed_frame(annotated, fps=effective_fps)
        self._update_runtime_after_frame(
            frame_width=frame_width,
            frame_height=frame_height,
            source_fps=source_fps,
            inference_ms=inference_ms,
            detection_count=detection_count,
            max_confidence=max_confidence,
            event=event,
        )

    def _ensure_model(self, pipeline: dict[str, object]) -> dict[str, object] | None:
        desired_model_id = str(pipeline.get("currentModelId") or "")
        if self._model and self._model_id == desired_model_id:
            return self.model_registry.get(self._model_id)

        model_info = self.model_registry.get(desired_model_id)
        if not model_info:
            self._set_runtime(status="model_error", message=f"model {desired_model_id} not found")
            return None
        if not model_info.get("weightsPresent"):
            self._set_runtime(
                status="model_error",
                message=f"weights missing for {desired_model_id}",
            )
            return None
        try:
            self._model = YOLO(str(model_info["weightsPath"]))
            self._model_id = str(model_info["id"])
            self._model_name = str(model_info.get("name") or self._model_id)
            self._set_runtime(status="running", message=f"model loaded: {self._model_name}")
            return model_info
        except Exception as error:
            self._set_runtime(status="model_error", message=f"could not load model: {error}")
            self._model = None
            return None

    def _encode_preview(self, frame) -> bytes | None:
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.preview_jpeg_quality)],
        )
        return encoded.tobytes() if success else None

    def _update_raw_preview(self, frame) -> None:
        encoded = self._encode_preview(frame)
        with self._lock:
            self._latest_raw_frame_jpeg = encoded
            self._runtime["latestRawFrameAvailable"] = encoded is not None
            self._runtime["latestRawFrameContentType"] = "image/jpeg"

    def _update_preview(self, frame) -> None:
        encoded = self._encode_preview(frame)
        with self._lock:
            self._latest_processed_frame_jpeg = encoded
            processed_available = encoded is not None
            self._runtime["latestFrameAvailable"] = processed_available
            self._runtime["latestFrameContentType"] = "image/jpeg"
            self._runtime["latestProcessedFrameAvailable"] = processed_available
            self._runtime["latestProcessedFrameContentType"] = "image/jpeg"

    def _processed_stream_url(self) -> str:
        return self.settings.processed_stream_url(self.pipeline_id)

    def _stop_processed_publisher(self) -> None:
        publisher = self._processed_publisher
        self._processed_publisher = None
        self._processed_publisher_signature = None
        if publisher:
            publisher.release()
        self._set_runtime(
            processedStreamReady=False,
            processedPublisherPid=None,
            processedStreamUrl=self._processed_stream_url(),
        )

    def _publish_processed_frame(self, frame, *, fps: float) -> None:
        if not self.settings.processed_rtsp_enabled:
            self._stop_processed_publisher()
            return

        frame_height, frame_width = frame.shape[:2]
        rounded_fps = max(1, int(round(float(fps or self.settings.processing_fps))))
        output_url = self._processed_stream_url()
        signature = (output_url, frame_width, frame_height, rounded_fps)

        if (
            self._processed_publisher_signature != signature
            or not self._processed_publisher
            or not self._processed_publisher.is_running()
        ):
            self._stop_processed_publisher()
            try:
                self._processed_publisher = _RtspFramePublisher(
                    ffmpeg_path=self.settings.ffmpeg_path,
                    output_url=output_url,
                    width=frame_width,
                    height=frame_height,
                    fps=float(rounded_fps),
                    bitrate=self.settings.processed_rtsp_bitrate,
                    bufsize=self.settings.processed_rtsp_bufsize,
                    preset=self.settings.processed_rtsp_preset,
                    write_timeout_seconds=self.settings.processed_rtsp_write_timeout_seconds,
                )
                self._processed_publisher_signature = signature
            except Exception as error:
                self._set_runtime(
                    processedStreamReady=False,
                    processedPublisherPid=None,
                    lastSourceError=f"processed stream publisher failed: {error}",
                )
                return

        if not self._processed_publisher.write(frame):
            self._stop_processed_publisher()
            self._set_runtime(lastSourceError="processed stream publisher stopped")
            return

        self._set_runtime(
            processedStreamReady=True,
            processedStreamUrl=output_url,
            processedPublisherPid=self._processed_publisher.pid,
        )

    def _clear_processed_preview(self) -> None:
        self._stop_processed_publisher()
        with self._lock:
            self._latest_processed_frame_jpeg = None
            self._runtime["latestFrameAvailable"] = False
            self._runtime["latestProcessedFrameAvailable"] = False

    def _clear_preview(self) -> None:
        self._stop_processed_publisher()
        with self._lock:
            self._latest_processed_frame_jpeg = None
            self._latest_raw_frame_jpeg = None
            self._runtime["latestFrameAvailable"] = False
            self._runtime["latestProcessedFrameAvailable"] = False
            self._runtime["latestRawFrameAvailable"] = False

    def _update_runtime_after_frame(
        self,
        *,
        frame_width: int,
        frame_height: int,
        source_fps: float,
        inference_ms: float,
        detection_count: int,
        max_confidence: float,
        event: EventUpdate,
    ) -> None:
        now_iso = utc_now()
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
                    "lastFrameAt": now_iso,
                    "currentEvent": event.active_event,
                }
            )
            if detection_count:
                self._runtime["lastDetectionAt"] = now_iso
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
        lines = [
            f"Model: {model_name or '-'}",
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
