from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import parse_qs, urlparse

from service.model_registry import ModelRegistry
from service.nvr_store import NvrStore
from service.pipeline_manager import PipelineManager
from service.retention import RetentionManager
from service.settings import ServiceSettings


class AnalyticsServiceApp:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.model_registry = ModelRegistry(settings)
        self.nvr_store = NvrStore(settings)
        self.retention_manager = RetentionManager(settings, self.nvr_store)
        self.pipeline_manager = PipelineManager(
            settings,
            self.model_registry,
            self.nvr_store,
            self.retention_manager,
        )

    def dispatch(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/health" and handler.command == "GET":
            pipelines = self.pipeline_manager.list_pipelines()
            active_rtsp_connections = 0
            pipelines_source_opened = 0
            pipelines_over_limit = 0

            for pipeline in pipelines:
                runtime = pipeline.get("runtime") if isinstance(pipeline, dict) else {}
                runtime_record = runtime if isinstance(runtime, dict) else {}
                source_opened = bool(runtime_record.get("sourceOpened"))
                active_connections = int(runtime_record.get("activeRtspConnections") or 0)

                if source_opened:
                    pipelines_source_opened += 1
                active_rtsp_connections += max(active_connections, 0)
                if active_connections > 1:
                    pipelines_over_limit += 1

            return self._send_json(
                handler,
                {
                    "ok": True,
                    "service": "pyrone-analytics",
                    "storage": self.nvr_store.storage_summary(),
                    "pipelineCount": len(pipelines),
                    "singleIngest": {
                        "activeRtspConnections": active_rtsp_connections,
                        "pipelinesSourceOpened": pipelines_source_opened,
                        "pipelinesOverLimit": pipelines_over_limit,
                        "healthy": pipelines_over_limit == 0,
                    },
                },
            )

        if path == "/v1/models" and handler.command == "GET":
            return self._send_json(handler, {"items": self.model_registry.list_models()})

        if path == "/v1/storage" and handler.command == "GET":
            retention = self.retention_manager.enforce()
            return self._send_json(handler, retention)

        if path == "/v1/pipelines" and handler.command == "GET":
            return self._send_json(handler, {"items": self.pipeline_manager.list_pipelines()})

        pipeline_match = re.fullmatch(r"/v1/pipelines/([^/]+)", path)
        if pipeline_match:
            drone_id = pipeline_match.group(1)
            if handler.command == "GET":
                pipeline = self.pipeline_manager.get_pipeline(drone_id)
                if pipeline is None:
                    return self._send_json(handler, {"error": "pipeline not found"}, status=404)
                return self._send_json(handler, pipeline)
            if handler.command == "PUT":
                payload = self._read_json(handler)
                pipeline = self.pipeline_manager.upsert_pipeline(drone_id, payload)
                return self._send_json(handler, pipeline)
            if handler.command == "DELETE":
                deleted = self.pipeline_manager.delete_pipeline(drone_id)
                if deleted is None:
                    return self._send_json(handler, {"error": "pipeline not found"}, status=404)
                return self._send_json(handler, {"ok": True, "pipeline": deleted})
            return self._send_json(handler, {"error": "method not allowed"}, status=405)

        pipeline_state_match = re.fullmatch(r"/v1/pipelines/([^/]+)/state", path)
        if pipeline_state_match:
            if handler.command != "POST":
                return self._send_json(handler, {"error": "method not allowed"}, status=405)
            drone_id = pipeline_state_match.group(1)
            try:
                pipeline = self.pipeline_manager.update_runtime_state(drone_id, self._read_json(handler))
            except KeyError:
                return self._send_json(handler, {"error": "pipeline not found"}, status=404)
            return self._send_json(handler, pipeline)

        pipeline_restart_match = re.fullmatch(r"/v1/pipelines/([^/]+)/restart", path)
        if pipeline_restart_match:
            if handler.command != "POST":
                return self._send_json(handler, {"error": "method not allowed"}, status=405)
            drone_id = pipeline_restart_match.group(1)
            try:
                pipeline = self.pipeline_manager.restart_pipeline(drone_id)
            except KeyError:
                return self._send_json(handler, {"error": "pipeline not found"}, status=404)
            return self._send_json(handler, {"ok": True, "pipeline": pipeline})

        frame_match = re.fullmatch(r"/v1/pipelines/([^/]+)/frame\.jpg", path)
        if frame_match:
            if handler.command != "GET":
                return self._send_json(handler, {"error": "method not allowed"}, status=405)
            frame = self.pipeline_manager.latest_processed_frame(frame_match.group(1))
            if frame is None:
                return self._send_json(handler, {"error": "frame not available"}, status=404)
            return self._send_bytes(handler, frame, content_type="image/jpeg")

        frame_raw_match = re.fullmatch(r"/v1/pipelines/([^/]+)/frame\.raw\.jpg", path)
        if frame_raw_match:
            if handler.command != "GET":
                return self._send_json(handler, {"error": "method not allowed"}, status=405)
            frame = self.pipeline_manager.latest_raw_frame(frame_raw_match.group(1))
            if frame is None:
                return self._send_json(handler, {"error": "frame not available"}, status=404)
            return self._send_bytes(handler, frame, content_type="image/jpeg")

        mjpeg_match = re.fullmatch(r"/v1/pipelines/([^/]+)/stream\.mjpg", path)
        if mjpeg_match:
            if handler.command != "GET":
                return self._send_json(handler, {"error": "method not allowed"}, status=405)
            return self._stream_mjpeg(handler, mjpeg_match.group(1), variant="processed")

        mjpeg_raw_match = re.fullmatch(r"/v1/pipelines/([^/]+)/stream\.raw\.mjpg", path)
        if mjpeg_raw_match:
            if handler.command != "GET":
                return self._send_json(handler, {"error": "method not allowed"}, status=405)
            return self._stream_mjpeg(handler, mjpeg_raw_match.group(1), variant="raw")

        if path == "/v1/recordings" and handler.command == "GET":
            retention = self.retention_manager.enforce()
            drone_id = query.get("droneId", [""])[0].strip() or None
            items = self.nvr_store.list_recordings(drone_id=drone_id)
            return self._send_json(
                handler,
                {
                    "items": items,
                    "storage": self.nvr_store.storage_summary(),
                    "retention": retention,
                },
            )

        if path == "/v1/recordings" and handler.command == "POST":
            payload = self._read_json(handler)
            entry = self.nvr_store.add_recording(payload)
            retention = self.retention_manager.enforce()
            return self._send_json(handler, {"ok": True, "recording": entry, "retention": retention}, status=201)

        recording_match = re.fullmatch(r"/v1/recordings/([^/]+)", path)
        if recording_match:
            recording_id = recording_match.group(1)
            if handler.command == "DELETE":
                deleted = self.nvr_store.delete_recording(recording_id, reason="manual")
                if deleted is None:
                    return self._send_json(handler, {"error": "recording not found"}, status=404)
                return self._send_json(
                    handler,
                    {
                        "ok": True,
                        "recording": deleted,
                        "storage": self.nvr_store.storage_summary(),
                    },
                )
            return self._send_json(handler, {"error": "method not allowed"}, status=405)

        recording_media_match = re.fullmatch(r"/v1/recordings/([^/]+)/media", path)
        if recording_media_match:
            if handler.command != "GET":
                return self._send_json(handler, {"error": "method not allowed"}, status=405)
            recording_id = recording_media_match.group(1)
            entry = self.nvr_store.get_recording(recording_id)
            if not entry:
                return self._send_json(handler, {"error": "recording not found"}, status=404)
            file_path = Path(str(entry.get("filePath") or "")).expanduser().resolve()
            if not file_path.exists() or not file_path.is_file():
                return self._send_json(handler, {"error": f"recording file not found: {file_path}"}, status=404)
            media_path = self._browser_media_path(file_path)
            return self._send_file(handler, media_path, range_header=handler.headers.get("Range", ""))

        self._send_json(handler, {"error": "not found"}, status=404)

    def _read_json(self, handler: BaseHTTPRequestHandler) -> dict[str, object]:
        length = int(handler.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = handler.rfile.read(length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _send_json(self, handler: BaseHTTPRequestHandler, payload: dict[str, object], *, status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        return self._send_bytes(handler, data, content_type="application/json; charset=utf-8", status=status)

    def _send_bytes(
        self,
        handler: BaseHTTPRequestHandler,
        payload: bytes,
        *,
        content_type: str,
        status: int = 200,
    ) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _send_file(self, handler: BaseHTTPRequestHandler, file_path: Path, *, range_header: str = "") -> None:
        stat = file_path.stat()
        content_type = self._content_type(file_path)
        if not range_header:
            handler.send_response(200)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("Accept-Ranges", "bytes")
            handler.send_header("Content-Length", str(stat.st_size))
            handler.end_headers()
            with file_path.open("rb") as stream:
                while True:
                    chunk = stream.read(64 * 1024)
                    if not chunk:
                        break
                    handler.wfile.write(chunk)
            return

        match = re.fullmatch(r"bytes=(\d*)-(\d*)", str(range_header or "").strip())
        if not match:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{stat.st_size}")
            handler.end_headers()
            return

        start = int(match.group(1)) if match.group(1) else 0
        end = int(match.group(2)) if match.group(2) else stat.st_size - 1
        start = max(0, start)
        end = min(stat.st_size - 1, end)
        if start > end or start >= stat.st_size:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{stat.st_size}")
            handler.end_headers()
            return

        handler.send_response(206)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Length", str(end - start + 1))
        handler.send_header("Content-Range", f"bytes {start}-{end}/{stat.st_size}")
        handler.end_headers()
        with file_path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)

    @staticmethod
    def _content_type(file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".mp4":
            return "video/mp4"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        return "application/octet-stream"

    def _browser_media_path(self, file_path: Path) -> Path:
        if file_path.suffix.lower() != ".mp4" or file_path.name.endswith(".browser.mp4"):
            return file_path

        sidecar_path = file_path.with_name(f"{file_path.stem}.browser.mp4")
        if self._is_fresh_media(sidecar_path, file_path):
            return sidecar_path

        if self._is_browser_playable_mp4(file_path):
            return file_path

        if self._transcode_browser_mp4(file_path, sidecar_path):
            return sidecar_path

        return file_path

    @staticmethod
    def _is_fresh_media(candidate: Path, source: Path) -> bool:
        try:
            return (
                candidate.exists()
                and candidate.is_file()
                and candidate.stat().st_size > 0
                and candidate.stat().st_mtime >= source.stat().st_mtime
            )
        except OSError:
            return False

    def _is_browser_playable_mp4(self, file_path: Path) -> bool:
        try:
            result = subprocess.run(
                [
                    self.settings.ffmpeg_path,
                    "-hide_banner",
                    "-i",
                    str(file_path),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )
        except Exception:
            return False

        details = result.stderr.lower()
        return "video: h264" in details or "video: avc1" in details

    def _transcode_browser_mp4(self, source_path: Path, output_path: Path) -> bool:
        tmp_path = output_path.with_name(f"{output_path.stem}.tmp.mp4")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            subprocess.run(
                [
                    self.settings.ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    "-profile:v",
                    "baseline",
                    "-movflags",
                    "+faststart",
                    str(tmp_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
            )
            tmp_path.replace(output_path)
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return False

    def _latest_frame_for_variant(self, drone_id: str, *, variant: str) -> bytes | None:
        if variant == "raw":
            return self.pipeline_manager.latest_raw_frame(drone_id)
        return self.pipeline_manager.latest_processed_frame(drone_id)

    def _stream_mjpeg(
        self,
        handler: BaseHTTPRequestHandler,
        drone_id: str,
        *,
        variant: str = "processed",
    ) -> None:
        boundary = "frame"
        pipeline = self.pipeline_manager.get_pipeline(drone_id)
        if pipeline is None:
            return self._send_json(handler, {"error": "pipeline not found"}, status=404)

        frame_interval = self._frame_interval_for_pipeline(pipeline)

        handler.send_response(200)
        handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler.end_headers()

        try:
            while True:
                frame = self._latest_frame_for_variant(drone_id, variant=variant)
                if frame:
                    header = (
                        f"--{boundary}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame)}\r\n\r\n"
                    ).encode("utf-8")
                    handler.wfile.write(header)
                    handler.wfile.write(frame)
                    handler.wfile.write(b"\r\n")
                    handler.wfile.flush()
                time.sleep(frame_interval)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _frame_interval_for_pipeline(self, pipeline: dict[str, object]) -> float:
        runtime = pipeline.get("runtime") if isinstance(pipeline.get("runtime"), dict) else {}
        candidates = [
            runtime.get("processingFps") if isinstance(runtime, dict) else None,
            pipeline.get("processingFps"),
            self.settings.processing_fps,
        ]

        target_fps = 1.0
        for candidate in candidates:
            try:
                parsed = float(candidate)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                target_fps = min(max(parsed, 1.0), 30.0)
                break

        return 1.0 / target_fps


def build_handler(app: AnalyticsServiceApp):
    class AnalyticsHandler(BaseHTTPRequestHandler):
        server_version = "PyrOneAnalytics/0.1"

        def do_GET(self) -> None:  # noqa: N802
            app.dispatch(self)

        def do_POST(self) -> None:  # noqa: N802
            app.dispatch(self)

        def do_PUT(self) -> None:  # noqa: N802
            app.dispatch(self)

        def do_DELETE(self) -> None:  # noqa: N802
            app.dispatch(self)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[analytics-api] {self.address_string()} - {format % args}")

    return AnalyticsHandler
