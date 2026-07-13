from __future__ import annotations

from collections import deque
from queue import Empty
import subprocess
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from service.latest_frame_queue import CapturedFrame, LatestFrameQueue


PUBLISHER_PROCESS_MARKER = "comment=pyrone-processed-publisher"
MIN_OPERATIONAL_WRITE_TIMEOUT_SECONDS = 8.0


_ALLOWED_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
}


def normalize_publisher_preset(value: str, fallback: str = "veryfast") -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _ALLOWED_PRESETS else fallback


def normalize_publisher_bitrate(value: str, fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    suffix = normalized[-1:] if normalized else ""
    number = normalized[:-1] if suffix in {"k", "m"} else normalized

    try:
        parsed = float(number)
    except (TypeError, ValueError):
        return fallback

    if parsed <= 0 or parsed > 100_000:
        return fallback

    if parsed.is_integer():
        number = str(int(parsed))
    else:
        number = f"{parsed:.3f}".rstrip("0").rstrip(".")
    return f"{number}{suffix}"


def validate_publish_url(output_url: str, transport: str) -> str:
    normalized = str(output_url or "").strip()
    try:
        parsed = urlsplit(normalized)
    except ValueError as error:
        raise ValueError(f"invalid {transport} publisher URL") from error
    expected_schemes = {"rtmp", "rtmps"} if transport == "rtmp" else {"rtsp", "rtsps"}

    if parsed.scheme.lower() not in expected_schemes:
        raise ValueError(f"invalid {transport} publisher URL scheme")
    if not parsed.hostname:
        raise ValueError(f"invalid {transport} publisher URL host")
    if not parsed.path or parsed.path == "/":
        raise ValueError(f"invalid {transport} publisher URL path")
    if parsed.fragment:
        raise ValueError("publisher URL must not include a fragment")

    return normalized


def sanitize_stream_url(value: str) -> str:
    normalized = str(value or "").strip()
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        scheme, separator, rest = normalized.partition("://")
        if separator and "@" in rest:
            return f"{scheme}://******@{rest.rsplit('@', 1)[-1]}"
        return normalized
    if not parsed.scheme or not parsed.netloc or parsed.username is None:
        return normalized

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        # The value will subsequently fail publisher validation; do not risk
        # exposing userinfo while constructing the diagnostic URL.
        host = parsed.netloc.rsplit("@", 1)[-1]
        port = None
    if port:
        host = f"{host}:{port}"
    netloc = (
        f"{parsed.username or ''}:******@{host}"
        if parsed.password is not None
        else f"******@{host}"
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))


def build_ffmpeg_publisher_command(
    *,
    ffmpeg_path: str,
    output_url: str,
    transport: str,
    width: int,
    height: int,
    fps: float,
    bitrate: str,
    bufsize: str,
    preset: str,
) -> list[str]:
    normalized_transport = "rtmp" if transport == "rtmp" else "rtsp"
    validated_url = validate_publish_url(output_url, normalized_transport)
    normalized_fps = min(max(float(fps or 1.0), 1.0), 30.0)
    gop = max(2, int(round(normalized_fps)))
    output_args = (
        ["-f", "flv", "-flvflags", "no_duration_filesize", validated_url]
        if normalized_transport == "rtmp"
        else ["-f", "rtsp", "-rtsp_transport", "tcp", "-muxdelay", "0.1", validated_url]
    )

    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{int(width)}x{int(height)}",
        "-framerate",
        f"{normalized_fps:.3f}",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        normalize_publisher_preset(preset),
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-bf",
        "0",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-x264-params",
        f"keyint={gop}:min-keyint={gop}:scenecut=0:repeat-headers=1",
        "-b:v",
        normalize_publisher_bitrate(bitrate, "2500k"),
        "-maxrate",
        normalize_publisher_bitrate(bitrate, "2500k"),
        "-bufsize",
        normalize_publisher_bitrate(bufsize, "5000k"),
        "-fps_mode",
        "cfr",
        "-metadata",
        PUBLISHER_PROCESS_MARKER,
        *output_args,
    ]


class FfmpegFramePublisher:
    """Paced H.264 publisher with a single latest-frame input slot."""

    def __init__(
        self,
        *,
        ffmpeg_path: str,
        output_url: str,
        transport: str,
        width: int,
        height: int,
        fps: float,
        bitrate: str,
        bufsize: str,
        preset: str,
        write_timeout_seconds: float,
        startup_timeout_seconds: float,
        ready_after_seconds: float,
        ready_freshness_seconds: float,
        stable_reset_seconds: float,
        stale_frame_seconds: float,
    ) -> None:
        self.output_url = validate_publish_url(output_url, transport)
        self.safe_output_url = sanitize_stream_url(self.output_url)
        self.transport = "rtmp" if transport == "rtmp" else "rtsp"
        self.width = int(width)
        self.height = int(height)
        self.fps = min(max(float(fps or 1.0), 1.0), 30.0)
        self.write_timeout_seconds = max(
            MIN_OPERATIONAL_WRITE_TIMEOUT_SECONDS,
            float(write_timeout_seconds or MIN_OPERATIONAL_WRITE_TIMEOUT_SECONDS),
        )
        self.startup_timeout_seconds = min(
            max(float(startup_timeout_seconds or 25.0), 15.0),
            30.0,
        )
        self.ready_after_seconds = max(0.5, float(ready_after_seconds or 2.0))
        self.ready_freshness_seconds = max(
            0.5,
            float(ready_freshness_seconds or 2.0),
            3.0 / self.fps,
        )
        self.stable_reset_seconds = max(
            self.ready_after_seconds,
            float(stable_reset_seconds or 20.0),
        )
        self.stale_frame_seconds = max(
            self.write_timeout_seconds,
            float(stale_frame_seconds or 5.0),
        )
        self._expected_payload_size = self.width * self.height * 3
        self._queue: LatestFrameQueue[bytes] = LatestFrameQueue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._failed = False
        self._last_error = ""
        self._process_started_at = time.monotonic()
        self._first_input_at = 0.0
        self._first_write_at = 0.0
        self._write_started_at = 0.0
        self._last_input_at = 0.0
        self._last_write_at = 0.0
        self._frames_received = 0
        self._frames_written = 0
        self._frames_dropped = 0
        self._stderr_tail: deque[str] = deque(maxlen=8)
        self._command = build_ffmpeg_publisher_command(
            ffmpeg_path=ffmpeg_path,
            output_url=self.output_url,
            transport=self.transport,
            width=self.width,
            height=self.height,
            fps=self.fps,
            bitrate=bitrate,
            bufsize=bufsize,
            preset=preset,
        )
        self._process: subprocess.Popen[bytes] | None = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        self._writer_thread = threading.Thread(
            target=self._run,
            name=f"pyrone-{self.transport}-publisher-{self._process.pid}",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name=f"pyrone-{self.transport}-publisher-stderr-{self._process.pid}",
            daemon=True,
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog,
            name=f"pyrone-{self.transport}-publisher-watchdog-{self._process.pid}",
            daemon=True,
        )
        self._writer_thread.start()
        self._stderr_thread.start()
        self._watchdog_thread.start()

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process and process.poll() is None else None

    @property
    def frames_received(self) -> int:
        with self._lock:
            return self._frames_received

    @property
    def frames_written(self) -> int:
        with self._lock:
            return self._frames_written

    @property
    def frames_dropped(self) -> int:
        with self._lock:
            return self._frames_dropped

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    @property
    def ready(self) -> bool:
        if not self.is_running():
            return False
        now = time.monotonic()
        with self._lock:
            first_write_at = self._first_write_at
            last_write_at = self._last_write_at
        if not first_write_at or not last_write_at:
            return False
        return (
            now - first_write_at >= self.ready_after_seconds
            and now - last_write_at <= self.ready_freshness_seconds
        )

    @property
    def stable(self) -> bool:
        if not self.ready:
            return False
        with self._lock:
            first_write_at = self._first_write_at
        return bool(
            first_write_at
            and time.monotonic() - first_write_at >= self.stable_reset_seconds
        )

    @property
    def uptime_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._process_started_at)

    def is_running(self) -> bool:
        process = self._process
        if not process or not process.stdin or process.poll() is not None:
            return False

        with self._lock:
            if self._failed:
                return False
            write_started_at = self._write_started_at
            first_input_at = self._first_input_at
            first_write_at = self._first_write_at

        now = time.monotonic()
        startup_started_at = write_started_at or first_input_at
        startup_timed_out = bool(
            not first_write_at
            and startup_started_at
            and now - startup_started_at > self.startup_timeout_seconds
        )
        write_timed_out = bool(
            first_write_at
            and write_started_at
            and now - write_started_at > self.write_timeout_seconds
        )
        if startup_timed_out or write_timed_out:
            self._mark_failed(
                "processed publisher startup/handshake timed out"
                if startup_timed_out
                else "processed publisher operational write timed out"
            )
            try:
                process.terminate()
            except (OSError, ProcessLookupError):
                pass
            return False
        return True

    def write(self, payload: bytes) -> bool:
        if not self.is_running() or len(payload) != self._expected_payload_size:
            return False

        now = time.monotonic()
        with self._lock:
            sequence = self._frames_received + 1
            self._frames_received = sequence
            if not self._first_input_at:
                self._first_input_at = now
            self._last_input_at = now

        dropped = self._queue.put_latest(
            CapturedFrame(
                sequence=sequence,
                value=payload,
                captured_monotonic=now,
                captured_at="",
            ),
        )
        if dropped:
            with self._lock:
                self._frames_dropped += dropped
        return True

    def _run(self) -> None:
        latest: CapturedFrame[bytes] | None = None
        last_written_sequence = 0
        next_write_at = time.monotonic()
        frame_interval = 1.0 / self.fps

        while not self._stop_event.is_set():
            now = time.monotonic()
            wait_for = 0.25 if latest is None else min(max(next_write_at - now, 0.0), 0.25)
            try:
                candidate = self._queue.get(timeout=wait_for)
                candidate, drained = self._queue.drain_to_latest(candidate)
                locally_superseded = 1 if latest and latest.sequence != last_written_sequence else 0
                latest = candidate
                if drained or locally_superseded:
                    with self._lock:
                        self._frames_dropped += drained + locally_superseded
            except Empty:
                pass

            if latest is None:
                continue

            now = time.monotonic()
            if now < next_write_at:
                continue

            # A temporary inference pause must not tear down MediaMTX readers.
            # Keep the last complete frame flowing at CFR; DroneWorker owns source
            # liveness and explicitly releases this publisher when ingest closes.

            if not self._write_payload(latest.value):
                return

            last_written_sequence = latest.sequence
            next_write_at = time.monotonic() + frame_interval

    def _write_payload(self, payload: bytes) -> bool:
        process = self._process
        if not process or not process.stdin or process.poll() is not None:
            if not self._stop_event.is_set():
                self._mark_failed("processed publisher process stopped")
            return False

        try:
            with self._lock:
                self._write_started_at = time.monotonic()
            view = memoryview(payload)
            written = 0
            while written < len(view):
                chunk_size = process.stdin.write(view[written:])
                if not chunk_size:
                    raise BrokenPipeError("processed publisher stdin closed")
                written += chunk_size
            with self._lock:
                self._frames_written += 1
                write_completed_at = time.monotonic()
                if not self._first_write_at:
                    self._first_write_at = write_completed_at
                self._last_write_at = write_completed_at
            return True
        except (BrokenPipeError, OSError, ValueError) as error:
            if not self._stop_event.is_set():
                self._mark_failed(str(error))
            return False
        finally:
            with self._lock:
                self._write_started_at = 0.0

    def _read_stderr(self) -> None:
        process = self._process
        if not process or not process.stderr:
            return
        try:
            for raw_line in iter(process.stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    self._stderr_tail.append(self._safe_error(line))
        except (OSError, ValueError):
            return

    def _watchdog(self) -> None:
        while not self._stop_event.wait(0.1):
            if not self.is_running():
                return

    def _safe_error(self, value: str) -> str:
        return str(value or "").replace(self.output_url, self.safe_output_url)[-1000:]

    def _mark_failed(self, error: str) -> None:
        detail = self._safe_error(error)
        if self._stderr_tail:
            detail = f"{detail}: {self._stderr_tail[-1]}" if detail else self._stderr_tail[-1]
        with self._lock:
            if self._failed and self._last_error:
                return
            self._failed = True
            self._last_error = detail
        print(
            f"[processed-publisher] output={self.safe_output_url} failed: {detail}",
            flush=True,
        )

    def release(self) -> None:
        process = self._process
        self._process = None
        self._stop_event.set()
        self._queue.clear()

        if process:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            except (OSError, ProcessLookupError):
                pass
            finally:
                for stream in (process.stdin, process.stderr):
                    try:
                        if stream:
                            stream.close()
                    except OSError:
                        pass

        current = threading.current_thread()
        for thread in (self._writer_thread, self._stderr_thread, self._watchdog_thread):
            if thread is not current:
                thread.join(timeout=1.0)
