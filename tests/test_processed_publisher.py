from __future__ import annotations

import os
from pathlib import Path
from collections import deque
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from service.processed_publisher import (
    PUBLISHER_PROCESS_MARKER,
    build_ffmpeg_publisher_command,
    sanitize_stream_url,
    validate_publish_url,
    FfmpegFramePublisher,
)
from service.settings import load_settings


class _FakePublisherProcess:
    def __init__(self) -> None:
        self.stdin = object()
        self.terminated = False

    def poll(self):
        return 1 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True


class _BlockingStdin:
    def __init__(self, stopped: threading.Event) -> None:
        self._stopped = stopped

    def write(self, _payload) -> int:
        self._stopped.wait(2.0)
        raise BrokenPipeError("fake blocked publisher stopped")

    @staticmethod
    def close() -> None:
        return


class _EmptyStderr:
    @staticmethod
    def readline() -> bytes:
        return b""

    @staticmethod
    def close() -> None:
        return


class _BlockingPublisherProcess:
    pid = 999999

    def __init__(self) -> None:
        self.stopped = threading.Event()
        self.stdin = _BlockingStdin(self.stopped)
        self.stderr = _EmptyStderr()

    def poll(self):
        return 1 if self.stopped.is_set() else None

    def terminate(self) -> None:
        self.stopped.set()

    def kill(self) -> None:
        self.stopped.set()

    @staticmethod
    def wait(timeout=None) -> int:
        return 0


def _publisher_timing_fixture() -> tuple[FfmpegFramePublisher, _FakePublisherProcess]:
    publisher = object.__new__(FfmpegFramePublisher)
    process = _FakePublisherProcess()
    publisher._process = process
    publisher._lock = threading.Lock()
    publisher._failed = False
    publisher._last_error = ""
    publisher._stderr_tail = deque(maxlen=8)
    publisher.output_url = "rtmp://media.internal:1935/processed/drone-1"
    publisher.safe_output_url = publisher.output_url
    publisher.write_timeout_seconds = 3.0
    publisher.startup_timeout_seconds = 25.0
    publisher.ready_after_seconds = 2.0
    publisher.ready_freshness_seconds = 2.0
    publisher.stable_reset_seconds = 20.0
    publisher._process_started_at = 90.0
    publisher._first_input_at = 100.0
    publisher._first_write_at = 0.0
    publisher._write_started_at = 100.0
    publisher._last_input_at = 100.0
    publisher._last_write_at = 0.0
    publisher._frames_received = 1
    publisher._frames_written = 0
    publisher._frames_dropped = 0
    return publisher, process


class ProcessedPublisherTests(unittest.TestCase):
    def test_watchdog_terminates_an_actually_blocked_handshake_write(self) -> None:
        process = _BlockingPublisherProcess()
        with patch("service.processed_publisher.subprocess.Popen", return_value=process):
            publisher = FfmpegFramePublisher(
                ffmpeg_path="ffmpeg",
                output_url="rtmp://media.internal:1935/processed/drone-1",
                transport="rtmp",
                width=2,
                height=2,
                fps=5.0,
                bitrate="2500k",
                bufsize="5000k",
                preset="veryfast",
                write_timeout_seconds=3.0,
                startup_timeout_seconds=15.0,
                ready_after_seconds=2.0,
                ready_freshness_seconds=2.0,
                stable_reset_seconds=20.0,
                stale_frame_seconds=8.0,
            )
        publisher.startup_timeout_seconds = 0.15
        self.assertTrue(publisher.write(bytes(12)))

        deadline = time.monotonic() + 1.0
        while not process.stopped.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)

        self.assertTrue(process.stopped.is_set())
        self.assertIn("startup/handshake", publisher.last_error)
        publisher.release()

    def test_handshake_uses_startup_timeout_then_operational_write_timeout(self) -> None:
        publisher, process = _publisher_timing_fixture()

        with patch("service.processed_publisher.time.monotonic", return_value=110.0):
            self.assertTrue(publisher.is_running())
        self.assertFalse(process.terminated)

        with patch("service.processed_publisher.time.monotonic", return_value=126.0):
            self.assertFalse(publisher.is_running())
        self.assertTrue(process.terminated)
        self.assertIn("startup/handshake", publisher.last_error)

        publisher, process = _publisher_timing_fixture()
        publisher._first_write_at = 100.0
        publisher._last_write_at = 110.0
        publisher._write_started_at = 110.0
        with patch("service.processed_publisher.time.monotonic", return_value=114.0):
            self.assertFalse(publisher.is_running())
        self.assertTrue(process.terminated)
        self.assertIn("operational write", publisher.last_error)

    def test_ready_requires_stability_liveness_and_fresh_frames(self) -> None:
        publisher, _process = _publisher_timing_fixture()
        publisher._first_write_at = 100.0
        publisher._write_started_at = 0.0
        publisher._last_write_at = 101.0
        publisher._last_input_at = 101.0
        publisher._frames_written = 10

        with patch("service.processed_publisher.time.monotonic", return_value=101.0):
            self.assertFalse(publisher.ready)

        publisher._last_write_at = 102.1
        publisher._last_input_at = 102.1
        with patch("service.processed_publisher.time.monotonic", return_value=102.1):
            self.assertTrue(publisher.ready)
            self.assertFalse(publisher.stable)

        with patch("service.processed_publisher.time.monotonic", return_value=105.0):
            self.assertFalse(publisher.ready)

        publisher._last_write_at = 121.0
        publisher._last_input_at = 121.0
        with patch("service.processed_publisher.time.monotonic", return_value=121.0):
            self.assertTrue(publisher.stable)

    def test_rtmp_command_is_low_latency_h264_and_cfr(self) -> None:
        command = build_ffmpeg_publisher_command(
            ffmpeg_path="ffmpeg",
            output_url="rtmp://media.internal:1935/processed/drone-1",
            transport="rtmp",
            width=1279,
            height=719,
            fps=20,
            bitrate="2500k",
            bufsize="5000k",
            preset="veryfast",
        )

        self.assertIn("libx264", command)
        self.assertIn("zerolatency", command)
        self.assertIn("baseline", command)
        self.assertIn("cfr", command)
        self.assertIn("flv", command)
        self.assertIn("pad=ceil(iw/2)*2:ceil(ih/2)*2", command)
        self.assertIn(PUBLISHER_PROCESS_MARKER, command)

    def test_publish_url_validation_and_redaction(self) -> None:
        url = "rtmp://publisher:secret@media.internal:1935/processed/drone-1"
        self.assertEqual(validate_publish_url(url, "rtmp"), url)
        redacted = sanitize_stream_url(url)
        self.assertNotIn("secret", redacted)
        self.assertIn("publisher:******@", redacted)
        self.assertNotIn(
            "token",
            sanitize_stream_url("rtmp://token@media.internal:1935/processed/drone-1"),
        )

        with self.assertRaises(ValueError):
            validate_publish_url("http://media.internal/processed/drone-1", "rtmp")

    def test_configured_template_targets_processed_drone_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                "PYRONE_IA_RUNTIME_DIR": str(root / "runtime"),
                "PYRONE_NVR_DIR": str(root / "nvr"),
                "PYRONE_PROCESSED_PUBLISH_TRANSPORT": "rtmp",
                "PYRONE_PROCESSED_RTMP_URL_TEMPLATE": (
                    "rtmp://192.168.210.84:1935/processed/{droneId}"
                ),
            }
            with patch.dict(os.environ, environment, clear=False):
                settings = load_settings()

        self.assertEqual(
            settings.processed_publish_url("drone / 1"),
            "rtmp://192.168.210.84:1935/processed/drone%20%2F%201",
        )

    def test_unconfigured_url_derives_only_from_known_ingest_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                "PYRONE_IA_RUNTIME_DIR": str(root / "runtime"),
                "PYRONE_NVR_DIR": str(root / "nvr"),
                "PYRONE_PROCESSED_PUBLISH_TRANSPORT": "rtmp",
                "PYRONE_PROCESSED_RTMP_URL_TEMPLATE": "",
            }
            with patch.dict(os.environ, environment, clear=False):
                settings = load_settings()

        self.assertEqual(
            settings.processed_publish_url(
                "drone-1",
                "rtsp://user:secret@192.168.210.84:8554/live/drone-1",
            ),
            "rtmp://192.168.210.84:1935/processed/drone-1",
        )
        self.assertEqual(
            settings.processed_publish_url("drone-1", "rtsp://example.org/foreign/path"),
            "",
        )

    def test_legacy_mjpeg_defaults_on_and_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_environment = {
                "PYRONE_IA_RUNTIME_DIR": str(root / "runtime"),
                "PYRONE_NVR_DIR": str(root / "nvr"),
            }
            with patch.dict(os.environ, base_environment, clear=True):
                self.assertTrue(load_settings().legacy_mjpeg_enabled)
            with patch.dict(
                os.environ,
                {**base_environment, "PYRONE_LEGACY_MJPEG_ENABLED": "false"},
                clear=True,
            ):
                self.assertFalse(load_settings().legacy_mjpeg_enabled)


if __name__ == "__main__":
    unittest.main()
