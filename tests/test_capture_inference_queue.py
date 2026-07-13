from __future__ import annotations

import threading
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from service.drone_worker import DroneWorker, _FfmpegY4mReader


class _BurstCapture:
    def __init__(self, frame_count: int) -> None:
        self._frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(frame_count)]
        self.released = False

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True

    @staticmethod
    def get(_prop_id: int) -> float:
        return 30.0


class CaptureInferenceQueueTests(unittest.TestCase):
    def test_y4m_reader_rejects_timebase_as_fps(self) -> None:
        reader = object.__new__(_FfmpegY4mReader)
        reader._fps = 0.0
        reader._width = 0
        reader._height = 0
        reader._frame_size = 0
        reader._header_ready = False
        reader._startup_deadline = 1.0
        reader._read_line = Mock(return_value=b"YUV4MPEG2 W1280 H720 F90000:1 Ip A0:0 C420jpeg")

        self.assertTrue(reader._read_y4m_header())
        self.assertEqual(reader._fps, 0.0)
        self.assertEqual(reader.get(cv2.CAP_PROP_FPS), 0.0)

    def test_y4m_reader_does_not_sweep_ffmpeg_processes_by_url(self) -> None:
        process = Mock()
        process.pid = 999999
        process.stdout = object()
        process.poll.return_value = None
        with (
            patch("service.drone_worker.subprocess.Popen", return_value=process) as popen,
            patch("service.drone_worker._terminate_marked_publisher_processes") as sweep,
        ):
            _FfmpegY4mReader(
                ffmpeg_path="ffmpeg",
                source="rtsp://media.internal/live/drone-1",
            )

        sweep.assert_not_called()
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_publisher_backoff_resets_only_after_sustained_stability(self) -> None:
        worker = object.__new__(DroneWorker)
        worker.settings = SimpleNamespace(processed_publish_retry_seconds=2.0)
        worker._processed_publisher_retry_delay = 16.0
        worker._processed_publisher_next_retry_at = 123.0

        worker._reset_publisher_backoff_if_stable(False)
        self.assertEqual(worker._processed_publisher_retry_delay, 16.0)
        self.assertEqual(worker._processed_publisher_next_retry_at, 123.0)

        worker._reset_publisher_backoff_if_stable(True)
        self.assertEqual(worker._processed_publisher_retry_delay, 2.0)
        self.assertEqual(worker._processed_publisher_next_retry_at, 0.0)

    def test_burst_keeps_latest_frame_and_reports_drops(self) -> None:
        worker = object.__new__(DroneWorker)
        worker._stop_event = threading.Event()
        worker._lock = threading.Lock()
        worker._pipeline = {
            "droneId": "drone-1",
            "analyticsEnabled": True,
            "processingFps": 1000.0,
        }
        worker._runtime = {
            "framesCaptured": 0,
            "framesDroppedBeforeInference": 0,
            "captureQueueDepth": 0,
        }
        worker.settings = SimpleNamespace(
            processing_fps=1000.0,
            rtsp_read_timeout_seconds=0.1,
        )
        processed: list[tuple[int, str, float, float]] = []

        def process_frame(self, frame, **kwargs) -> None:
            processed.append(
                (
                    int(frame[0, 0, 0]),
                    str(kwargs.get("captured_at") or ""),
                    float(kwargs.get("frame_age_ms") or 0.0),
                    float(kwargs.get("source_fps") or 0.0),
                )
            )
            time.sleep(0.002)

        worker._process_frame = MethodType(process_frame, worker)
        worker._stop_processed_publisher = MethodType(lambda self: None, worker)
        worker._clear_preview = MethodType(lambda self: None, worker)
        capture = _BurstCapture(20)

        worker._run_capture_session(
            capture,
            source_fps=0.0,
            initial_frame_interval=0.001,
        )

        self.assertTrue(capture.released)
        self.assertTrue(processed)
        self.assertEqual(processed[-1][0], 19)
        self.assertTrue(processed[-1][1])
        self.assertGreaterEqual(processed[-1][2], 0.0)
        self.assertEqual(processed[-1][3], 30.0)
        self.assertEqual(worker._runtime["sourceFps"], 30.0)
        self.assertLess(len(processed), 20)
        self.assertEqual(worker._runtime["framesCaptured"], 20)
        self.assertGreater(worker._runtime["framesDroppedBeforeInference"], 0)
        self.assertEqual(worker._runtime["captureQueueDepth"], 0)


if __name__ == "__main__":
    unittest.main()
