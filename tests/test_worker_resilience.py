from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from service.drone_worker import DroneWorker
from service.runtime_log import safe_console_log
from service.worker_process import DroneWorkerProcess


class _MutableRegistry:
    def __init__(self) -> None:
        self.model = None
        self.revision = 1
        self.last_reload_error = ""

    @staticmethod
    def canonical_id(model_id: str) -> str:
        if model_id == "pyronear-yolov8s-wide":
            return "pyrone-yolov8s-wide"
        return model_id

    def get(self, _model_id: str):
        return self.model


class _BlockingCapture:
    def __init__(self) -> None:
        self.released = threading.Event()

    def read(self):
        self.released.wait(2.0)
        return False, None

    def release(self) -> None:
        self.released.set()


class DroneWorkerResilienceTests(unittest.TestCase):
    def test_closed_stdout_cannot_crash_failure_logging(self) -> None:
        with patch("builtins.print", side_effect=ValueError("I/O on closed file")):
            safe_console_log("recoverable failure")

    def test_missing_model_retries_with_backoff_then_loads_legacy_alias(self) -> None:
        registry = _MutableRegistry()
        worker = object.__new__(DroneWorker)
        worker.model_registry = registry
        worker.settings = SimpleNamespace(
            model_retry_seconds=1.0,
            model_retry_max_seconds=4.0,
        )
        worker._lock = threading.Lock()
        worker._runtime = {}
        worker._model = None
        worker._model_id = ""
        worker._model_name = ""
        worker._model_loaded_signature = None
        worker._model_retry_delay = 1.0
        worker._model_retry_next_at = 0.0
        worker._model_retry_count = 0
        worker._model_registry_revision = 0
        pipeline = {"currentModelId": "pyronear-yolov8s-wide"}

        with patch("service.drone_worker.time.monotonic", return_value=10.0):
            self.assertIsNone(worker._ensure_model(pipeline))
        self.assertEqual(worker._runtime["modelRetryCount"], 1)
        self.assertEqual(worker._runtime["modelRetryDelaySeconds"], 1.0)

        with patch("service.drone_worker.time.monotonic", return_value=10.5):
            self.assertIsNone(worker._ensure_model(pipeline))
        self.assertEqual(worker._runtime["modelRetryCount"], 1)

        registry.model = {
            "id": "pyrone-yolov8s-wide",
            "name": "Pyrone",
            "weightsPath": "/tmp/pyrone-model.pt",
            "weightsPresent": True,
        }
        loaded = object()
        with (
            patch("service.drone_worker.time.monotonic", return_value=11.0),
            patch("service.drone_worker.YOLO", return_value=loaded) as yolo,
        ):
            model_info = worker._ensure_model(pipeline)

        self.assertEqual(model_info["id"], "pyrone-yolov8s-wide")
        self.assertIs(worker._model, loaded)
        self.assertEqual(worker._model_id, "pyrone-yolov8s-wide")
        self.assertEqual(worker._runtime["modelRetryDelaySeconds"], 0.0)
        yolo.assert_called_once_with("/tmp/pyrone-model.pt")

    def test_source_retry_backoff_is_bounded_and_reset_after_stability(self) -> None:
        worker = object.__new__(DroneWorker)
        worker.settings = SimpleNamespace(
            reconnect_delay_seconds=1.0,
            reconnect_delay_max_seconds=4.0,
        )
        worker._lock = threading.Lock()
        worker._runtime = {}
        worker._stop_event = threading.Event()
        worker._wake_event = threading.Event()
        worker._source_retry_delay = 1.0
        worker._source_retry_count = 0
        waits: list[float] = []
        worker._wait_interruptibly = MethodType(
            lambda self, delay: waits.append(delay) or False,
            worker,
        )

        worker._wait_for_source_retry("offline")
        worker._wait_for_source_retry("offline")
        worker._wait_for_source_retry("offline")
        worker._wait_for_source_retry("offline")

        self.assertEqual(waits, [1.0, 2.0, 4.0, 4.0])
        self.assertEqual(worker._runtime["sourceRetryCount"], 4)
        worker._reset_source_retry_backoff()
        self.assertEqual(worker._source_retry_delay, 1.0)
        self.assertTrue(worker._runtime["sourceStable"])

    def test_capture_stall_is_detected_and_session_returns_for_reconnect(self) -> None:
        worker = object.__new__(DroneWorker)
        worker.settings = SimpleNamespace(
            processing_fps=20.0,
            rtsp_read_timeout_seconds=0.05,
            capture_stall_timeout_seconds=0.05,
            source_stable_reset_seconds=1.0,
        )
        worker._lock = threading.Lock()
        worker._runtime = {
            "sourceStallCount": 0,
            "framesCaptured": 0,
            "framesDroppedBeforeInference": 0,
            "captureQueueDepth": 0,
        }
        worker._pipeline = {"droneId": "drone-1", "analyticsEnabled": True}
        worker._stop_event = threading.Event()
        worker._stop_processed_publisher = MethodType(lambda self: None, worker)
        worker._clear_preview = MethodType(lambda self: None, worker)
        capture = _BlockingCapture()

        worker._run_capture_session(
            capture,
            source_fps=0.0,
            initial_frame_interval=0.05,
        )

        self.assertTrue(capture.released.is_set())
        self.assertEqual(worker._runtime["status"], "waiting_source")
        self.assertEqual(worker._runtime["sourceStallCount"], 1)
        self.assertIn("stalled", worker._runtime["lastSourceError"])


class WorkerProcessRespawnTests(unittest.TestCase):
    def _worker(self) -> DroneWorkerProcess:
        worker = object.__new__(DroneWorkerProcess)
        worker.settings = SimpleNamespace(
            reconnect_delay_seconds=1.0,
            reconnect_delay_max_seconds=4.0,
            source_stable_reset_seconds=5.0,
        )
        worker._started_monotonic = 0.0
        worker._restart_retry_delay = 1.0
        worker._restart_next_at = 0.0
        worker._restart_count = 0
        worker._last_restart_error = ""
        worker.is_running = Mock(return_value=False)
        worker.start = Mock(return_value=True)
        worker._write_runtime_overlay = Mock()
        return worker

    def test_crashed_process_respawns_forever_with_bounded_backoff(self) -> None:
        worker = self._worker()

        self.assertTrue(worker.ensure_running(now=10.0))
        self.assertEqual(worker._restart_count, 1)
        worker.start.assert_called_once()

        self.assertFalse(worker.ensure_running(now=10.5))
        worker.start.assert_called_once()

        self.assertTrue(worker.ensure_running(now=11.0))
        self.assertEqual(worker._restart_count, 2)
        self.assertEqual(worker._restart_retry_delay, 4.0)

        self.assertTrue(worker.ensure_running(now=13.0))
        self.assertEqual(worker._restart_count, 3)
        self.assertEqual(worker._restart_retry_delay, 4.0)

    def test_stable_process_resets_respawn_backoff(self) -> None:
        worker = self._worker()
        worker.is_running.return_value = True
        worker._started_monotonic = 10.0
        worker._restart_retry_delay = 4.0
        worker._restart_next_at = 20.0
        worker._last_restart_error = "previous failure"

        self.assertTrue(worker.ensure_running(now=16.0))
        self.assertEqual(worker._restart_retry_delay, 1.0)
        self.assertEqual(worker._restart_next_at, 0.0)
        self.assertEqual(worker._last_restart_error, "")


if __name__ == "__main__":
    unittest.main()
