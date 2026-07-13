from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from service.worker_main import SingleDroneWorkerProcess


class _FakeWorker:
    def __init__(self) -> None:
        self.processed_sequence = 1
        self.raw_sequence = 1
        self.processed_frame = b"processed-frame"
        self.raw_frame = b"raw-frame"
        self.status = "running"

    def runtime_snapshot(self) -> dict[str, object]:
        return {
            "status": self.status,
            "sourceOpened": True,
            "latestProcessedFrameAvailable": self.processed_frame is not None,
            "latestRawFrameAvailable": self.raw_frame is not None,
        }

    def latest_processed_frame_snapshot(self) -> tuple[int, bytes | None]:
        return self.processed_sequence, self.processed_frame

    def latest_raw_frame_snapshot(self) -> tuple[int, bytes | None]:
        return self.raw_sequence, self.raw_frame

    @staticmethod
    def latest_detections() -> dict[str, object]:
        return {"droneId": "drone-1", "items": []}


class WorkerMainPublicationTests(unittest.TestCase):
    def test_same_frame_sequence_is_not_rewritten_or_rebuffered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = SingleDroneWorkerProcess(
                drone_id="drone-1",
                worker_dir=Path(temporary),
            )
            runner._buffer_slot_count = 4
            runner._reset_frame_buffer()
            worker = _FakeWorker()

            runner._publish_worker_state(worker)  # type: ignore[arg-type]
            first_mtime = runner.processed_frame_path.stat().st_mtime_ns
            first_sequence = runner._buffer_sequence["processed"]
            first_manifest = json.loads(
                (runner.buffer_dir / "processed.json").read_text(encoding="utf-8")
            )

            runner._publish_worker_state(worker)  # type: ignore[arg-type]

            self.assertEqual(runner.processed_frame_path.stat().st_mtime_ns, first_mtime)
            self.assertEqual(runner._buffer_sequence["processed"], first_sequence)
            self.assertEqual(
                json.loads((runner.buffer_dir / "processed.json").read_text(encoding="utf-8")),
                first_manifest,
            )

    def test_disabled_legacy_mode_writes_only_runtime_and_detections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = SingleDroneWorkerProcess(
                drone_id="drone-1",
                worker_dir=Path(temporary),
            )
            runner._legacy_mjpeg_enabled = False
            runner.processed_frame_path.write_bytes(b"stale")
            runner.raw_frame_path.write_bytes(b"stale")
            runner._remove_legacy_frame_artifacts()

            runner._publish_worker_state(_FakeWorker())  # type: ignore[arg-type]

            self.assertFalse(runner.processed_frame_path.exists())
            self.assertFalse(runner.raw_frame_path.exists())
            self.assertFalse(runner.buffer_dir.exists())
            self.assertTrue(runner.runtime_path.exists())
            self.assertTrue(runner.detections_path.exists())

            runtime_mtime = runner.runtime_path.stat().st_mtime_ns
            detections_mtime = runner.detections_path.stat().st_mtime_ns
            worker = _FakeWorker()
            runner._publish_worker_state(worker)  # type: ignore[arg-type]
            self.assertEqual(runner.runtime_path.stat().st_mtime_ns, runtime_mtime)
            self.assertEqual(runner.detections_path.stat().st_mtime_ns, detections_mtime)

            worker.status = "waiting_source"
            runner._publish_worker_state(worker)  # type: ignore[arg-type]
            updated_runtime = json.loads(runner.runtime_path.read_text(encoding="utf-8"))
            self.assertEqual(updated_runtime["status"], "waiting_source")


if __name__ == "__main__":
    unittest.main()
