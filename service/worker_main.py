from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from service.drone_worker import DroneWorker
from service.model_registry import ModelRegistry
from service.nvr_store import NvrStore
from service.retention import RetentionManager
from service.schemas import utc_now
from service.settings import load_settings


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.tmp",
    )
    try:
        tmp_path.write_bytes(payload)
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes_atomic(path, f"{json.dumps(payload, indent=2)}\n".encode("utf-8"))


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


class SingleDroneWorkerProcess:
    def __init__(self, *, drone_id: str, worker_dir: Path) -> None:
        self.drone_id = drone_id
        self.worker_dir = worker_dir
        self.pipeline_path = worker_dir / "pipeline.json"
        self.runtime_path = worker_dir / "runtime.json"
        self.processed_frame_path = worker_dir / "processed.jpg"
        self.raw_frame_path = worker_dir / "raw.jpg"
        self.detections_path = worker_dir / "detections.latest.json"
        self.buffer_dir = worker_dir / "buffer"
        self.stop_event = threading.Event()
        self._buffer_entries: dict[str, list[dict[str, object]]] = {
            "processed": [],
            "raw": [],
        }
        self._buffer_sequence: dict[str, int] = {
            "processed": 0,
            "raw": 0,
        }
        self._published_frame_sequence: dict[str, int] = {
            "processed": -1,
            "raw": -1,
        }
        self._buffer_slot_count = 1
        self._legacy_mjpeg_enabled = True
        self._last_state_publish_at = 0.0
        self._last_state_priority_signature: tuple[object, ...] | None = None
        self._state_publish_interval_seconds = 0.25

    def run(self) -> int:
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        settings = load_settings()
        self._legacy_mjpeg_enabled = settings.legacy_mjpeg_enabled
        self._buffer_slot_count = max(1, int(max(settings.processing_fps, 1.0) * max(settings.mjpeg_buffer_seconds, 1.0)) + 30)
        if self._legacy_mjpeg_enabled:
            self._reset_frame_buffer()
        else:
            self._remove_legacy_frame_artifacts()
        model_registry = ModelRegistry(settings)
        store = NvrStore(settings)
        retention_manager = RetentionManager(settings, store)

        pipeline = self._wait_for_pipeline()
        if not pipeline:
            self._write_runtime({"status": "stopped", "message": "pipeline file not available"})
            return 2

        worker = DroneWorker(
            settings=settings,
            model_registry=model_registry,
            store=store,
            pipeline=pipeline,
            on_recording_saved=lambda _recording: retention_manager.enforce(),
        )

        last_pipeline_mtime = self._pipeline_mtime()
        worker.start()

        try:
            while not self.stop_event.is_set():
                current_mtime = self._pipeline_mtime()
                if current_mtime and current_mtime != last_pipeline_mtime:
                    next_pipeline = _read_json(self.pipeline_path)
                    if next_pipeline:
                        worker.update_pipeline(next_pipeline)
                        last_pipeline_mtime = current_mtime

                self._publish_worker_state(worker)
                time.sleep(0.05)
        finally:
            worker.stop()
            runtime = worker.runtime_snapshot()
            runtime.update(
                {
                    "status": "stopped",
                    "message": "dedicated worker process stopped",
                    "workerProcessPid": None,
                    "workerProcessRunning": False,
                    "workerProcessMode": "process",
                }
            )
            self._write_runtime(runtime)

        return 0

    def _wait_for_pipeline(self) -> dict[str, Any]:
        deadline = time.monotonic() + 10
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            pipeline = _read_json(self.pipeline_path)
            if pipeline:
                return pipeline
            time.sleep(0.2)
        return {}

    def _pipeline_mtime(self) -> float:
        try:
            return self.pipeline_path.stat().st_mtime
        except OSError:
            return 0.0

    def _publish_worker_state(self, worker: DroneWorker) -> None:
        runtime = worker.runtime_snapshot()
        runtime.update(
            {
                "workerProcessPid": None,
                "workerProcessRunning": True,
                "workerProcessMode": "process",
            }
        )

        now = time.monotonic()
        current_event = runtime.get("currentEvent")
        current_event_id = (
            current_event.get("eventId")
            if isinstance(current_event, dict)
            else None
        )
        runtime_status = str(runtime.get("status") or "")
        priority_signature = (
            runtime_status,
            runtime.get("message") if runtime_status != "running" else "",
            runtime.get("sourceOpened"),
            runtime.get("processedStreamReady"),
            runtime.get("processedPublisherPid"),
            runtime.get("processedPublisherError"),
            runtime.get("lastSourceError"),
            current_event_id,
        )
        if (
            not self._legacy_mjpeg_enabled
            and self._last_state_publish_at
            and now - self._last_state_publish_at < self._state_publish_interval_seconds
            and priority_signature == self._last_state_priority_signature
        ):
            return
        self._last_state_publish_at = now
        self._last_state_priority_signature = priority_signature

        detections = worker.latest_detections()

        if self._legacy_mjpeg_enabled:
            processed_sequence, processed_frame = worker.latest_processed_frame_snapshot()
            raw_sequence, raw_frame = worker.latest_raw_frame_snapshot()

            if processed_sequence != self._published_frame_sequence["processed"]:
                self._published_frame_sequence["processed"] = processed_sequence
                if processed_frame:
                    _write_bytes_atomic(self.processed_frame_path, processed_frame)
                    self._append_buffer_frame("processed", processed_frame)
                elif not runtime.get("latestProcessedFrameAvailable"):
                    _remove_if_exists(self.processed_frame_path)
                    self._clear_buffer("processed")
                    detections = {
                        "droneId": self.drone_id,
                        "frameAt": "",
                        "modelId": runtime.get("modelId") or "",
                        "modelName": runtime.get("modelName") or "",
                        "sensorType": "",
                        "frameWidth": None,
                        "frameHeight": None,
                        "items": [],
                    }

            if raw_sequence != self._published_frame_sequence["raw"]:
                self._published_frame_sequence["raw"] = raw_sequence
                if raw_frame:
                    _write_bytes_atomic(self.raw_frame_path, raw_frame)
                    self._append_buffer_frame("raw", raw_frame)
                elif not runtime.get("latestRawFrameAvailable"):
                    _remove_if_exists(self.raw_frame_path)
                    self._clear_buffer("raw")

        _write_json_atomic(self.detections_path, detections)
        self._write_runtime(runtime)

    def _write_runtime(self, runtime: dict[str, Any]) -> None:
        runtime.update(
            {
                "workerProcessDroneId": self.drone_id,
                "workerProcessUpdatedAt": utc_now(),
            }
        )
        _write_json_atomic(self.runtime_path, runtime)

    def _reset_frame_buffer(self) -> None:
        self._buffer_entries = {
            "processed": [],
            "raw": [],
        }
        self._buffer_sequence = {
            "processed": 0,
            "raw": 0,
        }
        self._published_frame_sequence = {
            "processed": -1,
            "raw": -1,
        }

        for variant in ("processed", "raw"):
            variant_dir = self.buffer_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            for frame_path in variant_dir.glob("*.jpg"):
                _remove_if_exists(frame_path)
            self._write_buffer_manifest(variant)

    def _clear_buffer(self, variant: str) -> None:
        if not self._buffer_entries.get(variant):
            return
        self._buffer_entries[variant] = []
        self._write_buffer_manifest(variant)

    def _remove_legacy_frame_artifacts(self) -> None:
        _remove_if_exists(self.processed_frame_path)
        _remove_if_exists(self.raw_frame_path)
        if not self.buffer_dir.exists():
            return
        for path in sorted(self.buffer_dir.rglob("*"), reverse=True):
            if path.is_file():
                _remove_if_exists(path)
                continue
            try:
                path.rmdir()
            except OSError:
                pass
        try:
            self.buffer_dir.rmdir()
        except OSError:
            pass

    def _append_buffer_frame(self, variant: str, frame: bytes) -> None:
        if variant not in self._buffer_entries:
            return

        self._buffer_sequence[variant] += 1
        sequence = self._buffer_sequence[variant]
        slot = sequence % self._buffer_slot_count
        variant_dir = self.buffer_dir / variant
        frame_path = variant_dir / f"{slot:04d}.jpg"
        _write_bytes_atomic(frame_path, frame)

        entries = [
            item
            for item in self._buffer_entries[variant]
            if int(item.get("slot") or -1) != slot
        ]
        entries.append(
            {
                "seq": sequence,
                "slot": slot,
                "createdAt": time.time(),
                "file": f"{variant}/{frame_path.name}",
                "size": len(frame),
            },
        )
        self._buffer_entries[variant] = entries[-self._buffer_slot_count :]
        self._write_buffer_manifest(variant)

    def _write_buffer_manifest(self, variant: str) -> None:
        manifest_path = self.buffer_dir / f"{variant}.json"
        _write_json_atomic(
            manifest_path,
            {
                "variant": variant,
                "droneId": self.drone_id,
                "items": self._buffer_entries.get(variant, []),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one PyrOne IA drone worker process.")
    parser.add_argument("--drone-id", required=True)
    parser.add_argument("--worker-dir", required=True)
    args = parser.parse_args()

    runner = SingleDroneWorkerProcess(
        drone_id=str(args.drone_id),
        worker_dir=Path(str(args.worker_dir)).expanduser().resolve(),
    )

    def stop(_signum: int, _frame: object) -> None:
        runner.stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    raise SystemExit(runner.run())


if __name__ == "__main__":
    main()
