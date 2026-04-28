from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from pathlib import Path
from typing import Any

from service.drone_worker import DroneWorker
from service.model_registry import ModelRegistry
from service.nvr_store import NvrStore
from service.retention import RetentionManager
from service.settings import load_settings


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(path)


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
        self.stop_event = threading.Event()

    def run(self) -> int:
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        settings = load_settings()
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
                time.sleep(0.35)
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

        processed_frame = worker.latest_processed_frame()
        raw_frame = worker.latest_raw_frame()

        if processed_frame:
            _write_bytes_atomic(self.processed_frame_path, processed_frame)
        elif not runtime.get("latestProcessedFrameAvailable"):
            _remove_if_exists(self.processed_frame_path)

        if raw_frame:
            _write_bytes_atomic(self.raw_frame_path, raw_frame)
        elif not runtime.get("latestRawFrameAvailable"):
            _remove_if_exists(self.raw_frame_path)

        self._write_runtime(runtime)

    def _write_runtime(self, runtime: dict[str, Any]) -> None:
        runtime.update(
            {
                "workerProcessDroneId": self.drone_id,
                "workerProcessUpdatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _write_json_atomic(self.runtime_path, runtime)


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
