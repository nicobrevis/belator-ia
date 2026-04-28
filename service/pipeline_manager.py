from __future__ import annotations

import json

from service.drone_worker import DroneWorker
from service.model_registry import ModelRegistry
from service.nvr_store import NvrStore
from service.retention import RetentionManager
from service.schemas import normalize_pipeline_payload, normalize_pipeline_state_payload, utc_now
from service.settings import ServiceSettings


class PipelineManager:
    def __init__(
        self,
        settings: ServiceSettings,
        model_registry: ModelRegistry,
        store: NvrStore,
        retention_manager: RetentionManager,
    ) -> None:
        self.settings = settings
        self.model_registry = model_registry
        self.store = store
        self.retention_manager = retention_manager
        self._pipelines: dict[str, dict[str, object]] = {}
        self._workers: dict[str, DroneWorker] = {}
        self._load()
        self._ensure_workers()

    def list_pipelines(self) -> list[dict[str, object]]:
        pipelines = [self._copy_pipeline(pipeline) for pipeline in self._pipelines.values()]
        return sorted(pipelines, key=lambda item: str(item["droneName"]).lower())

    def get_pipeline(self, drone_id: str) -> dict[str, object] | None:
        pipeline = self._pipelines.get(str(drone_id or "").strip())
        return self._copy_pipeline(pipeline) if pipeline else None

    def upsert_pipeline(self, drone_id: str, payload: dict[str, object] | None) -> dict[str, object]:
        previous = self._pipelines.get(str(drone_id or "").strip())
        pipeline = normalize_pipeline_payload(drone_id, payload, previous)
        sensor_type = str(previous.get("sensorType", "unknown") if previous else "unknown")
        current_model_id = self._select_model_id(pipeline, sensor_type)
        now = utc_now()
        next_pipeline = {
            **pipeline,
            "createdAt": previous.get("createdAt", now) if previous else now,
            "updatedAt": now,
            "lastTelemetryAt": previous.get("lastTelemetryAt") if previous else "",
            "cameraMode": previous.get("cameraMode") if previous else "",
            "sensorType": sensor_type,
            "currentModelId": current_model_id,
            "processedRtspUrl": self.settings.processed_stream_url(str(pipeline["droneId"])),
            "status": self._status_for_pipeline(pipeline, previous),
        }
        restart_required = self._worker_restart_required(previous, next_pipeline)
        self._pipelines[str(next_pipeline["droneId"])] = next_pipeline
        self._persist()
        self._ensure_worker(str(next_pipeline["droneId"]), restart=restart_required)
        return self._copy_pipeline(next_pipeline)

    def delete_pipeline(self, drone_id: str) -> dict[str, object] | None:
        pipeline = self._pipelines.pop(str(drone_id or "").strip(), None)
        if pipeline is None:
            return None
        self._stop_worker(str(drone_id or "").strip())
        self._persist()
        return self._copy_pipeline(pipeline)

    def update_runtime_state(self, drone_id: str, payload: dict[str, object] | None) -> dict[str, object]:
        key = str(drone_id or "").strip()
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            raise KeyError(key)
        previous = dict(pipeline)
        state = normalize_pipeline_state_payload(payload, pipeline)
        pipeline["sensorType"] = state["sensorType"]
        pipeline["cameraMode"] = state["cameraMode"]
        pipeline["lastTelemetryAt"] = state["lastTelemetryAt"]
        pipeline["currentModelId"] = self._select_model_id(pipeline, str(pipeline["sensorType"]))
        pipeline["updatedAt"] = utc_now()
        if pipeline["analyticsEnabled"]:
            pipeline["status"] = "configured"
        restart_required = self._worker_restart_required(previous, pipeline)
        self._ensure_worker(key, restart=restart_required)
        self._persist()
        return self._copy_pipeline(pipeline)

    def latest_frame(self, drone_id: str) -> bytes | None:
        return self.latest_processed_frame(drone_id)

    def latest_processed_frame(self, drone_id: str) -> bytes | None:
        worker = self._workers.get(str(drone_id or "").strip())
        return worker.latest_processed_frame() if worker else None

    def latest_raw_frame(self, drone_id: str) -> bytes | None:
        worker = self._workers.get(str(drone_id or "").strip())
        return worker.latest_raw_frame() if worker else None

    def _select_model_id(self, pipeline: dict[str, object], sensor_type: str) -> str:
        if str(pipeline.get("modelSelectionMode", "manual")) == "auto":
            auto_model_map = pipeline.get("autoModelMap") if isinstance(pipeline.get("autoModelMap"), dict) else {}
            candidate = (
                str(auto_model_map.get(sensor_type) or "").strip()
                or str(auto_model_map.get("unknown") or "").strip()
                or str(pipeline.get("manualModelId") or "").strip()
            )
            return self.model_registry.resolve(candidate, sensor_type)
        return self.model_registry.resolve(str(pipeline.get("manualModelId") or "").strip(), sensor_type)

    def _status_for_pipeline(
        self,
        pipeline: dict[str, object],
        previous: dict[str, object] | None,
    ) -> str:
        if not pipeline["analyticsEnabled"]:
            return "disabled"
        if previous and previous.get("status") == "running":
            return "running"
        return "configured"

    def _load(self) -> None:
        if not self.settings.state_path.exists():
            return
        payload = json.loads(self.settings.state_path.read_text(encoding="utf-8"))
        raw_items = payload.get("pipelines", []) if isinstance(payload, dict) else []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            drone_id = str(item.get("droneId") or "").strip()
            if not drone_id:
                continue
            pipeline = normalize_pipeline_payload(drone_id, item, item)
            pipeline["createdAt"] = str(item.get("createdAt") or utc_now())
            pipeline["updatedAt"] = str(item.get("updatedAt") or pipeline["createdAt"])
            pipeline["lastTelemetryAt"] = str(item.get("lastTelemetryAt") or "")
            pipeline["cameraMode"] = str(item.get("cameraMode") or "")
            pipeline["sensorType"] = str(item.get("sensorType") or "unknown")
            pipeline["currentModelId"] = self._select_model_id(pipeline, str(pipeline["sensorType"]))
            pipeline["processedRtspUrl"] = self.settings.processed_stream_url(drone_id)
            pipeline["status"] = self._status_for_pipeline(pipeline, item)
            self._pipelines[drone_id] = pipeline

    def _persist(self) -> None:
        payload = {
            "pipelines": [
                {
                    "droneId": pipeline["droneId"],
                    "droneName": pipeline["droneName"],
                    "rtspUrl": pipeline["rtspUrl"],
                    "analyticsEnabled": pipeline["analyticsEnabled"],
                    "modelSelectionMode": pipeline["modelSelectionMode"],
                    "manualModelId": pipeline["manualModelId"],
                    "autoModelMap": dict(pipeline.get("autoModelMap", {})),
                    "confidenceThreshold": pipeline["confidenceThreshold"],
                    "processingFps": pipeline["processingFps"],
                    "recordOnEvent": pipeline["recordOnEvent"],
                    "videoSourceMode": pipeline["videoSourceMode"],
                    "createdAt": pipeline["createdAt"],
                    "updatedAt": pipeline["updatedAt"],
                    "lastTelemetryAt": pipeline.get("lastTelemetryAt", ""),
                    "cameraMode": pipeline.get("cameraMode", ""),
                    "sensorType": pipeline.get("sensorType", "unknown"),
                }
                for pipeline in self._pipelines.values()
            ]
        }
        self.settings.state_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    def _copy_pipeline(self, pipeline: dict[str, object] | None) -> dict[str, object]:
        if not pipeline:
            return {}
        public = {
            **pipeline,
            "autoModelMap": dict(pipeline.get("autoModelMap", {})),
            "latestFramePath": f"/v1/pipelines/{pipeline['droneId']}/frame.jpg",
            "mjpegStreamPath": f"/v1/pipelines/{pipeline['droneId']}/stream.mjpg",
            "latestRawFramePath": f"/v1/pipelines/{pipeline['droneId']}/frame.raw.jpg",
            "rawMjpegStreamPath": f"/v1/pipelines/{pipeline['droneId']}/stream.raw.mjpg",
            "processedStreamReady": False,
        }
        worker = self._workers.get(str(pipeline["droneId"]))
        if worker:
            runtime = worker.runtime_snapshot()
            public["runtime"] = runtime
            public["processedStreamReady"] = bool(runtime.get("processedStreamReady"))
        else:
            public["runtime"] = {
                "status": "disabled" if not pipeline["analyticsEnabled"] else "configured",
                "message": "worker not running",
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
                "latestFrameAvailable": False,
                "latestProcessedFrameAvailable": False,
                "latestRawFrameAvailable": False,
                "processedStreamReady": False,
                "processedStreamUrl": str(pipeline.get("processedRtspUrl") or ""),
                "processedPublisherPid": None,
                "currentEvent": None,
                "lastRecording": None,
            }
        return public

    def _ensure_workers(self) -> None:
        for drone_id in list(self._pipelines):
            self._ensure_worker(drone_id)
        for drone_id in list(self._workers):
            if drone_id not in self._pipelines:
                self._stop_worker(drone_id)

    def _ensure_worker(self, drone_id: str, *, restart: bool = False) -> None:
        pipeline = self._pipelines.get(drone_id)
        if not pipeline:
            self._stop_worker(drone_id)
            return
        if not pipeline["analyticsEnabled"]:
            self._stop_worker(drone_id)
            return
        worker = self._workers.get(drone_id)
        if restart and worker is not None:
            self._stop_worker(drone_id)
            worker = None
        if worker is None:
            worker = DroneWorker(
                settings=self.settings,
                model_registry=self.model_registry,
                store=self.store,
                pipeline=pipeline,
                on_recording_saved=self._handle_recording_saved,
            )
            self._workers[drone_id] = worker
            worker.start()
        else:
            worker.update_pipeline(pipeline)

    def _stop_worker(self, drone_id: str) -> None:
        worker = self._workers.pop(drone_id, None)
        if worker:
            worker.stop()

    def _handle_recording_saved(self, _recording: dict[str, object]) -> None:
        self.retention_manager.enforce()

    def _worker_restart_required(
        self,
        previous: dict[str, object] | None,
        next_pipeline: dict[str, object],
    ) -> bool:
        if not previous:
            return False
        previous_auto_model_map = previous.get("autoModelMap") if isinstance(previous.get("autoModelMap"), dict) else {}
        next_auto_model_map = next_pipeline.get("autoModelMap") if isinstance(next_pipeline.get("autoModelMap"), dict) else {}
        return (
            str(previous.get("rtspUrl") or "") != str(next_pipeline.get("rtspUrl") or "")
            or str(previous.get("currentModelId") or "") != str(next_pipeline.get("currentModelId") or "")
            or str(previous.get("manualModelId") or "") != str(next_pipeline.get("manualModelId") or "")
            or str(previous.get("modelSelectionMode") or "") != str(next_pipeline.get("modelSelectionMode") or "")
            or float(previous.get("confidenceThreshold") or 0.0) != float(next_pipeline.get("confidenceThreshold") or 0.0)
            or float(previous.get("processingFps") or 0.0) != float(next_pipeline.get("processingFps") or 0.0)
            or dict(previous_auto_model_map) != dict(next_auto_model_map)
        )
