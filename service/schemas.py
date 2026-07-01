from __future__ import annotations

from datetime import datetime, timezone

VALID_MODEL_SELECTION_MODES = {"manual", "auto"}
VALID_VIDEO_SOURCE_MODES = {"raw", "processed"}
VALID_RECORDING_SEGMENT_MODES = {"time", "size"}
KNOWN_SENSOR_TYPES = {"unknown", "wide", "thermal", "zoom", "visual"}
PIPELINE_AUTO_MODEL_KEYS = ("unknown", "wide", "thermal", "zoom", "visual")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_drone_id(value: object, fallback: str = "drone") -> str:
    normalized = "".join(
        character.lower() if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value or "").strip()
    )
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or fallback


def normalize_string(value: object, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def coerce_bool(value: object, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def normalize_model_selection_mode(value: object, fallback: str = "manual") -> str:
    normalized = str(value or fallback).strip().lower()
    return normalized if normalized in VALID_MODEL_SELECTION_MODES else fallback


def normalize_video_source_mode(value: object, fallback: str = "raw") -> str:
    normalized = str(value or fallback).strip().lower()
    return normalized if normalized in VALID_VIDEO_SOURCE_MODES else fallback


def normalize_recording_segment_mode(value: object, fallback: str = "time") -> str:
    normalized = str(value or fallback).strip().lower()
    return normalized if normalized in VALID_RECORDING_SEGMENT_MODES else fallback


def normalize_recording_segment_minutes(value: object, fallback: int = 5) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = fallback
    return min(max(normalized, 1), 120)


def normalize_recording_segment_max_mb(value: object, fallback: int = 200) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = fallback
    return min(max(normalized, 20), 4096)


def normalize_sensor_type(value: object, fallback: str = "unknown") -> str:
    normalized = str(value or fallback).strip().lower()
    tokenized = "".join(character if character.isalnum() else " " for character in normalized)
    tokens = {token for token in tokenized.split() if token}
    if (
        normalized in {"thermal", "infrared"}
        or "infra red" in normalized
        or "ir" in tokens
        or "tir" in tokens
    ):
        return "thermal"
    if normalized in {"wide", "visible", "visual"} or "wide angle" in normalized or "wide" in tokens:
        return "wide"
    return normalized if normalized in KNOWN_SENSOR_TYPES else fallback


def normalize_confidence_threshold(value: object, fallback: float = 0.15) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = fallback
    return min(max(normalized, 0.01), 0.99)


def normalize_processing_fps(value: object, fallback: float = 20.0) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = fallback
    return min(max(normalized, 0.5), 30.0)


def normalize_auto_model_map(value: object, fallback: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(fallback or {})
    source = value if isinstance(value, dict) else {}
    normalized: dict[str, str] = {}
    for key in PIPELINE_AUTO_MODEL_KEYS:
        candidate = source.get(key, base.get(key, ""))
        normalized[key] = normalize_string(candidate, base.get(key, ""))
    return normalized


def normalize_pipeline_payload(
    drone_id: str,
    payload: dict[str, object] | None,
    fallback: dict[str, object] | None = None,
) -> dict[str, object]:
    source = payload or {}
    previous = fallback or {}
    return {
        "droneId": normalize_drone_id(drone_id, normalize_drone_id(previous.get("droneId"), "drone")),
        "droneName": normalize_string(source.get("droneName"), normalize_string(previous.get("droneName"), drone_id)),
        "rtspUrl": normalize_string(source.get("rtspUrl"), normalize_string(previous.get("rtspUrl"))),
        "analyticsEnabled": coerce_bool(source.get("analyticsEnabled"), bool(previous.get("analyticsEnabled", True))),
        "sourceOnline": coerce_bool(source.get("sourceOnline"), bool(previous.get("sourceOnline", True))),
        "modelSelectionMode": normalize_model_selection_mode(
            source.get("modelSelectionMode"),
            normalize_model_selection_mode(previous.get("modelSelectionMode"), "manual"),
        ),
        "manualModelId": normalize_string(
            source.get("manualModelId"),
            normalize_string(previous.get("manualModelId")),
        ),
        "autoModelMap": normalize_auto_model_map(
            source.get("autoModelMap"),
            previous.get("autoModelMap") if isinstance(previous.get("autoModelMap"), dict) else None,
        ),
        "confidenceThreshold": normalize_confidence_threshold(
            source.get("confidenceThreshold"),
            normalize_confidence_threshold(previous.get("confidenceThreshold"), 0.15),
        ),
        "processingFps": normalize_processing_fps(
            source.get("processingFps"),
            normalize_processing_fps(previous.get("processingFps"), 20.0),
        ),
        "recordOnEvent": coerce_bool(source.get("recordOnEvent"), bool(previous.get("recordOnEvent", True))),
        "recordingSegmentMode": normalize_recording_segment_mode(
            source.get("recordingSegmentMode"),
            normalize_recording_segment_mode(previous.get("recordingSegmentMode"), "time"),
        ),
        "recordingSegmentMinutes": normalize_recording_segment_minutes(
            source.get("recordingSegmentMinutes"),
            normalize_recording_segment_minutes(previous.get("recordingSegmentMinutes"), 5),
        ),
        "recordingSegmentMaxMb": normalize_recording_segment_max_mb(
            source.get("recordingSegmentMaxMb"),
            normalize_recording_segment_max_mb(previous.get("recordingSegmentMaxMb"), 200),
        ),
        "videoSourceMode": normalize_video_source_mode(
            source.get("videoSourceMode"),
            normalize_video_source_mode(previous.get("videoSourceMode"), "raw"),
        ),
    }


def normalize_pipeline_state_payload(
    payload: dict[str, object] | None,
    fallback: dict[str, object] | None = None,
) -> dict[str, object]:
    source = payload or {}
    previous = fallback or {}
    return {
        "sensorType": normalize_sensor_type(
            source.get("sensorType"),
            normalize_sensor_type(previous.get("sensorType"), "unknown"),
        ),
        "cameraMode": normalize_string(source.get("cameraMode"), normalize_string(previous.get("cameraMode"))),
        "lastTelemetryAt": normalize_string(
            source.get("lastTelemetryAt"),
            normalize_string(previous.get("lastTelemetryAt")),
        ),
    }
