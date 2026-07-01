from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import quote

SERVICE_DIR = Path(__file__).resolve().parent
REPO_DIR = SERVICE_DIR.parent
DEFAULT_RUNTIME_DIR = SERVICE_DIR / "runtime"
DEFAULT_NVR_DIR = DEFAULT_RUNTIME_DIR / "nvr"
PREFERRED_NVR_DIR = Path("/srv/pyrone-nvr")


def _read_int(name: str, fallback: int) -> int:
    try:
        return int(os.environ.get(name, fallback))
    except (TypeError, ValueError):
        return fallback


def _read_float(name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(name, fallback))
    except (TypeError, ValueError):
        return fallback


def _read_bool(name: str, fallback: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServiceSettings:
    repo_dir: Path
    service_dir: Path
    runtime_dir: Path
    api_host: str
    api_port: int
    model_catalog_path: Path
    state_path: Path
    nvr_mount_dir: Path
    nvr_db_path: Path
    cache_dir: Path
    events_dir: Path
    snapshots_dir: Path
    processed_rtsp_host: str
    processed_rtsp_port: int
    processed_rtsp_url_template: str
    processed_publish_transport: str
    processed_whip_url_template: str
    processed_rtmp_url_template: str
    processed_rtsp_enabled: bool
    processed_rtsp_bitrate: str
    processed_rtsp_bufsize: str
    processed_rtsp_preset: str
    processed_rtsp_write_timeout_seconds: float
    storage_high_watermark: float
    storage_low_watermark: float
    ffmpeg_path: str
    inference_device: str
    inference_confidence: float
    inference_image_size: int
    second_stage_enabled: bool
    second_stage_model_path: Path
    second_stage_conf_low: float
    second_stage_conf_high: float
    second_stage_crop_size: int
    second_stage_image_size: int
    processing_fps: float
    rtsp_read_timeout_seconds: float
    reconnect_delay_seconds: float
    pre_event_seconds: float
    post_event_seconds: float
    event_cooldown_seconds: float
    event_min_positive_frames: int
    event_confirmation_window_frames: int
    recording_segment_mode: str
    recording_segment_minutes: int
    recording_segment_max_mb: int
    preview_jpeg_quality: int
    max_frame_width: int
    mjpeg_buffer_seconds: float

    def processed_stream_url(self, drone_id: str) -> str:
        safe_id = quote(str(drone_id).strip(), safe="")
        if self.processed_rtsp_url_template:
            return self.processed_rtsp_url_template.replace("{droneId}", safe_id).replace(
                "{streamKey}",
                safe_id,
            )

        return f"rtsp://{self.processed_rtsp_host}:{self.processed_rtsp_port}/processed/{safe_id}"

    def processed_publish_url(self, drone_id: str) -> str:
        safe_id = quote(str(drone_id).strip(), safe="")

        if self.processed_publish_transport == "whip":
            template = self.processed_whip_url_template or "http://127.0.0.1:8889/processed/{droneId}/whip"
            return template.replace("{droneId}", safe_id).replace("{streamKey}", safe_id)

        if self.processed_publish_transport == "rtmp":
            template = self.processed_rtmp_url_template or "rtmp://127.0.0.1:1935/processed/{droneId}"
            return template.replace("{droneId}", safe_id).replace("{streamKey}", safe_id)

        return self.processed_stream_url(drone_id)


def ensure_directories(settings: ServiceSettings) -> None:
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    settings.nvr_mount_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.events_dir.mkdir(parents=True, exist_ok=True)
    settings.snapshots_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> ServiceSettings:
    runtime_dir = Path(
        os.environ.get("PYRONE_IA_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR))
    ).expanduser().resolve()
    default_nvr_dir = (
        PREFERRED_NVR_DIR
        if PREFERRED_NVR_DIR.exists() and os.access(PREFERRED_NVR_DIR, os.W_OK | os.X_OK)
        else DEFAULT_NVR_DIR
    )
    nvr_mount_dir = Path(
        os.environ.get("PYRONE_NVR_DIR", str(default_nvr_dir))
    ).expanduser().resolve()
    storage_high_watermark = min(max(_read_float("PYRONE_NVR_HIGH_WATERMARK", 0.90), 0.50), 0.99)
    storage_low_watermark = min(max(_read_float("PYRONE_NVR_LOW_WATERMARK", 0.85), 0.10), 0.98)
    if storage_low_watermark >= storage_high_watermark:
        storage_low_watermark = max(0.10, round(storage_high_watermark - 0.05, 2))

    processed_publish_transport = os.environ.get(
        "PYRONE_PROCESSED_PUBLISH_TRANSPORT",
        "rtmp",
    ).strip().lower()

    if processed_publish_transport not in {"rtmp", "rtsp", "whip"}:
        processed_publish_transport = "rtmp"

    settings = ServiceSettings(
        repo_dir=REPO_DIR,
        service_dir=SERVICE_DIR,
        runtime_dir=runtime_dir,
        api_host=os.environ.get("PYRONE_IA_HOST", "127.0.0.1").strip() or "127.0.0.1",
        api_port=_read_int("PYRONE_IA_PORT", 8765),
        model_catalog_path=Path(
            os.environ.get("PYRONE_MODEL_CATALOG", str(SERVICE_DIR / "models" / "catalog.json"))
        ).expanduser().resolve(),
        state_path=runtime_dir / "pipelines.json",
        nvr_mount_dir=nvr_mount_dir,
        nvr_db_path=Path(
            os.environ.get("PYRONE_NVR_DB_PATH", str(nvr_mount_dir / "pyrone_nvr.db"))
        ).expanduser().resolve(),
        cache_dir=nvr_mount_dir / "cache",
        events_dir=nvr_mount_dir / "events",
        snapshots_dir=nvr_mount_dir / "snapshots",
        processed_rtsp_host=os.environ.get("PYRONE_PROCESSED_RTSP_HOST", "127.0.0.1").strip()
        or "127.0.0.1",
        processed_rtsp_port=_read_int("PYRONE_PROCESSED_RTSP_PORT", 8554),
        processed_rtsp_url_template=os.environ.get(
            "PYRONE_PROCESSED_RTSP_URL_TEMPLATE",
            "",
        ).strip(),
        processed_publish_transport=processed_publish_transport,
        processed_whip_url_template=os.environ.get(
            "PYRONE_PROCESSED_WHIP_URL_TEMPLATE",
            "http://127.0.0.1:8889/processed/{droneId}/whip",
        ).strip(),
        processed_rtmp_url_template=os.environ.get(
            "PYRONE_PROCESSED_RTMP_URL_TEMPLATE",
            "rtmp://127.0.0.1:1935/processed/{droneId}",
        ).strip(),
        processed_rtsp_enabled=_read_bool("PYRONE_PROCESSED_RTSP_ENABLED", True),
        processed_rtsp_bitrate=os.environ.get("PYRONE_PROCESSED_RTSP_BITRATE", "2500k").strip()
        or "2500k",
        processed_rtsp_bufsize=os.environ.get("PYRONE_PROCESSED_RTSP_BUFSIZE", "5000k").strip()
        or "5000k",
        processed_rtsp_preset=os.environ.get("PYRONE_PROCESSED_RTSP_PRESET", "veryfast").strip()
        or "veryfast",
        processed_rtsp_write_timeout_seconds=max(
            1.0,
            _read_float("PYRONE_PROCESSED_RTSP_WRITE_TIMEOUT_SECONDS", 3.0),
        ),
        storage_high_watermark=storage_high_watermark,
        storage_low_watermark=storage_low_watermark,
        ffmpeg_path=os.environ.get("PYRONE_IA_FFMPEG_PATH", "ffmpeg").strip() or "ffmpeg",
        inference_device=os.environ.get("PYRONE_INFERENCE_DEVICE", "auto").strip() or "auto",
        inference_confidence=min(max(_read_float("PYRONE_INFERENCE_CONFIDENCE", 0.15), 0.01), 0.99),
        inference_image_size=max(320, _read_int("PYRONE_INFERENCE_IMAGE_SIZE", 960)),
        second_stage_enabled=_read_bool("PYRONE_SECOND_STAGE_ENABLED", True),
        second_stage_model_path=Path(
            os.environ.get(
                "PYRONE_SECOND_STAGE_MODEL_PATH",
                str(SERVICE_DIR / "models" / "ad_phash3_early_smoke_best.pt"),
            )
        ).expanduser().resolve(),
        second_stage_conf_low=min(
            max(_read_float("PYRONE_SECOND_STAGE_CONF_LOW", 0.10), 0.0),
            0.99,
        ),
        second_stage_conf_high=min(
            max(_read_float("PYRONE_SECOND_STAGE_CONF_HIGH", 0.30), 0.01),
            1.0,
        ),
        second_stage_crop_size=min(
            max(_read_int("PYRONE_SECOND_STAGE_CROP_SIZE", 224), 64),
            1024,
        ),
        second_stage_image_size=min(
            max(_read_int("PYRONE_SECOND_STAGE_IMAGE_SIZE", 224), 64),
            1024,
        ),
        processing_fps=max(0.5, _read_float("PYRONE_PROCESSING_FPS", 20.0)),
        rtsp_read_timeout_seconds=max(0.5, _read_float("PYRONE_RTSP_READ_TIMEOUT_SECONDS", 5.0)),
        reconnect_delay_seconds=max(0.5, _read_float("PYRONE_RECONNECT_DELAY_SECONDS", 2.0)),
        pre_event_seconds=max(0.0, _read_float("PYRONE_PRE_EVENT_SECONDS", 10.0)),
        post_event_seconds=max(1.0, _read_float("PYRONE_POST_EVENT_SECONDS", 15.0)),
        event_cooldown_seconds=max(0.0, _read_float("PYRONE_EVENT_COOLDOWN_SECONDS", 20.0)),
        event_min_positive_frames=max(1, _read_int("PYRONE_EVENT_MIN_POSITIVE_FRAMES", 3)),
        event_confirmation_window_frames=max(
            1,
            _read_int("PYRONE_EVENT_CONFIRMATION_WINDOW_FRAMES", 6),
        ),
        recording_segment_mode=(
            "size"
            if os.environ.get("PYRONE_RECORDING_SEGMENT_MODE", "time").strip().lower() == "size"
            else "time"
        ),
        recording_segment_minutes=min(
            max(_read_int("PYRONE_RECORDING_SEGMENT_MINUTES", 5), 1),
            120,
        ),
        recording_segment_max_mb=min(
            max(_read_int("PYRONE_RECORDING_SEGMENT_MAX_MB", 200), 20),
            4096,
        ),
        preview_jpeg_quality=min(max(_read_int("PYRONE_PREVIEW_JPEG_QUALITY", 92), 30), 100),
        max_frame_width=max(320, _read_int("PYRONE_MAX_FRAME_WIDTH", 1280)),
        mjpeg_buffer_seconds=min(
            max(_read_float("PYRONE_MJPEG_BUFFER_SECONDS", 1.0), 0.0),
            10.0,
        ),
    )
    ensure_directories(settings)
    return settings
