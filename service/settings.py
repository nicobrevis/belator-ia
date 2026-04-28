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
    processed_rtsp_enabled: bool
    processed_rtsp_bitrate: str
    processed_rtsp_bufsize: str
    processed_rtsp_preset: str
    storage_high_watermark: float
    storage_low_watermark: float
    ffmpeg_path: str
    inference_device: str
    inference_confidence: float
    inference_image_size: int
    processing_fps: float
    rtsp_read_timeout_seconds: float
    reconnect_delay_seconds: float
    pre_event_seconds: float
    post_event_seconds: float
    event_cooldown_seconds: float
    event_min_positive_frames: int
    event_confirmation_window_frames: int
    preview_jpeg_quality: int
    max_frame_width: int

    def processed_stream_url(self, drone_id: str) -> str:
        safe_id = quote(str(drone_id).strip(), safe="")
        return f"rtsp://{self.processed_rtsp_host}:{self.processed_rtsp_port}/processed/{safe_id}"


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
        processed_rtsp_enabled=_read_bool("PYRONE_PROCESSED_RTSP_ENABLED", True),
        processed_rtsp_bitrate=os.environ.get("PYRONE_PROCESSED_RTSP_BITRATE", "2500k").strip()
        or "2500k",
        processed_rtsp_bufsize=os.environ.get("PYRONE_PROCESSED_RTSP_BUFSIZE", "5000k").strip()
        or "5000k",
        processed_rtsp_preset=os.environ.get("PYRONE_PROCESSED_RTSP_PRESET", "veryfast").strip()
        or "veryfast",
        storage_high_watermark=storage_high_watermark,
        storage_low_watermark=storage_low_watermark,
        ffmpeg_path=os.environ.get("PYRONE_IA_FFMPEG_PATH", "ffmpeg").strip() or "ffmpeg",
        inference_device=os.environ.get("PYRONE_INFERENCE_DEVICE", "auto").strip() or "auto",
        inference_confidence=min(max(_read_float("PYRONE_INFERENCE_CONFIDENCE", 0.15), 0.01), 0.99),
        inference_image_size=max(320, _read_int("PYRONE_INFERENCE_IMAGE_SIZE", 960)),
        processing_fps=max(0.5, _read_float("PYRONE_PROCESSING_FPS", 4.0)),
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
        preview_jpeg_quality=min(max(_read_int("PYRONE_PREVIEW_JPEG_QUALITY", 80), 30), 100),
        max_frame_width=max(320, _read_int("PYRONE_MAX_FRAME_WIDTH", 1280)),
    )
    ensure_directories(settings)
    return settings
