from __future__ import annotations

import json
from pathlib import Path
import threading

from service.settings import ServiceSettings


# Keep persisted pipeline configurations valid when a model is renamed.  New
# renames should be added here (or declared through a model's ``aliases`` list
# in catalog.json) instead of requiring every stored drone profile to be
# rewritten at exactly the same time as the catalog deployment.
MODEL_ID_ALIASES = {
    "pyronear-yolov8s-wide": "pyrone-yolov8s-wide",
}

DEFAULT_MODEL_CATALOG = {
    "models": [
        {
            "id": "smoke-wide-v1",
            "name": "Smoke Wide v1",
            "description": "Wide-angle smoke segmentation baseline for daylight RGB footage.",
            "sensorTypes": ["wide", "visual", "zoom", "unknown"],
            "defaultFor": ["wide", "visual", "unknown"],
            "weightsPath": "smoke_dataset/runs/pyrone_172_v1_yolo11n_seg_e80_i960_b2/weights/best.pt",
            "enabled": True,
        },
        {
            "id": "smoke-thermal-v1",
            "name": "Smoke Thermal v1",
            "description": "Thermal smoke segmentation baseline for H30T IR video.",
            "sensorTypes": ["thermal"],
            "defaultFor": ["thermal"],
            "weightsPath": "smoke_dataset/runs/thermal_v1_yolo11n_seg_e100_i1024_b2/weights/best.pt",
            "enabled": True,
        },
    ]
}


class ModelRegistry:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._models: dict[str, dict[str, object]] = {}
        self._aliases: dict[str, str] = {}
        self._has_catalog_state = False
        self._catalog_signature: tuple[bool, int, int] | None = None
        self._revision = 0
        self._last_reload_error = ""
        self.reload()

    def reload(self) -> bool:
        catalog_path = self.settings.model_catalog_path
        signature = self._catalog_file_signature()
        try:
            if signature[0]:
                payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            else:
                # A catalog can briefly disappear while it is being replaced
                # (for example, by an atomic deployment).  Once a valid
                # catalog has been loaded, keep serving it and retry the file
                # on subsequent lookups instead of silently switching every
                # running pipeline back to the built-in defaults.
                with self._lock:
                    if self._has_catalog_state:
                        self._last_reload_error = (
                            f"model catalog is unavailable: {catalog_path}"
                        )
                        return False
                payload = DEFAULT_MODEL_CATALOG
            raw_models = payload.get("models", payload) if isinstance(payload, dict) else payload
            if not isinstance(raw_models, list):
                raise ValueError("model catalog must contain a models list")

            models: dict[str, dict[str, object]] = {}
            aliases: dict[str, str] = {}
            for item in raw_models:
                model = self._normalize_model(item if isinstance(item, dict) else {})
                model_id = str(model["id"])
                models[model_id] = model
                for alias in model.get("aliases", []):
                    normalized_alias = str(alias or "").strip()
                    if normalized_alias and normalized_alias != model_id:
                        aliases[normalized_alias] = model_id

            for alias, target in MODEL_ID_ALIASES.items():
                if target in models and alias not in models:
                    aliases[alias] = target
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            with self._lock:
                self._last_reload_error = str(error)
                if not self._has_catalog_state:
                    self._load_fallback_catalog_locked()
            return False

        with self._lock:
            self._models = models
            self._aliases = aliases
            self._has_catalog_state = True
            self._catalog_signature = signature
            self._last_reload_error = ""
            self._revision += 1
        return True

    def reload_if_changed(self) -> bool:
        signature = self._catalog_file_signature()
        with self._lock:
            changed = signature != self._catalog_signature
        return self.reload() if changed else False

    @property
    def revision(self) -> int:
        self.reload_if_changed()
        with self._lock:
            return self._revision

    @property
    def last_reload_error(self) -> str:
        with self._lock:
            return self._last_reload_error

    def list_models(self) -> list[dict[str, object]]:
        self.reload_if_changed()
        with self._lock:
            models = [self._copy_model(model) for model in self._models.values()]
        return sorted(models, key=lambda item: str(item["name"]).lower())

    def get(self, model_id: str) -> dict[str, object] | None:
        self.reload_if_changed()
        requested_id = str(model_id or "").strip()
        with self._lock:
            canonical_id = self._canonical_id_locked(requested_id)
            model = self._models.get(canonical_id)
            return self._copy_model(model) if model else None

    def canonical_id(self, model_id: str) -> str:
        self.reload_if_changed()
        with self._lock:
            return self._canonical_id_locked(str(model_id or "").strip())

    def resolve(self, model_id: str, sensor_type: str = "unknown") -> str:
        self.reload_if_changed()
        with self._lock:
            canonical_id = self._canonical_id_locked(str(model_id or "").strip())
            model = self._models.get(canonical_id)
            if model and model["enabled"]:
                return str(model["id"])
            return self._default_model_id_locked(sensor_type)

    def default_model_id(self, sensor_type: str = "unknown") -> str:
        self.reload_if_changed()
        with self._lock:
            return self._default_model_id_locked(sensor_type)

    def _default_model_id_locked(self, sensor_type: str = "unknown") -> str:
        target = str(sensor_type or "unknown").strip().lower() or "unknown"
        for model in self._models.values():
            if model["enabled"] and target in model["defaultFor"]:
                return str(model["id"])
        for model in self._models.values():
            if model["enabled"]:
                return str(model["id"])
        raise RuntimeError("No enabled analytics models are configured.")

    def _normalize_model(self, raw: dict[str, object]) -> dict[str, object]:
        model_id = str(raw.get("id") or "").strip() or "unknown-model"
        weights_path = Path(str(raw.get("weightsPath") or "")).expanduser()
        if not weights_path.is_absolute():
            weights_path = (self.settings.repo_dir / weights_path).resolve()
        sensor_types = raw.get("sensorTypes") if isinstance(raw.get("sensorTypes"), list) else []
        default_for = raw.get("defaultFor") if isinstance(raw.get("defaultFor"), list) else sensor_types
        aliases = raw.get("aliases") if isinstance(raw.get("aliases"), list) else []
        return {
            "id": model_id,
            "name": str(raw.get("name") or model_id).strip() or model_id,
            "description": str(raw.get("description") or "").strip(),
            "sensorTypes": [str(value).strip().lower() for value in sensor_types if str(value).strip()],
            "defaultFor": [str(value).strip().lower() for value in default_for if str(value).strip()],
            "aliases": [str(value).strip() for value in aliases if str(value).strip()],
            "weightsPath": str(weights_path),
            "weightsPresent": weights_path.exists(),
            "enabled": bool(raw.get("enabled", True)),
            "imageSize": int(raw.get("imageSize") or 0) or None,
            "iouThreshold": self._normalize_optional_float(raw.get("iouThreshold")),
            "preprocessor": self._normalize_preprocessor(raw.get("preprocessor")),
            "compositeIntermediateWidth": self._normalize_optional_int(
                raw.get("compositeIntermediateWidth")
            ),
            "compositeNmsIou": self._normalize_optional_float(raw.get("compositeNmsIou")),
        }

    def _copy_model(self, model: dict[str, object] | None) -> dict[str, object]:
        if not model:
            return {}
        weights_path = Path(str(model.get("weightsPath") or ""))
        return {
            **model,
            "sensorTypes": list(model.get("sensorTypes", [])),
            "defaultFor": list(model.get("defaultFor", [])),
            "aliases": list(model.get("aliases", [])),
            # Weights can be copied into place after the catalog is loaded.  Do
            # not require a service restart just to observe that transition.
            "weightsPresent": weights_path.is_file(),
        }

    def _catalog_file_signature(self) -> tuple[bool, int, int]:
        try:
            stat = self.settings.model_catalog_path.stat()
        except OSError:
            return False, 0, 0
        return True, int(stat.st_mtime_ns), int(stat.st_size)

    def _canonical_id_locked(self, model_id: str) -> str:
        current = model_id
        visited: set[str] = set()
        while current in self._aliases and current not in visited:
            visited.add(current)
            current = self._aliases[current]
        return current

    def _load_fallback_catalog_locked(self) -> None:
        models: dict[str, dict[str, object]] = {}
        for item in DEFAULT_MODEL_CATALOG["models"]:
            model = self._normalize_model(item)
            models[str(model["id"])] = model
        self._models = models
        self._aliases = {
            alias: target
            for alias, target in MODEL_ID_ALIASES.items()
            if target in models and alias not in models
        }
        self._has_catalog_state = True
        self._revision += 1

    @staticmethod
    def _normalize_optional_float(value: object) -> float | None:
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0:
            return None
        return normalized

    @staticmethod
    def _normalize_optional_int(value: object) -> int | None:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0:
            return None
        return normalized

    @staticmethod
    def _normalize_preprocessor(value: object) -> str:
        normalized = str(value or "none").strip().lower()
        return normalized if normalized in {"none", "skyline_composite"} else "none"
