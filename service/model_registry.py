from __future__ import annotations

import json
from pathlib import Path

from service.settings import ServiceSettings

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
        self._models: dict[str, dict[str, object]] = {}
        self.reload()

    def reload(self) -> None:
        catalog_path = self.settings.model_catalog_path
        if catalog_path.exists():
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        else:
            payload = DEFAULT_MODEL_CATALOG
        raw_models = payload.get("models", payload) if isinstance(payload, dict) else payload
        self._models = {}
        for item in raw_models or []:
            model = self._normalize_model(item if isinstance(item, dict) else {})
            self._models[model["id"]] = model

    def list_models(self) -> list[dict[str, object]]:
        return sorted((self._copy_model(model) for model in self._models.values()), key=lambda item: str(item["name"]).lower())

    def get(self, model_id: str) -> dict[str, object] | None:
        model = self._models.get(str(model_id or "").strip())
        return self._copy_model(model) if model else None

    def resolve(self, model_id: str, sensor_type: str = "unknown") -> str:
        model = self._models.get(str(model_id or "").strip())
        if model and model["enabled"]:
            return str(model["id"])
        return self.default_model_id(sensor_type)

    def default_model_id(self, sensor_type: str = "unknown") -> str:
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
        return {
            "id": model_id,
            "name": str(raw.get("name") or model_id).strip() or model_id,
            "description": str(raw.get("description") or "").strip(),
            "sensorTypes": [str(value).strip().lower() for value in sensor_types if str(value).strip()],
            "defaultFor": [str(value).strip().lower() for value in default_for if str(value).strip()],
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
        return {
            **model,
            "sensorTypes": list(model.get("sensorTypes", [])),
            "defaultFor": list(model.get("defaultFor", [])),
        }

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
