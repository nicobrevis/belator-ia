from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO

from service.settings import ServiceSettings


VISIBLE_SENSOR_TYPES = {"unknown", "wide", "visual", "zoom"}


@dataclass(frozen=True)
class SecondStageSummary:
    enabled: bool
    applied: bool
    available: bool
    dropped_low: int = 0
    auto_kept_high: int = 0
    sent_to_classifier: int = 0
    kept_after_classifier: int = 0
    rejected_by_classifier: int = 0
    elapsed_ms: float = 0.0
    reason: str = ""


class SecondStageClassifier:
    def __init__(self, *, settings: ServiceSettings, device: str) -> None:
        self.settings = settings
        self.device = device
        self._model: YOLO | None = None
        self._load_error = ""

    def refine_result(
        self,
        frame,
        result,
        *,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ):
        if not self._should_apply(model_info=model_info, pipeline=pipeline):
            return result, SecondStageSummary(
                enabled=self.settings.second_stage_enabled,
                applied=False,
                available=self._weights_path().exists(),
                reason=self._skip_reason(model_info=model_info, pipeline=pipeline),
            )

        model = self._ensure_model()
        if model is None:
            return result, SecondStageSummary(
                enabled=True,
                applied=False,
                available=False,
                reason=self._load_error or "second stage classifier unavailable",
            )

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) <= 0:
            return result, SecondStageSummary(enabled=True, applied=True, available=True)

        started = perf_counter()
        keep_indices: list[int] = []
        dropped_low = 0
        auto_kept_high = 0
        sent_to_classifier = 0
        kept_after_classifier = 0
        rejected_by_classifier = 0
        xyxy = boxes.xyxy.detach().cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
        confidences = boxes.conf.detach().cpu().tolist() if getattr(boxes, "conf", None) is not None else []

        for index, bbox_values in enumerate(xyxy):
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            decision = self._decision_for_candidate(
                frame,
                bbox=tuple(float(value) for value in bbox_values[:4]),
                confidence=confidence,
                classifier=model,
            )
            if decision == "drop_low":
                dropped_low += 1
            elif decision == "keep_high":
                auto_kept_high += 1
                keep_indices.append(index)
            elif decision == "keep_classifier":
                sent_to_classifier += 1
                kept_after_classifier += 1
                keep_indices.append(index)
            elif decision == "reject_classifier":
                sent_to_classifier += 1
                rejected_by_classifier += 1

        filtered = result[keep_indices] if len(keep_indices) != len(xyxy) else result
        return filtered, SecondStageSummary(
            enabled=True,
            applied=True,
            available=True,
            dropped_low=dropped_low,
            auto_kept_high=auto_kept_high,
            sent_to_classifier=sent_to_classifier,
            kept_after_classifier=kept_after_classifier,
            rejected_by_classifier=rejected_by_classifier,
            elapsed_ms=(perf_counter() - started) * 1000.0,
        )

    def refine_items(
        self,
        frame,
        items: list[dict[str, object]],
        *,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> tuple[list[dict[str, object]], SecondStageSummary]:
        if not self._should_apply(model_info=model_info, pipeline=pipeline):
            return items, SecondStageSummary(
                enabled=self.settings.second_stage_enabled,
                applied=False,
                available=self._weights_path().exists(),
                reason=self._skip_reason(model_info=model_info, pipeline=pipeline),
            )

        model = self._ensure_model()
        if model is None:
            return items, SecondStageSummary(
                enabled=True,
                applied=False,
                available=False,
                reason=self._load_error or "second stage classifier unavailable",
            )

        started = perf_counter()
        kept_items: list[dict[str, object]] = []
        dropped_low = 0
        auto_kept_high = 0
        sent_to_classifier = 0
        kept_after_classifier = 0
        rejected_by_classifier = 0

        for item in items:
            bbox = item.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) < 4:
                kept_items.append(item)
                continue

            decision = self._decision_for_candidate(
                frame,
                bbox=tuple(float(value) for value in bbox[:4]),
                confidence=float(item.get("confidence") or 0.0),
                classifier=model,
            )
            if decision == "drop_low":
                dropped_low += 1
                continue
            if decision == "keep_high":
                auto_kept_high += 1
                kept_items.append(item)
                continue
            if decision == "keep_classifier":
                sent_to_classifier += 1
                kept_after_classifier += 1
                kept_items.append({**item, "secondStage": "foreground"})
                continue
            if decision == "reject_classifier":
                sent_to_classifier += 1
                rejected_by_classifier += 1

        return kept_items, SecondStageSummary(
            enabled=True,
            applied=True,
            available=True,
            dropped_low=dropped_low,
            auto_kept_high=auto_kept_high,
            sent_to_classifier=sent_to_classifier,
            kept_after_classifier=kept_after_classifier,
            rejected_by_classifier=rejected_by_classifier,
            elapsed_ms=(perf_counter() - started) * 1000.0,
        )

    def detector_confidence_threshold(
        self,
        requested_threshold: float,
        *,
        model_info: dict[str, object],
        pipeline: dict[str, object],
    ) -> float:
        if not self._should_apply(model_info=model_info, pipeline=pipeline):
            return requested_threshold
        return min(float(requested_threshold), self.settings.second_stage_conf_low)

    def _decision_for_candidate(
        self,
        frame,
        *,
        bbox: tuple[float, float, float, float],
        confidence: float,
        classifier: YOLO,
    ) -> str:
        conf_low = self.settings.second_stage_conf_low
        conf_high = self.settings.second_stage_conf_high
        if confidence < conf_low:
            return "drop_low"
        if confidence >= conf_high:
            return "keep_high"

        crop = self._centered_crop(frame, bbox=bbox, crop_size=self.settings.second_stage_crop_size)
        if crop is None or crop.size == 0:
            return "reject_classifier"

        try:
            result = classifier.predict(
                source=crop,
                imgsz=self.settings.second_stage_image_size,
                device=self.device,
                verbose=False,
            )[0]
        except Exception as error:
            self._load_error = f"classifier predict failed: {error}"
            return "keep_classifier"

        label = self._classification_label(result)
        return "keep_classifier" if label != "background" else "reject_classifier"

    @staticmethod
    def _centered_crop(frame, *, bbox: tuple[float, float, float, float], crop_size: int):
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        half_size = crop_size / 2.0

        left = int(round(center_x - half_size))
        top = int(round(center_y - half_size))
        right = left + crop_size
        bottom = top + crop_size

        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - frame_width)
        pad_bottom = max(0, bottom - frame_height)

        clipped_left = max(0, left)
        clipped_top = max(0, top)
        clipped_right = min(frame_width, right)
        clipped_bottom = min(frame_height, bottom)

        crop = frame[clipped_top:clipped_bottom, clipped_left:clipped_right]
        if crop.size == 0:
            return None

        if pad_left or pad_top or pad_right or pad_bottom:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
        return crop

    @staticmethod
    def _classification_label(result) -> str:
        probabilities = getattr(result, "probs", None)
        if probabilities is None:
            return "background"

        class_index = int(probabilities.top1)
        names = getattr(result, "names", {}) or {}
        if isinstance(names, dict):
            return str(names.get(class_index, "background")).strip().lower()
        if isinstance(names, (list, tuple)) and 0 <= class_index < len(names):
            return str(names[class_index]).strip().lower()
        return str(class_index)

    def _should_apply(self, *, model_info: dict[str, object], pipeline: dict[str, object]) -> bool:
        if not self.settings.second_stage_enabled:
            return False
        if self.settings.second_stage_conf_low >= self.settings.second_stage_conf_high:
            return False
        if not self._weights_path().exists():
            return False
        sensor_type = str(pipeline.get("sensorType") or "unknown").strip().lower()
        return sensor_type in VISIBLE_SENSOR_TYPES and not self._uses_thermal_model(
            model_info=model_info,
            pipeline=pipeline,
        )

    def _skip_reason(self, *, model_info: dict[str, object], pipeline: dict[str, object]) -> str:
        if not self.settings.second_stage_enabled:
            return "disabled"
        if self.settings.second_stage_conf_low >= self.settings.second_stage_conf_high:
            return "invalid confidence band"
        if not self._weights_path().exists():
            return "weights missing"
        if self._uses_thermal_model(model_info=model_info, pipeline=pipeline):
            return "thermal model"
        sensor_type = str(pipeline.get("sensorType") or "unknown").strip().lower()
        if sensor_type not in VISIBLE_SENSOR_TYPES:
            return f"sensor {sensor_type} not supported"
        return ""

    @staticmethod
    def _uses_thermal_model(*, model_info: dict[str, object], pipeline: dict[str, object]) -> bool:
        sensor_type = str(pipeline.get("sensorType") or "").strip().lower()
        model_id = str(model_info.get("id") or pipeline.get("currentModelId") or "").strip().lower()
        model_name = str(model_info.get("name") or "").strip().lower()
        return (
            sensor_type in {"thermal", "ir", "infrared"}
            or "thermal" in model_id
            or "thermal" in model_name
        )

    def _ensure_model(self) -> YOLO | None:
        if self._model is not None:
            return self._model

        weights_path = self._weights_path()
        if not weights_path.exists():
            self._load_error = f"weights missing: {weights_path}"
            return None

        try:
            self._model = YOLO(str(weights_path))
            self._load_error = ""
            return self._model
        except Exception as error:
            self._load_error = f"could not load classifier: {error}"
            self._model = None
            return None

    def _weights_path(self) -> Path:
        return self.settings.second_stage_model_path
