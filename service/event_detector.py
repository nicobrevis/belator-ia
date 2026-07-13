from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from uuid import uuid4

from service.schemas import utc_now


@dataclass(frozen=True)
class EventUpdate:
    event_id: str | None
    started: bool
    active: bool
    ended: bool
    positive: bool
    detection_count: int
    max_confidence: float
    started_at: str | None
    ended_at: str | None
    active_event: dict[str, object] | None


class EventDetector:
    def __init__(
        self,
        *,
        min_positive_frames: int = 3,
        confirmation_window_frames: int = 6,
        post_event_seconds: float = 15.0,
        cooldown_seconds: float = 20.0,
    ) -> None:
        self.min_positive_frames = max(1, int(min_positive_frames))
        self.confirmation_window_frames = max(self.min_positive_frames, int(confirmation_window_frames))
        self.post_event_seconds = max(0.0, float(post_event_seconds))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._candidate_positive_frames = 0
        self._cooldown_until_monotonic = 0.0
        self._current_event: dict[str, object] | None = None
        self._positive_history: deque[bool] = deque(maxlen=self.confirmation_window_frames)

    def update(self, *, positive: bool, detection_count: int, max_confidence: float) -> EventUpdate:
        now_monotonic = time.monotonic()
        now_iso = utc_now()
        started = False
        ended = False
        ended_at: str | None = None
        self._positive_history.append(bool(positive))

        if positive:
            self._candidate_positive_frames += 1
            if self._current_event:
                self._current_event["lastPositiveAtMonotonic"] = now_monotonic
                self._current_event["lastPositiveAt"] = now_iso
                self._current_event["detectionFrames"] = int(self._current_event["detectionFrames"]) + 1
                self._current_event["detectionsTotal"] = int(self._current_event["detectionsTotal"]) + detection_count
                self._current_event["maxConfidence"] = max(
                    float(self._current_event["maxConfidence"]),
                    float(max_confidence),
                )
            elif (
                sum(self._positive_history) >= self.min_positive_frames
                and now_monotonic >= self._cooldown_until_monotonic
            ):
                started = True
                confirmation_count = max(self._candidate_positive_frames, sum(self._positive_history))
                event_id = str(uuid4())
                self._current_event = {
                    "eventId": event_id,
                    "startedAt": now_iso,
                    "startedAtMonotonic": now_monotonic,
                    "lastPositiveAt": now_iso,
                    "lastPositiveAtMonotonic": now_monotonic,
                    "detectionFrames": confirmation_count,
                    "detectionsTotal": detection_count,
                    "maxConfidence": float(max_confidence),
                }
        else:
            self._candidate_positive_frames = 0

        if self._current_event and not positive:
            idle_seconds = now_monotonic - float(self._current_event["lastPositiveAtMonotonic"])
            if idle_seconds >= self.post_event_seconds:
                ended = True
                ended_at = now_iso
                self._cooldown_until_monotonic = now_monotonic + self.cooldown_seconds

        active_event = self._copy_current_event()
        if ended and self._current_event:
            active_event = {
                **active_event,
                "endedAt": ended_at,
            } if active_event else None
            self._current_event = None
            self._candidate_positive_frames = 0

        return EventUpdate(
            event_id=active_event["eventId"] if active_event else None,
            started=started,
            active=active_event is not None and not ended,
            ended=ended,
            positive=positive,
            detection_count=int(detection_count),
            max_confidence=float(max_confidence),
            started_at=active_event["startedAt"] if active_event else None,
            ended_at=ended_at,
            active_event=active_event,
        )

    def current_event(self) -> dict[str, object] | None:
        return self._copy_current_event()

    def _copy_current_event(self) -> dict[str, object] | None:
        if not self._current_event:
            return None
        return {
            "eventId": self._current_event["eventId"],
            "startedAt": self._current_event["startedAt"],
            "lastPositiveAt": self._current_event["lastPositiveAt"],
            "detectionFrames": int(self._current_event["detectionFrames"]),
            "detectionsTotal": int(self._current_event["detectionsTotal"]),
            "maxConfidence": float(self._current_event["maxConfidence"]),
        }
