from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess

import cv2

from service.event_detector import EventUpdate
from service.nvr_store import NvrStore
from service.schemas import utc_now
from service.settings import ServiceSettings


def _safe_part(value: object, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip()).strip("-")
    return normalized or fallback


class EventRecorder:
    def __init__(self, settings: ServiceSettings, store: NvrStore) -> None:
        self.settings = settings
        self.store = store
        self._buffer: deque[tuple[str, object]] = deque()
        self._writer: cv2.VideoWriter | None = None
        self._recording: dict[str, object] | None = None
        self._event_payload: dict[str, object] | None = None
        self._event_metadata: dict[str, object] | None = None
        self._segment_index = 0
        self._frame_size: tuple[int, int] | None = None
        self._buffer_limit = 1

    def configure(self, *, frame_width: int, frame_height: int, fps: float) -> None:
        frame_width = max(1, int(frame_width))
        frame_height = max(1, int(frame_height))
        fps = max(1.0, float(fps))
        frame_size = (frame_width, frame_height)
        if self._frame_size == frame_size and self._recording and abs(float(self._recording["fps"]) - fps) < 0.01:
            return
        if self._writer:
            self._finalize(utc_now(), clear_event=True)
        self._frame_size = frame_size
        self._buffer_limit = max(1, int(round(self.settings.pre_event_seconds * fps)))

    def ingest(
        self,
        frame,
        *,
        timestamp: str,
        event: EventUpdate,
        metadata: dict[str, object],
        fps: float,
    ) -> dict[str, object] | None:
        if self._frame_size is None:
            self.configure(frame_width=frame.shape[1], frame_height=frame.shape[0], fps=fps)

        if not self._writer:
            self._buffer.append((timestamp, frame.copy()))
            while len(self._buffer) > self._buffer_limit:
                self._buffer.popleft()

        if event.started and event.active_event:
            self._start(event.active_event, metadata, fps, frame)

        if self._writer and not event.started:
            self._writer.write(frame)

        if event.ended:
            return self._finalize(event.ended_at or timestamp, clear_event=True)

        if self._writer and self._should_rotate(timestamp):
            recording = self._finalize(timestamp, clear_event=False)
            self._start_segment(
                event_payload=self._event_payload or event.active_event or {},
                metadata=self._event_metadata or metadata,
                fps=fps,
                event_frame=frame,
                started_at=timestamp,
            )
            return recording
        return None

    def close(self) -> dict[str, object] | None:
        if not self._writer:
            return None
        return self._finalize(utc_now(), clear_event=True)

    def _start(self, event_payload: dict[str, object], metadata: dict[str, object], fps: float, event_frame) -> None:
        self._event_payload = dict(event_payload)
        self._event_metadata = dict(metadata)
        self._segment_index = 0
        self._start_segment(
            event_payload=event_payload,
            metadata=metadata,
            fps=fps,
            event_frame=event_frame,
            started_at=str(event_payload.get("startedAt") or utc_now()),
        )

    def _start_segment(
        self,
        *,
        event_payload: dict[str, object],
        metadata: dict[str, object],
        fps: float,
        event_frame,
        started_at: str,
    ) -> None:
        if self._frame_size is None:
            self._frame_size = (event_frame.shape[1], event_frame.shape[0])
        self._segment_index += 1
        segment_index = self._segment_index
        event_id = str(event_payload.get("eventId") or utc_now())
        segment_started_at = str(started_at or utc_now())
        event_time = self._parse_timestamp(segment_started_at)
        day_dir = (
            self.settings.events_dir
            / event_time.strftime("%Y")
            / event_time.strftime("%m")
            / event_time.strftime("%d")
            / _safe_part(metadata.get("droneId"), "drone")
        )
        day_dir.mkdir(parents=True, exist_ok=True)
        file_stem = "_".join(
            [
                event_time.strftime("%Y%m%dT%H%M%SZ"),
                _safe_part(metadata.get("droneId"), "drone"),
                _safe_part(metadata.get("eventType"), "event"),
                _safe_part(metadata.get("modelId"), "model"),
                f"seg-{segment_index:03d}",
            ]
        )
        file_path = day_dir / f"{file_stem}.mp4"
        working_file_path = day_dir / f"{file_stem}.raw.mp4"
        writer = cv2.VideoWriter(
            str(working_file_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, float(fps)),
            self._frame_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for recording {file_path}")

        snapshot_dir = self.settings.snapshots_dir / _safe_part(metadata.get("droneId"), "drone")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{file_stem}.jpg"
        cv2.imwrite(str(snapshot_path), event_frame)
        segment_mode = self._segment_mode(metadata)
        segment_minutes = self._segment_minutes(metadata)
        segment_max_mb = self._segment_max_mb(metadata)

        self._writer = writer
        self._recording = {
            "recordingId": f"{event_id}-seg-{segment_index:03d}",
            "droneId": str(metadata.get("droneId") or ""),
            "eventType": str(metadata.get("eventType") or "smoke_detection"),
            "modelId": str(metadata.get("modelId") or ""),
            "sensorType": str(metadata.get("sensorType") or "unknown"),
            "startedAt": segment_started_at,
            "filePath": str(file_path),
            "workingFilePath": str(working_file_path),
            "snapshotPath": str(snapshot_path),
            "fps": float(fps),
            "eventId": event_id,
            "segmentIndex": segment_index,
            "segmentMode": segment_mode,
            "segmentMinutes": segment_minutes,
            "segmentMaxMb": segment_max_mb,
        }
        if segment_index == 1:
            for _, buffered_frame in list(self._buffer):
                self._writer.write(buffered_frame)
            self._buffer.clear()
        else:
            self._writer.write(event_frame)

    def _finalize(self, ended_at: str, *, clear_event: bool) -> dict[str, object] | None:
        if not self._writer or not self._recording:
            return None

        self._writer.release()
        self._writer = None
        file_path = Path(str(self._recording["filePath"]))
        working_file_path = Path(str(self._recording.get("workingFilePath") or file_path))

        if working_file_path.exists() and working_file_path != file_path:
            if self._transcode_browser_mp4(working_file_path, file_path):
                try:
                    working_file_path.unlink()
                except OSError:
                    pass
            else:
                working_file_path.replace(file_path)

        recording = self.store.add_recording(
            {
                **self._recording,
                "endedAt": ended_at,
                "sizeBytes": file_path.stat().st_size if file_path.exists() else 0,
            }
        )
        self._recording = None
        if clear_event:
            self._event_payload = None
            self._event_metadata = None
            self._segment_index = 0
        return recording

    def _should_rotate(self, timestamp: str) -> bool:
        if not self._recording:
            return False

        if str(self._recording.get("segmentMode") or "time") == "size":
            working_file_path = Path(str(self._recording.get("workingFilePath") or self._recording.get("filePath")))
            try:
                return working_file_path.stat().st_size >= int(self._recording.get("segmentMaxMb") or 200) * 1024 * 1024
            except OSError:
                return False

        started_at = self._parse_timestamp(str(self._recording.get("startedAt") or utc_now()))
        current_at = self._parse_timestamp(timestamp)
        return (
            current_at - started_at
        ).total_seconds() >= int(self._recording.get("segmentMinutes") or 5) * 60

    def _segment_mode(self, metadata: dict[str, object]) -> str:
        mode = str(metadata.get("recordingSegmentMode") or self.settings.recording_segment_mode).strip().lower()
        return "size" if mode == "size" else "time"

    def _segment_minutes(self, metadata: dict[str, object]) -> int:
        try:
            value = int(metadata.get("recordingSegmentMinutes") or self.settings.recording_segment_minutes)
        except (TypeError, ValueError):
            value = self.settings.recording_segment_minutes
        return min(max(value, 1), 120)

    def _segment_max_mb(self, metadata: dict[str, object]) -> int:
        try:
            value = int(metadata.get("recordingSegmentMaxMb") or self.settings.recording_segment_max_mb)
        except (TypeError, ValueError):
            value = self.settings.recording_segment_max_mb
        return min(max(value, 20), 4096)

    def _transcode_browser_mp4(self, source_path: Path, output_path: Path) -> bool:
        tmp_path = output_path.with_name(f"{output_path.stem}.tmp.mp4")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            subprocess.run(
                [
                    self.settings.ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    "-profile:v",
                    "baseline",
                    "-movflags",
                    "+faststart",
                    str(tmp_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tmp_path.replace(output_path)
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return False

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)
