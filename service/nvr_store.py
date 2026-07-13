from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
from uuid import uuid4

from service.schemas import normalize_string, utc_now
from service.settings import ServiceSettings


class NvrStore:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._ensure_schema()

    def list_recordings(
        self,
        drone_id: str | None = None,
        *,
        include_deleted: bool = False,
        limit: int = 500,
        ascending: bool = False,
    ) -> list[dict[str, object]]:
        order = "ASC" if ascending else "DESC"
        where = []
        params: list[object] = []
        if drone_id:
            where.append("drone_id = ?")
            params.append(drone_id)
        if not include_deleted:
            where.append("deleted_at IS NULL")
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"""
            SELECT
              recording_id,
              drone_id,
              event_type,
              model_id,
              sensor_type,
              started_at,
              ended_at,
              file_path,
              size_bytes,
              created_at,
              deleted_at,
              deleted_reason
            FROM recordings
            {where_sql}
            ORDER BY datetime(started_at) {order}, datetime(created_at) {order}
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_recording(self, recording_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                  recording_id,
                  drone_id,
                  event_type,
                  model_id,
                  sensor_type,
                  started_at,
                  ended_at,
                  file_path,
                  size_bytes,
                  created_at,
                  deleted_at,
                  deleted_reason
                FROM recordings
                WHERE recording_id = ?
                """,
                (recording_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def add_recording(self, payload: dict[str, object]) -> dict[str, object]:
        recording_id = normalize_string(payload.get("recordingId"), str(uuid4()))
        file_path = self._resolve_recording_path(payload.get("filePath"))
        entry = {
            "recordingId": recording_id,
            "droneId": normalize_string(payload.get("droneId")),
            "eventType": normalize_string(payload.get("eventType"), "detection"),
            "modelId": normalize_string(payload.get("modelId")),
            "sensorType": normalize_string(payload.get("sensorType"), "unknown"),
            "startedAt": normalize_string(payload.get("startedAt"), utc_now()),
            "endedAt": normalize_string(payload.get("endedAt"), utc_now()),
            "filePath": str(file_path),
            "sizeBytes": int(payload.get("sizeBytes") or (file_path.stat().st_size if file_path.exists() else 0)),
            "createdAt": normalize_string(payload.get("createdAt"), utc_now()),
            "deletedAt": None,
            "deletedReason": None,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO recordings (
                  recording_id,
                  drone_id,
                  event_type,
                  model_id,
                  sensor_type,
                  started_at,
                  ended_at,
                  file_path,
                  size_bytes,
                  created_at,
                  deleted_at,
                  deleted_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["recordingId"],
                    entry["droneId"],
                    entry["eventType"],
                    entry["modelId"],
                    entry["sensorType"],
                    entry["startedAt"],
                    entry["endedAt"],
                    entry["filePath"],
                    entry["sizeBytes"],
                    entry["createdAt"],
                    entry["deletedAt"],
                    entry["deletedReason"],
                ),
            )
        return entry

    def delete_recording(self, recording_id: str, *, reason: str = "manual") -> dict[str, object] | None:
        entry = self.get_recording(recording_id)
        if not entry:
            return None
        file_path = Path(str(entry["filePath"]))
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError:
                pass
        entry["deletedAt"] = utc_now()
        entry["deletedReason"] = reason
        entry["sizeBytes"] = int(file_path.stat().st_size) if file_path.exists() else int(entry.get("sizeBytes") or 0)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE recordings
                SET deleted_at = ?, deleted_reason = ?
                WHERE recording_id = ?
                """,
                (entry["deletedAt"], entry["deletedReason"], recording_id),
            )
        return entry

    def storage_summary(self) -> dict[str, object]:
        usage = shutil.disk_usage(self.settings.nvr_mount_dir)
        used_ratio = (usage.used / usage.total) if usage.total else 0.0
        return {
            "mountPath": str(self.settings.nvr_mount_dir),
            "totalBytes": usage.total,
            "usedBytes": usage.used,
            "freeBytes": usage.free,
            "usedRatio": round(used_ratio, 6),
            "highWatermark": self.settings.storage_high_watermark,
            "lowWatermark": self.settings.storage_low_watermark,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.nvr_db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recordings (
                  recording_id TEXT PRIMARY KEY,
                  drone_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  model_id TEXT,
                  sensor_type TEXT,
                  started_at TEXT NOT NULL,
                  ended_at TEXT NOT NULL,
                  file_path TEXT NOT NULL,
                  size_bytes INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  deleted_at TEXT,
                  deleted_reason TEXT
                )
                """
            )

    def _resolve_recording_path(self, value: object) -> Path:
        path = Path(str(value or "")).expanduser()
        if not path.is_absolute():
            path = (self.settings.nvr_mount_dir / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, object]:
        return {
            "recordingId": row["recording_id"],
            "droneId": row["drone_id"],
            "eventType": row["event_type"],
            "modelId": row["model_id"],
            "sensorType": row["sensor_type"],
            "startedAt": row["started_at"],
            "endedAt": row["ended_at"],
            "filePath": row["file_path"],
            "sizeBytes": row["size_bytes"],
            "createdAt": row["created_at"],
            "deletedAt": row["deleted_at"],
            "deletedReason": row["deleted_reason"],
        }
