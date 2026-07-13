from __future__ import annotations

from service.nvr_store import NvrStore
from service.settings import ServiceSettings


class RetentionManager:
    def __init__(self, settings: ServiceSettings, store: NvrStore) -> None:
        self.settings = settings
        self.store = store

    def enforce(self) -> dict[str, object]:
        storage = self.store.storage_summary()
        deleted: list[dict[str, object]] = []
        if storage["usedRatio"] < self.settings.storage_high_watermark:
            return {
                "triggered": False,
                "deletedCount": 0,
                "deleted": deleted,
                "storage": storage,
            }

        for recording in self.store.list_recordings(include_deleted=False, limit=100_000, ascending=True):
            if storage["usedRatio"] <= self.settings.storage_low_watermark:
                break
            deleted_recording = self.store.delete_recording(str(recording["recordingId"]), reason="retention")
            if deleted_recording:
                deleted.append(deleted_recording)
            storage = self.store.storage_summary()

        return {
            "triggered": True,
            "deletedCount": len(deleted),
            "deleted": deleted,
            "storage": storage,
        }
