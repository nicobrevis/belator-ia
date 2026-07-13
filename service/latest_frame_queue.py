from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Generic, TypeVar


FrameValue = TypeVar("FrameValue")


@dataclass(frozen=True, slots=True)
class CapturedFrame(Generic[FrameValue]):
    sequence: int
    value: FrameValue
    captured_monotonic: float
    captured_at: str


class LatestFrameQueue(Generic[FrameValue]):
    """A single-slot queue that always keeps the newest captured frame."""

    capacity = 1

    def __init__(self) -> None:
        self._queue: Queue[CapturedFrame[FrameValue]] = Queue(maxsize=self.capacity)

    def put_latest(self, frame: CapturedFrame[FrameValue]) -> int:
        """Insert ``frame`` and return how many older queued frames were evicted."""

        dropped = 0
        while True:
            try:
                self._queue.put_nowait(frame)
                return dropped
            except Full:
                try:
                    self._queue.get_nowait()
                    dropped += 1
                except Empty:
                    # A consumer won the race after Queue.full() was observed.
                    continue

    def get(self, timeout: float | None = None) -> CapturedFrame[FrameValue]:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> CapturedFrame[FrameValue]:
        return self._queue.get_nowait()

    def drain_to_latest(
        self,
        current: CapturedFrame[FrameValue] | None = None,
    ) -> tuple[CapturedFrame[FrameValue] | None, int]:
        latest = current
        drained = 0

        while True:
            try:
                latest = self._queue.get_nowait()
                drained += 1
            except Empty:
                return latest, drained

    def clear(self) -> int:
        _, removed = self.drain_to_latest()
        return removed

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()
