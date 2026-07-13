from __future__ import annotations

import unittest

from service.latest_frame_queue import CapturedFrame, LatestFrameQueue


class LatestFrameQueueTests(unittest.TestCase):
    def test_new_frame_evicts_old_frame(self) -> None:
        frames: LatestFrameQueue[str] = LatestFrameQueue()
        first = CapturedFrame(1, "old", 1.0, "first")
        newest = CapturedFrame(2, "new", 2.0, "second")

        self.assertEqual(frames.put_latest(first), 0)
        self.assertEqual(frames.put_latest(newest), 1)
        self.assertEqual(frames.qsize(), 1)
        self.assertEqual(frames.get_nowait(), newest)

    def test_drain_returns_newest_without_growing_queue(self) -> None:
        frames: LatestFrameQueue[str] = LatestFrameQueue()
        current = CapturedFrame(1, "current", 1.0, "first")
        latest = CapturedFrame(2, "latest", 2.0, "second")
        frames.put_latest(latest)

        selected, superseded = frames.drain_to_latest(current)

        self.assertEqual(selected, latest)
        self.assertEqual(superseded, 1)
        self.assertTrue(frames.empty())


if __name__ == "__main__":
    unittest.main()
