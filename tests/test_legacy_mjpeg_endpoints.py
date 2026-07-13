from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import unittest

from service.api import AnalyticsServiceApp
from service.pipeline_manager import PipelineManager


class _Handler:
    def __init__(self, path: str) -> None:
        self.path = path
        self.command = "GET"
        self.headers: dict[str, str] = {}
        self.rfile = BytesIO()
        self.wfile = BytesIO()
        self.status = 0

    def send_response(self, status: int) -> None:
        self.status = status

    @staticmethod
    def send_header(_name: str, _value: str) -> None:
        return

    @staticmethod
    def end_headers() -> None:
        return


class LegacyMjpegEndpointTests(unittest.TestCase):
    def test_legacy_frame_and_stream_endpoints_are_unavailable_when_disabled(self) -> None:
        app = object.__new__(AnalyticsServiceApp)
        app.settings = SimpleNamespace(legacy_mjpeg_enabled=False)
        paths = (
            "/v1/pipelines/drone-1/frame.jpg",
            "/v1/pipelines/drone-1/frame.raw.jpg",
            "/v1/pipelines/drone-1/stream.mjpg",
            "/v1/pipelines/drone-1/stream.raw.mjpg",
        )

        for path in paths:
            handler = _Handler(path)
            app.dispatch(handler)
            self.assertEqual(handler.status, 404, path)
            self.assertIn(b"legacy MJPEG is disabled", handler.wfile.getvalue())

    def test_disabled_legacy_paths_are_not_advertised(self) -> None:
        manager = object.__new__(PipelineManager)
        manager.settings = SimpleNamespace(
            legacy_mjpeg_enabled=False,
            processed_publish_transport="rtmp",
        )
        manager._workers = {}

        public = manager._copy_pipeline(
            {
                "droneId": "drone-1",
                "analyticsEnabled": True,
                "sourceOnline": True,
                "processedRtspUrl": "rtsp://media/processed/drone-1",
            }
        )

        self.assertEqual(public["latestFramePath"], "")
        self.assertEqual(public["mjpegStreamPath"], "")
        self.assertEqual(public["latestRawFramePath"], "")
        self.assertEqual(public["rawMjpegStreamPath"], "")


if __name__ == "__main__":
    unittest.main()
