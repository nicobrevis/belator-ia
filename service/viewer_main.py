from __future__ import annotations

from http.server import ThreadingHTTPServer
import os

from service.viewer import StandaloneIaViewer, build_viewer_handler, load_viewer_settings


def _read_port() -> int:
    try:
        return int(os.environ.get("PYRONE_VIEWER_PORT", "3001"))
    except ValueError:
        return 3001


def main() -> None:
    host = os.environ.get("PYRONE_VIEWER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = _read_port()
    settings = load_viewer_settings()
    app = StandaloneIaViewer(settings)
    server = ThreadingHTTPServer((host, port), build_viewer_handler(app))
    print(
        "[ia-viewer] listening on "
        f"http://{host}:{port} "
        f"(analytics: {settings.analytics_api_url})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ia-viewer] stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
