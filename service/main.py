from __future__ import annotations

from http.server import ThreadingHTTPServer

from service.api import AnalyticsServiceApp, build_handler
from service.settings import load_settings


def main() -> None:
    settings = load_settings()
    app = AnalyticsServiceApp(settings)
    server = ThreadingHTTPServer((settings.api_host, settings.api_port), build_handler(app))
    print(
        "[analytics] listening on "
        f"http://{settings.api_host}:{settings.api_port} "
        f"(NVR: {settings.nvr_mount_dir})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[analytics] stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
