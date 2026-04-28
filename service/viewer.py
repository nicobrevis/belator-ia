from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


DEFAULT_ANALYTICS_API_URL = "http://127.0.0.1:8765"
DEFAULT_HLS_URL_TEMPLATE = "http://127.0.0.1:8888/processed/{droneId}/index.m3u8"


@dataclass(frozen=True)
class ViewerSettings:
    analytics_api_url: str
    hls_url_template: str
    poll_seconds: float


def load_viewer_settings() -> ViewerSettings:
    analytics_api_url = (
        os.environ.get("PYRONE_VIEWER_ANALYTICS_API_URL", DEFAULT_ANALYTICS_API_URL)
        .strip()
        .rstrip("/")
        or DEFAULT_ANALYTICS_API_URL
    )
    hls_url_template = (
        os.environ.get("PYRONE_VIEWER_HLS_URL_TEMPLATE", DEFAULT_HLS_URL_TEMPLATE).strip()
        or DEFAULT_HLS_URL_TEMPLATE
    )

    try:
        poll_seconds = float(os.environ.get("PYRONE_VIEWER_POLL_SECONDS", "2"))
    except ValueError:
        poll_seconds = 2.0

    return ViewerSettings(
        analytics_api_url=analytics_api_url,
        hls_url_template=hls_url_template,
        poll_seconds=max(1.0, min(poll_seconds, 15.0)),
    )


class StandaloneIaViewer:
    def __init__(self, settings: ViewerSettings) -> None:
        self.settings = settings

    def dispatch(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        path = parsed.path.rstrip("/") or "/"

        if handler.command in {"GET", "HEAD"} and path == "/":
            return self._send_html(handler, self._viewer_html())

        if handler.command in {"GET", "HEAD"} and path == "/api/pipelines":
            return self._send_json(handler, self._pipeline_payload())

        if handler.command in {"GET", "HEAD"} and path == "/api/health":
            return self._send_json(handler, self._fetch_json("/health"))

        return self._send_json(handler, {"error": "not found"}, status=404)

    def _pipeline_payload(self) -> dict[str, object]:
        payload = self._fetch_json("/v1/pipelines")
        items = payload.get("items") if isinstance(payload, dict) else []
        pipelines = items if isinstance(items, list) else []

        return {
            "ok": True,
            "generatedAt": _utc_timestamp(),
            "analyticsApiUrl": self.settings.analytics_api_url,
            "pollSeconds": self.settings.poll_seconds,
            "items": [
                self._viewer_pipeline(item)
                for item in pipelines
                if isinstance(item, dict)
            ],
        }

    def _viewer_pipeline(self, pipeline: dict[str, object]) -> dict[str, object]:
        drone_id = str(pipeline.get("droneId") or "").strip()
        runtime = pipeline.get("runtime") if isinstance(pipeline.get("runtime"), dict) else {}
        runtime_record = runtime if isinstance(runtime, dict) else {}

        return {
            "droneId": drone_id,
            "droneName": str(pipeline.get("droneName") or drone_id or "Drone"),
            "hlsUrl": self._hls_url(drone_id),
            "mjpegUrl": (
                f"{self.settings.analytics_api_url}/v1/pipelines/{quote(drone_id, safe='')}/stream.mjpg"
                if drone_id
                else ""
            ),
            "snapshotUrl": (
                f"{self.settings.analytics_api_url}/v1/pipelines/{quote(drone_id, safe='')}/frame.jpg"
                if drone_id
                else ""
            ),
            "status": str(runtime_record.get("status") or pipeline.get("status") or "unknown"),
            "message": str(runtime_record.get("message") or ""),
            "sourceUrl": str(runtime_record.get("sourceUrl") or pipeline.get("rtspUrl") or ""),
            "sourceOpened": bool(runtime_record.get("sourceOpened")),
            "processedStreamReady": bool(runtime_record.get("processedStreamReady")),
            "processedStreamUrl": str(runtime_record.get("processedStreamUrl") or ""),
            "modelId": str(runtime_record.get("modelId") or pipeline.get("currentModelId") or ""),
            "modelName": str(runtime_record.get("modelName") or ""),
            "sensorType": str(pipeline.get("sensorType") or "unknown"),
            "cameraMode": str(pipeline.get("cameraMode") or ""),
            "processingFps": _number_or_none(runtime_record.get("processingFps")),
            "framesProcessed": int(_number_or_none(runtime_record.get("framesProcessed")) or 0),
            "lastFrameAt": str(runtime_record.get("lastFrameAt") or ""),
            "lastSourceError": str(runtime_record.get("lastSourceError") or ""),
            "frameWidth": _number_or_none(runtime_record.get("frameWidth")),
            "frameHeight": _number_or_none(runtime_record.get("frameHeight")),
        }

    def _hls_url(self, drone_id: str) -> str:
        safe_drone_id = quote(str(drone_id or "").strip(), safe="")
        return self.settings.hls_url_template.replace("{droneId}", safe_drone_id).replace(
            "{streamKey}",
            safe_drone_id,
        )

    def _fetch_json(self, path: str) -> dict[str, object]:
        request = Request(
            f"{self.settings.analytics_api_url}{path}",
            headers={"Accept": "application/json"},
            method="GET",
        )

        try:
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload if isinstance(payload, dict) else {}
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            return {"ok": False, "error": detail or f"analytics returned {error.code}"}
        except (OSError, URLError, TimeoutError) as error:
            return {"ok": False, "error": str(error)}

    def _send_html(self, handler: BaseHTTPRequestHandler, html: str, *, status: int = 200) -> None:
        self._send_bytes(
            handler,
            html.encode("utf-8"),
            content_type="text/html; charset=utf-8",
            status=status,
        )

    def _send_json(self, handler: BaseHTTPRequestHandler, payload: dict[str, object], *, status: int = 200) -> None:
        self._send_bytes(
            handler,
            json.dumps(payload, indent=2).encode("utf-8"),
            content_type="application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(
        self,
        handler: BaseHTTPRequestHandler,
        payload: bytes,
        *,
        content_type: str,
        status: int = 200,
    ) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(payload)

    def _viewer_html(self) -> str:
        return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PyrOne IA Live</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #07112e;
      --panel: rgba(13, 26, 62, 0.84);
      --panel-strong: rgba(16, 31, 73, 0.96);
      --line: rgba(160, 183, 230, 0.18);
      --text: #f6f8ff;
      --muted: #9fb0d2;
      --accent: #7cc7ff;
      --good: #48d597;
      --warn: #f2b84b;
      --bad: #ff6b7b;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 18% 14%, rgba(80, 132, 255, 0.22), transparent 34%),
        radial-gradient(circle at 84% 10%, rgba(124, 199, 255, 0.18), transparent 30%),
        linear-gradient(140deg, #07112e 0%, #0b173c 48%, #07112e 100%);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    main {{
      width: min(1540px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0;
    }}

    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(24px, 2.4vw, 40px);
      letter-spacing: -0.04em;
    }}

    .eyebrow {{
      margin: 0 0 6px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.22em;
      text-transform: uppercase;
    }}

    .header-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    select, button {{
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
      padding: 0 14px;
      font-weight: 700;
      outline: none;
    }}

    select {{
      min-width: min(440px, 80vw);
    }}

    option {{
      color: #0a1028;
    }}

    button {{
      cursor: pointer;
    }}

    .stage {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      align-items: stretch;
    }}

    .video-card {{
      position: relative;
      overflow: hidden;
      min-height: min(72vh, 820px);
      border: 1px solid var(--line);
      border-radius: 30px;
      background: #020716;
      box-shadow: 0 32px 120px rgba(0, 0, 0, 0.42);
    }}

    video, .fallback-frame {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #020716;
    }}

    .fallback-frame {{
      display: none;
    }}

    .overlay {{
      position: absolute;
      left: 18px;
      right: 18px;
      bottom: 18px;
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 14px;
      pointer-events: none;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 999px;
      background: rgba(2, 7, 22, 0.72);
      color: var(--text);
      padding: 9px 12px;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.13em;
      backdrop-filter: blur(14px);
    }}

    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--warn);
      box-shadow: 0 0 0 6px rgba(242, 184, 75, 0.14);
    }}

    .dot.live {{
      background: var(--good);
      box-shadow: 0 0 0 6px rgba(72, 213, 151, 0.14);
    }}

    .dot.error {{
      background: var(--bad);
      box-shadow: 0 0 0 6px rgba(255, 107, 123, 0.14);
    }}

    aside {{
      border: 1px solid var(--line);
      border-radius: 26px;
      background: var(--panel);
      padding: 18px;
      backdrop-filter: blur(18px);
      overflow: hidden;
    }}

    .metric {{
      padding: 13px 0;
      border-bottom: 1px solid var(--line);
    }}

    .metric:last-child {{
      border-bottom: 0;
    }}

    .label {{
      margin: 0 0 5px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }}

    .value {{
      margin: 0;
      overflow-wrap: anywhere;
      font-size: 14px;
      line-height: 1.5;
    }}

    .muted {{
      color: var(--muted);
    }}

    .error-box {{
      display: none;
      margin-top: 12px;
      border: 1px solid rgba(255, 107, 123, 0.26);
      border-radius: 18px;
      background: rgba(255, 107, 123, 0.12);
      color: #ffd6dc;
      padding: 12px;
      font-size: 13px;
      line-height: 1.5;
    }}

    @media (max-width: 1100px) {{
      .stage {{
        grid-template-columns: 1fr;
      }}

      aside {{
        order: -1;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <p class="eyebrow">PyrOne IA Live</p>
        <h1>Transmision IA independiente</h1>
      </div>
      <div class="header-actions">
        <select id="pipelineSelect" aria-label="Seleccionar drone"></select>
        <button id="reloadButton" type="button">Actualizar</button>
      </div>
    </header>

    <section class="stage">
      <div class="video-card">
        <video id="video" autoplay muted playsinline controls></video>
        <img id="fallbackFrame" class="fallback-frame" alt="Ultimo frame IA" />
        <div class="overlay">
          <span class="pill"><span id="statusDot" class="dot"></span><span id="playerStatus">Conectando</span></span>
          <span class="pill" id="transportLabel">HLS / MediaMTX</span>
        </div>
      </div>

      <aside>
        <div class="metric">
          <p class="label">Drone</p>
          <p class="value" id="droneName">Sin pipeline</p>
        </div>
        <div class="metric">
          <p class="label">Modelo</p>
          <p class="value" id="modelName">-</p>
        </div>
        <div class="metric">
          <p class="label">Estado IA</p>
          <p class="value" id="pipelineStatus">-</p>
        </div>
        <div class="metric">
          <p class="label">FPS / Frames</p>
          <p class="value" id="frameStats">-</p>
        </div>
        <div class="metric">
          <p class="label">Sensor / Camara</p>
          <p class="value" id="sensorStats">-</p>
        </div>
        <div class="metric">
          <p class="label">Resolucion</p>
          <p class="value" id="resolutionStats">-</p>
        </div>
        <div class="metric">
          <p class="label">Fuente RC</p>
          <p class="value muted" id="sourceUrl">-</p>
        </div>
        <div class="metric">
          <p class="label">Stream IA</p>
          <p class="value muted" id="hlsUrl">-</p>
        </div>
        <div id="errorBox" class="error-box"></div>
      </aside>
    </section>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/hls.js@1.6.16/dist/hls.min.js"></script>
  <script>
    const pollSeconds = {self.settings.poll_seconds:.2f};
    const state = {{
      hls: null,
      pipelines: [],
      selectedId: "",
      activeUrl: "",
      reconnectTimer: 0,
      snapshotTimer: 0,
    }};

    const els = {{
      select: document.getElementById("pipelineSelect"),
      reloadButton: document.getElementById("reloadButton"),
      video: document.getElementById("video"),
      fallbackFrame: document.getElementById("fallbackFrame"),
      statusDot: document.getElementById("statusDot"),
      playerStatus: document.getElementById("playerStatus"),
      transportLabel: document.getElementById("transportLabel"),
      droneName: document.getElementById("droneName"),
      modelName: document.getElementById("modelName"),
      pipelineStatus: document.getElementById("pipelineStatus"),
      frameStats: document.getElementById("frameStats"),
      sensorStats: document.getElementById("sensorStats"),
      resolutionStats: document.getElementById("resolutionStats"),
      sourceUrl: document.getElementById("sourceUrl"),
      hlsUrl: document.getElementById("hlsUrl"),
      errorBox: document.getElementById("errorBox"),
    }};

    function setPlayerStatus(status, detail) {{
      els.statusDot.className = "dot" + (status === "live" ? " live" : status === "error" ? " error" : "");
      els.playerStatus.textContent = detail || (status === "live" ? "En vivo" : status === "error" ? "Error" : "Conectando");
    }}

    function showError(message) {{
      if (!message) {{
        els.errorBox.style.display = "none";
        els.errorBox.textContent = "";
        return;
      }}

      els.errorBox.style.display = "block";
      els.errorBox.textContent = message;
    }}

    function destroyPlayer() {{
      if (state.hls) {{
        try {{ state.hls.destroy(); }} catch (error) {{}}
        state.hls = null;
      }}

      if (state.reconnectTimer) {{
        window.clearTimeout(state.reconnectTimer);
        state.reconnectTimer = 0;
      }}

      els.video.pause();
      els.video.removeAttribute("src");
      els.video.load();
      state.activeUrl = "";
    }}

    function startSnapshotFallback(pipeline) {{
      if (state.snapshotTimer) {{
        window.clearInterval(state.snapshotTimer);
        state.snapshotTimer = 0;
      }}

      if (!pipeline?.snapshotUrl) {{
        return;
      }}

      els.fallbackFrame.style.display = "block";
      const refresh = () => {{
        els.fallbackFrame.src = pipeline.snapshotUrl + "?ts=" + Date.now();
      }};
      refresh();
      state.snapshotTimer = window.setInterval(refresh, 1000);
    }}

    function stopSnapshotFallback() {{
      if (state.snapshotTimer) {{
        window.clearInterval(state.snapshotTimer);
        state.snapshotTimer = 0;
      }}
      els.fallbackFrame.style.display = "none";
      els.fallbackFrame.removeAttribute("src");
    }}

    function reconnectCurrent(delayMs = 1200) {{
      if (state.reconnectTimer) {{
        return;
      }}

      setPlayerStatus("connecting", "Reconectando");
      state.reconnectTimer = window.setTimeout(() => {{
        state.reconnectTimer = 0;
        const pipeline = currentPipeline();
        if (pipeline) {{
          connectPipeline(pipeline, true);
        }}
      }}, delayMs);
    }}

    function connectPipeline(pipeline, force = false) {{
      if (!pipeline?.hlsUrl) {{
        destroyPlayer();
        startSnapshotFallback(pipeline);
        setPlayerStatus("error", "Sin HLS");
        return;
      }}

      if (!force && state.activeUrl === pipeline.hlsUrl) {{
        return;
      }}

      destroyPlayer();
      stopSnapshotFallback();
      showError(pipeline.lastSourceError || "");
      setPlayerStatus("connecting", "Cargando HLS");
      state.activeUrl = pipeline.hlsUrl;
      els.transportLabel.textContent = "HLS / MediaMTX";

      els.video.onplaying = () => setPlayerStatus("live", "En vivo");
      els.video.onloadeddata = () => setPlayerStatus("live", "En vivo");
      els.video.onerror = () => reconnectCurrent();
      els.video.onemptied = () => reconnectCurrent();
      els.video.onabort = () => reconnectCurrent();

      if (els.video.canPlayType("application/vnd.apple.mpegurl")) {{
        els.video.src = pipeline.hlsUrl;
        els.video.play().catch(() => null);
        return;
      }}

      if (window.Hls && window.Hls.isSupported()) {{
        const hls = new window.Hls({{
          backBufferLength: 20,
          enableWorker: true,
          liveDurationInfinity: true,
          liveMaxLatencyDurationCount: 8,
          liveSyncDurationCount: 3,
          lowLatencyMode: true,
          maxBufferLength: 18,
          maxLiveSyncPlaybackRate: 1.5,
        }});
        state.hls = hls;
        hls.on(window.Hls.Events.ERROR, (_event, data) => {{
          if (!data.fatal) {{
            return;
          }}

          if (data.type === window.Hls.ErrorTypes.NETWORK_ERROR) {{
            hls.startLoad();
            return;
          }}

          if (data.type === window.Hls.ErrorTypes.MEDIA_ERROR) {{
            hls.recoverMediaError();
            return;
          }}

          reconnectCurrent();
        }});
        hls.attachMedia(els.video);
        hls.loadSource(pipeline.hlsUrl);
        els.video.play().catch(() => null);
        return;
      }}

      startSnapshotFallback(pipeline);
      setPlayerStatus("error", "HLS.js no disponible");
      showError("El navegador no soporta HLS nativo y no pudo cargar HLS.js. Se muestra el ultimo frame como respaldo.");
    }}

    function currentPipeline() {{
      return state.pipelines.find((pipeline) => pipeline.droneId === state.selectedId) || state.pipelines[0] || null;
    }}

    function updateSelect() {{
      const current = state.selectedId;
      els.select.innerHTML = "";

      if (state.pipelines.length === 0) {{
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "Sin pipelines activos";
        els.select.appendChild(option);
        state.selectedId = "";
        return;
      }}

      for (const pipeline of state.pipelines) {{
        const option = document.createElement("option");
        option.value = pipeline.droneId;
        option.textContent = pipeline.droneName + (pipeline.processedStreamReady ? " - IA lista" : " - esperando IA");
        els.select.appendChild(option);
      }}

      if (current && state.pipelines.some((pipeline) => pipeline.droneId === current)) {{
        state.selectedId = current;
      }} else {{
        const ready = state.pipelines.find((pipeline) => pipeline.processedStreamReady && pipeline.sourceOpened);
        state.selectedId = (ready || state.pipelines[0]).droneId;
      }}

      els.select.value = state.selectedId;
    }}

    function renderPipeline(pipeline) {{
      if (!pipeline) {{
        els.droneName.textContent = "Sin pipeline";
        els.modelName.textContent = "-";
        els.pipelineStatus.textContent = "-";
        els.frameStats.textContent = "-";
        els.sensorStats.textContent = "-";
        els.resolutionStats.textContent = "-";
        els.sourceUrl.textContent = "-";
        els.hlsUrl.textContent = "-";
        setPlayerStatus("error", "Sin pipeline");
        return;
      }}

      els.droneName.textContent = pipeline.droneName || pipeline.droneId;
      els.modelName.textContent = pipeline.modelName || pipeline.modelId || "-";
      els.pipelineStatus.textContent = [
        pipeline.status || "unknown",
        pipeline.sourceOpened ? "fuente abierta" : "sin fuente",
        pipeline.processedStreamReady ? "stream listo" : "stream no listo",
      ].join(" / ");
      els.frameStats.textContent = `${{pipeline.processingFps || 0}} fps / ${{pipeline.framesProcessed || 0}} frames`;
      els.sensorStats.textContent = `${{pipeline.sensorType || "unknown"}} / ${{pipeline.cameraMode || "sin dato"}}`;
      els.resolutionStats.textContent = pipeline.frameWidth && pipeline.frameHeight
        ? `${{pipeline.frameWidth}} x ${{pipeline.frameHeight}}`
        : "-";
      els.sourceUrl.textContent = pipeline.sourceUrl || "-";
      els.hlsUrl.textContent = pipeline.hlsUrl || "-";
      showError(pipeline.lastSourceError || "");
    }}

    async function loadPipelines() {{
      try {{
        const response = await fetch("/api/pipelines", {{ cache: "no-store" }});
        const payload = await response.json();

        if (!response.ok || payload.ok === false) {{
          throw new Error(payload.error || "No se pudo consultar la API IA");
        }}

        state.pipelines = Array.isArray(payload.items) ? payload.items : [];
        updateSelect();
        const pipeline = currentPipeline();
        renderPipeline(pipeline);
        if (pipeline) {{
          connectPipeline(pipeline);
        }}
      }} catch (error) {{
        showError(error instanceof Error ? error.message : "Error desconocido");
        setPlayerStatus("error", "API IA no disponible");
      }}
    }}

    els.select.addEventListener("change", () => {{
      state.selectedId = els.select.value;
      const pipeline = currentPipeline();
      renderPipeline(pipeline);
      if (pipeline) {{
        connectPipeline(pipeline, true);
      }}
    }});

    els.reloadButton.addEventListener("click", () => {{
      loadPipelines();
      const pipeline = currentPipeline();
      if (pipeline) {{
        connectPipeline(pipeline, true);
      }}
    }});

    loadPipelines();
    window.setInterval(loadPipelines, Math.round(pollSeconds * 1000));
  </script>
</body>
</html>
"""


def _number_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_viewer_handler(app: StandaloneIaViewer):
    class ViewerHandler(BaseHTTPRequestHandler):
        server_version = "PyrOneIaViewer/0.1"

        def do_GET(self) -> None:  # noqa: N802
            app.dispatch(self)

        def do_HEAD(self) -> None:  # noqa: N802
            app.dispatch(self)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[ia-viewer] {self.address_string()} - {format % args}")

    return ViewerHandler
