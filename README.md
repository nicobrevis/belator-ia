# belator-ia

Repositorio de trabajo para el dataset y entrenamiento de detección/segmentación de humo con YOLO.

Contenido principal:

- `smoke_dataset/`: scripts, datasets preparados, exports, runs y predicciones
- `videos nuevos/`: videos adicionales usados para ampliar y validar el dataset
- `PyrOne 172/`: export base anotado desde CVAT
- `service/`: servicio interno de analitica en vivo, catalogo de modelos y base NVR

El detalle del flujo de trabajo y entrenamiento está en `smoke_dataset/README.md`.

## Servicio de analitica

El esqueleto inicial del backend de inferencia y NVR vive en `service/`.

Para iniciarlo desde la raiz del repositorio:

```bash
python3 -m service.main
```

Por defecto expone una API interna en `http://127.0.0.1:8765`.
Si `/srv/pyrone-nvr` existe y es escribible, lo usa como NVR; si no, cae en `service/runtime/nvr`.
También puedes forzar la ruta con `PYRONE_NVR_DIR=/srv/pyrone-nvr`.

Endpoints utiles del servicio:

- `GET /v1/models`
- `GET /v1/pipelines`
- `GET /v1/pipelines/<droneId>/frame.jpg`
- `GET /v1/pipelines/<droneId>/stream.mjpg`
- `GET /v1/recordings?droneId=<droneId>`

### Segunda etapa de clasificacion

El servicio puede usar un clasificador YOLOv8s adicional para validar detecciones
visuales de baja/media confianza antes de generar eventos, clips y focos en mapa.
Esto aplica solo a sensores `wide`, `visual`, `zoom` y `unknown`; no se aplica a
`thermal`/`IR`.

Variables principales:

```bash
PYRONE_SECOND_STAGE_ENABLED=true
PYRONE_SECOND_STAGE_MODEL_PATH=service/models/ad_phash3_early_smoke_best.pt
PYRONE_SECOND_STAGE_CONF_LOW=0.10
PYRONE_SECOND_STAGE_CONF_HIGH=0.30
PYRONE_SECOND_STAGE_CROP_SIZE=224
PYRONE_SECOND_STAGE_IMAGE_SIZE=224
```

La regla es:

- confianza menor a `CONF_LOW`: descarta.
- confianza mayor o igual a `CONF_HIGH`: acepta.
- confianza intermedia: corta un crop centrado en la deteccion y lo valida con
  el clasificador `background`/`foreground`.

Cuando esta segunda etapa esta activa, el umbral efectivo del detector baja como
maximo hasta `CONF_LOW` para no perder candidatos intermedios antes de que el
clasificador pueda revisarlos.

## Disco NVR real

El disco operativo definido para este proyecto es el Toshiba `Z5S2A0M1FW9J`.
La ruta esperada por el servicio es:

```bash
/srv/pyrone-nvr
```

Importante:

- Si el disco todavia no tiene sistema de archivos, primero hay que prepararlo y montarlo ahi.
- El servicio no formatea discos automaticamente.
- Mientras `/srv/pyrone-nvr` no exista y no sea escribible, la evidencia se seguira guardando en `service/runtime/nvr`.
