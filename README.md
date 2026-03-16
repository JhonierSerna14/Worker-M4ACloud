# M4A Worker

Worker local para procesar jobs de transcripcion y resumen de clases.
Se conecta al backend por HTTP polling, descarga audio o transcripciones previas,
genera el resultado y reporta progreso durante todo el proceso.

## Flujo

1. Consulta `GET /api/v1/worker/jobs/next` cada `POLL_INTERVAL_SECONDS` segundos.
2. Reclama el job con `POST /api/v1/worker/jobs/{id}/claim`.
3. Si el job tiene audio, lo descarga y lo transcribe con `faster-whisper`.
4. Si el job es reproceso, descarga la transcripcion existente.
5. Genera un resumen HTML con Gemini o Groq.
6. Entrega el resultado con `POST /api/v1/worker/jobs/{id}/complete`.
7. Si algo falla, reporta `POST /api/v1/worker/jobs/{id}/fail`.

## Setup rapido

### Windows PowerShell

```powershell
cd Worker
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edita .env con tus credenciales reales
python worker.py
```

### Linux / macOS

```bash
cd Worker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edita .env con tus credenciales reales
python worker.py
```

## Dependencias

`requirements.txt` dependencias directas del codigo:

- `httpx` para comunicacion con el backend y proveedores de IA.
- `python-dotenv` para cargar configuracion desde `.env`.
- `loguru` para logs.
- `faster-whisper` para transcripcion.
- `ctranslate2` porque el codigo lo importa directamente para deteccion de CUDA.
- `requests` porque `faster-whisper` lo necesita al importar en este entorno.
- `Pillow` y `pystray` para mostrar el icono en la bandeja de Windows.

## Variables de entorno activas

### Requeridas

| Variable | Descripcion |
|---|---|
| `BACKEND_URL` | URL base del backend, sin `/api` y sin slash final. Ejemplo: `https://m4a.onrender.com` |
| `WORKER_SECRET_KEY` | Clave compartida que el backend valida en el header `X-Worker-Key` |

### Worker y polling

| Variable | Default | Descripcion |
|---|---:|---|
| `POLL_INTERVAL_SECONDS` | `10` | Intervalo entre consultas al backend |
| `LOG_LEVEL` | `INFO` | Nivel de logging para consola |

### Whisper

| Variable | Default | Descripcion |
|---|---:|---|
| `WHISPER_MODEL_SIZE` | `medium` | Modelo a cargar |
| `WHISPER_DEVICE` | `auto` | `auto`, `cuda` o `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8_float16` | Tipo de computo usado por `faster-whisper` |
| `WHISPER_LANGUAGE` | `es` | Idioma esperado de la transcripcion |
| `WHISPER_BEAM_SIZE` | `5` | Beam size |
| `WHISPER_BEST_OF` | `5` | Best-of sampling |
| `WHISPER_TEMPERATURE` | `0.0` | Temperatura |
| `WHISPER_COMPRESSION_RATIO` | `2.0` | Umbral de compresion |
| `WHISPER_NO_SPEECH_THRESHOLD` | `0.7` | Umbral para segmentos sin voz |
| `WHISPER_REPETITION_PENALTY` | `1.1` | Penalizacion por repeticiones |
| `VAD_FILTER_ENABLED` | `true` | Activa VAD |
| `VAD_MIN_SILENCE_DURATION_MS` | `250` | Silencio minimo antes de cortar un segmento |
| `VAD_SPEECH_PAD_MS` | `400` | Padding de voz alrededor de segmentos |

Nota: el codigo mantiene compatibilidad con `VAD_MIN_SPEECH_DURATION_MS`, pero el nombre correcto para esta configuracion es `VAD_MIN_SILENCE_DURATION_MS`.

### IA

| Variable | Default | Descripcion |
|---|---:|---|
| `SUMMARY_PROVIDER` | `gemini` | `gemini`, `groq` o `disabled` |
| `GEMINI_API_KEY` | vacío | API key para Gemini |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Modelo Gemini |
| `GROQ_API_KEY` | vacío | API key para Groq |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Modelo Groq |
| `GROQ_REQUEST_DELAY` | `2.5` | Delay entre llamadas a Groq en procesamiento por bloques |
| `AI_TEMPERATURE` | `0.1` | Temperatura de generacion |
| `AI_REQUEST_TIMEOUT` | `120` | Timeout de peticiones a IA |
| `MAX_TRANSCRIPT_SIZE_SINGLE` | `15000` | Maximo de caracteres antes de dividir en bloques |
| `AI_CHUNK_MAX_CHARS` | `12000` | Tamano maximo de bloque para resumen general |
| `GROQ_CHUNK_MAX_CHARS` | `8000` | Tamano maximo de bloque cuando Groq es proveedor primario |
| `GROQ_SAFE_PROMPT_CHARS` | `10000` | Limite de seguridad para evitar errores 413 en Groq |

Si `SUMMARY_PROVIDER=disabled` o no hay API keys validas, el worker genera un fallback HTML minimo en lugar del resumen enriquecido.

## Ejecucion

```bash
python worker.py
```

Al iniciar correctamente, el worker valida conectividad con el backend, carga Whisper y entra en polling continuo.

## Bandeja de Windows

En Windows puedes lanzar el worker con icono de bandeja usando:

```powershell
python worker_tray.py
```

El icono muestra:

- Un anillo de progreso del job actual.
- Un numero interno con la cantidad de jobs pendientes por procesar.
- Estado visual distinto para conexion, procesamiento y error.
- El worker escribe logs en `worker.log` dentro de la carpeta del proyecto.

El conteo pendiente usa el dato que entregue el backend si existe en `GET /api/v1/worker/status` o en el payload del job. Si el backend no expone ese conteo, el icono seguira mostrando progreso por job y un conteo local minimo.

## Inicio automatico en Windows

Para registrar el worker en el inicio de sesion de Windows:

```powershell
python install_windows_startup.py
```

Ese script crea un lanzador `.vbs` en la carpeta Startup del usuario actual y arranca `worker_tray.py` con `pythonw.exe`, sin consola visible.
