import asyncio
import random
import re
import time
import unicodedata
from enum import Enum

import httpx
from loguru import logger

from .config import (
    AI_CHUNK_MAX_CHARS,
    AI_REQUEST_TIMEOUT,
    AI_TEMPERATURE,
    DEFAULT_MAX_SINGLE_CHARS,
    GEMINI_API_KEY,
    GEMINI_CHUNK_CHARS,
    GEMINI_FALLBACK_MODEL,
    GEMINI_MAX_ATTEMPTS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_SINGLE_CHARS,
    GEMINI_MODEL,
    GEMINI_TASKS_MODEL,
    MAX_TRANSCRIPT_SIZE_SINGLE,
    OLLAMA_BASE_URL,
    OLLAMA_CONNECT_TIMEOUT,
    OLLAMA_ENABLED,
    OLLAMA_MAX_ATTEMPTS,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_RETRY_DELAY,
    OLLAMA_SAFE_PROMPT_CHARS,
    SUMMARY_PROVIDER,
    SUMMARY_TARGET_WORDS_MAX,
    SUMMARY_TARGET_WORDS_MIN,
)

ACADEMIC_LOAD_SECTION_TITLE = "📝 Carga Academica y Fechas Importantes"
NO_TASKS_MESSAGE = (
    "<p><strong>No fue posible identificar actividades o fechas de entrega de forma confiable en esta salida. "
    "Revisar transcripcion completa para confirmar.</strong></p>"
)
LONG_AUDIO_THRESHOLD_CHARS = 50000
GEMINI_LONG_AUDIO_FALLBACK_MODEL = "gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT = (
    "Eres un asistente academico hiper-detallista especializado en crear apuntes exhaustivos de clase a partir de "
    "transcripciones de audio. Tu trabajo es EXTRAER, PRESERVAR y ORGANIZAR fielmente el contenido "
    "real de la clase. ESTA PROHIBIDO hacer resúmenes genéricos o superficiales.\n\n"
    "REGLA FUNDAMENTAL: Tienes que mencionar TODOS los ejemplos específicos, anécdotas, casos de estudio, "
    "herramientas, empresas, y datos técnicos que el profesor haya mencionado en la transcripcion. "
    "Tu redacción debe ser creativa, inmersiva y rica en vocabulario, PERO tienes ESTRICTAMENTE PROHIBIDO inventar "
    "información, hechos o ejemplos que no estén explícitamente en el audio. Mantén una fidelidad absoluta al contenido original.\n"
    "Prioriza una lectura fluida, continua y académica en párrafos bien estructurados. Usa listas de forma moderada, "
    "solo para enumerar conceptos concretos o pasos."
)

TASK_EXTRACTION_PROMPT = """
Eres un analizador de texto estricto. Tu ÚNICO y EXCLUSIVO objetivo es extraer tareas, entregables, exposiciones y compromisos de la transcripción. 
ESTÁ TERMINANTEMENTE PROHIBIDO REDACTAR UN RESUMEN DE LA CLASE. No incluyas introducciones, no generes apuntes de los temas vistos, no incluyas conclusiones.

REGLAS DE FORMATO (OBLIGATORIAS):
- Tu respuesta debe ser EXCLUSIVAMENTE código HTML puro. Cero Markdown. No uses ```html ni ```.
- Inicia tu respuesta directamente con la etiqueta: <h2>📝 Carga Academica y Fechas Importantes</h2>
- Usa una lista <ul> y <li> para desglosar las actividades. Si no hay, usa <p>.

SECCION DE TAREAS Y ACCIONES:
- Busca minuciosamente talleres, trabajos, ejercicios, examenes, parciales, entregas, lecturas, proyectos o exposiciones.
- Para CADA actividad extraída incluye: 1) Tipo en negrita (ej. <strong>Exposición</strong>), 2) Instrucciones exactas, 3) Fecha exacta o "Fecha no especificada", 4) Evidencia textual corta entre comillas.
- Si tras revisar TODO no hay NADA, tu respuesta debe ser ÚNICAMENTE:
<h2>📝 Carga Academica y Fechas Importantes</h2>
<p><strong>No se identificaron tareas, exposiciones ni fechas de entrega explícitas en la transcripción.</strong></p>
""".strip()

HTML_FORMAT_RULES = """
REGLAS DE FORMATO (HTML puro, sin Markdown ni bloques de codigo):
- Usa <h1> para el titulo principal, <h2> para secciones y <h3> para subsecciones por cada bloque discutido.
- Escribe el desarrollo de los temas empleando párrafos extensos y fluidos (<p>), asegurando una lectura coherente, continua y de inmersión.
- Usa listas (<ul> y <li>) de manera MODERADA, únicamente para agrupar elementos concretos, enumerar pasos de un proceso o listar ventajas/desventajas. NO conviertas el texto en un esquema gigante de viñetas.
- Usa <table> para cronogramas o comparaciones.
- Usa <strong>, <em> para resaltar terminos clave.
- NO incluyas ```html, ``` ni estilos inline.
""".strip()

TASK_DETECTION_RULES = """
SECCION DE TAREAS Y ACCIONES (MUY IMPORTANTE):
- Tu máxima prioridad como asistente es extraer ENTREGABLES, TAREAS, EXPOSICIONES, PRESENTACIONES, parciales, quices, lecturas o "para la próxima clase". Presta atención especial a instrucciones operativas.
- Documentalas en <h2>📝 Carga Academica y Fechas Importantes</h2> usando <ul><li>.
- Para CADA actividad debes incluir OBLIGATORIAMENTE: 1) Tipo de actividad en negrita (ej. <strong>Exposición final</strong>), 2) Instrucciones exactas y contexto de qué hay que hacer, 3) Fecha o plazo exacto, 4) Una evidencia textual corta.
- No omitas actividades por falta de fecha exacta; usa "Fecha no especificada".
- Esta sección es CRÍTICA para el estudiante. Escudriña cada párrafo de la transcripción para encontrar compromisos académicos.
- Si genuinamente tras revisar todo no hay ninguna mención, escribe exactamente:
    <p><strong>No se identificaron tareas, exposiciones ni fechas de entrega explícitas en la transcripción.</strong></p>
""".strip()

EXPANDED_NOTES_RULES = """
ESTILO DE APUNTES EXPANDIDOS (NO resumen corto):
- Actua como transcriptor y editor academico: recupera la mayor cantidad posible de la clase.
- Exploralos a fondo: desarrolla el contexto, argumentos, TODOS los ejemplos concretos (nombres de empresas, tecnologías, situaciones históricas) y casos de estudio en bloques de texto continuo.
- Entrelaza las ideas con buena redacción en lugar de recurrir al facilismo de crear infinitas sublistas. El estudiante debe sentir que está leyendo un libro de texto creado de la clase, fluido y completo.
""".strip()


class AIProvider(Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    DISABLED = "disabled"


PROVIDER_RATE_LIMIT_UNTIL: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Cliente HTTP reutilizable (connection pooling — evita overhead por request)
# ---------------------------------------------------------------------------

_ai_http_client: httpx.AsyncClient | None = None


def _get_ai_http_client() -> httpx.AsyncClient:
    """Retorna el cliente httpx compartido para llamadas a APIs de IA."""
    global _ai_http_client
    if _ai_http_client is None or _ai_http_client.is_closed:
        _ai_http_client = httpx.AsyncClient(timeout=AI_REQUEST_TIMEOUT)
    return _ai_http_client


class AIServiceUnavailable(RuntimeError):
    """Indica que no se pudo completar la generacion con ningun proveedor."""


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _target_word_range(transcript: str) -> tuple[int, int]:
    transcript_words = _word_count(transcript)
    if transcript_words <= 0:
        return 500, 900

    # Calibra por bandas para evitar exigir de mas en entradas medianas,
    # manteniendo objetivos altos en transcripciones largas.
    if transcript_words <= 500:
        proportional_min = int(transcript_words * 0.50)
        proportional_max = int(transcript_words * 0.95)
    elif transcript_words <= 1800:
        proportional_min = int(transcript_words * 0.45)
        proportional_max = int(transcript_words * 0.82)
    else:
        proportional_min = int(transcript_words * 0.52)
        proportional_max = int(transcript_words * 0.86)

    max_words = min(5000, max(SUMMARY_TARGET_WORDS_MAX, proportional_max))
    min_words = max(500, min(max_words - 250, max(SUMMARY_TARGET_WORDS_MIN, proportional_min)))
    return min_words, max_words


def _build_summary_prompt(transcript: str, materia: str = "") -> str:
    min_words, max_words = _target_word_range(transcript)
    materia_ctx = (
        f"\nCONTEXTO OPCIONAL DE MATERIA: {materia}\n"
        if materia.strip()
        else ""
    )
    return f"""Transforma la transcripcion en apuntes de clase extensos y de alta fidelidad.

OBJETIVO:
- Entregar una transcripción convertida en apuntes exhaustivos, detallando CADA subtema y anécdota.
- Mantener fidelidad extrema: incluye los mismos ejemplos numéricos o de la vida real (nombres, empresas, herramientas) de la clase original.
- Nunca reduzcas una explicación larga a un solo párrafo.
- Prioriza respuestas muy largas y ultra-detalladas.

LONGITUD OBJETIVO:
- Minimo recomendado: {min_words} palabras.
- Maximo recomendado: {max_words} palabras.
- Si hace falta para cubrir bien el contenido, puedes superar el maximo recomendado.

ESTRUCTURA REQUERIDA (HTML puro):
<h1>📚 [Titulo de la clase o tema principal]</h1>
<h2>🚀 Resumen Ejecutivo</h2>
<p>[2-3 parrafos sobre lo cubierto, el enfoque de la clase y expectativas para el estudiante.]</p>
<h2>📖 Desarrollo Tematico</h2>
[Crea un Subcapitulo <h3> distinto por cada bloque y sub-tema. MENCIONA los ejemplos específicos. ]
[Desarrolla el contenido profundo en párrafos completos, extensos y conectados (<p>). Mantén la continuidad argumentativa de la clase.]
[Usa listas <ul><li> SOLO cuando sea indispensable enumerar. Evita el abuso de viñetas.]
[NO omitas anécdotas, analogías históricas, debates ni empresas mencionadas.]
<h2>🛠 Herramientas y Recursos</h2>
[Solo si se mencionaron recursos.]
<h2>🧠 Preguntas de Repaso</h2>
[Generar al menos 5 preguntas de formato detallado derivadas de lo visto, para poner a prueba el conocimiento.]

REGLAS DE CALIDAD:
- Desarrolla ideas en parrafos completos y conectados.
- Incluye ejemplos, decisiones, argumentos y matices presentes en la transcripcion.
- En "Preguntas de Repaso" prioriza recall: incluye preguntas derivadas del material.
- Para cada pregunta agrega un formato detallado.
- Evita frases meta sobre tus limitaciones como modelo.

{HTML_FORMAT_RULES}
{EXPANDED_NOTES_RULES}
{materia_ctx}
TRANSCRIPCION:
{transcript}
"""


def _build_chunk_prompt(chunk: str, section_num: int, total_sections: int, materia: str = "") -> str:
    materia_ctx = f"\nCONTEXTO OPCIONAL DE MATERIA: {materia}\n" if materia.strip() else ""
    return f"""Analiza este fragmento de una transcripcion y extrae apuntes parciales de alta calidad.

OBJETIVO:
- Sintetiza con profundidad, en formato HTML.
- Conserva hechos, ejemplos y explicaciones presentes en el texto.
- No inventes contenido.

FORMATO:
- Empieza con <h3>[Tema detectado]</h3> en cada idea fuerte.
- Escribe la descripción de contextos, matices y ejemplos de forma narrativa y continua dentro de párrafos <p>.
- Usa listas <ul> y <li> de manera restringida, solo si hay que hacer enumeraciones obvias.
- El texto debe fluir lógicamente como un libro de apuntes.

{HTML_FORMAT_RULES}
{EXPANDED_NOTES_RULES}
{materia_ctx}
FRAGMENTO {section_num}/{total_sections}:
{chunk}
"""


def _build_unification_prompt(combined_summaries: str, transcript: str, materia: str = "") -> str:
    materia_ctx = f"\nCONTEXTO OPCIONAL DE MATERIA: {materia}\n" if materia.strip() else ""
    return f"""Eres un Redactor Academico Maestro ("El Director"). Recibes borradores detallados, fragmentados a partir de una clase inmensa.
Tu trabajo es UNIFICAR estos apuntes aislados en un solo documento maestro, cohesivo, inmaculado y ultra-detallado, con formato HTML puro.

OBJETIVO:
- Reconstruir la clase entera con fluidez perfecta.
- MANTENER Y FUNDIR TODO EL NIVEL DE DETALLE de los borradores en la seccion "Desarrollo Tematico". No omitas contenido valioso o ejemplos.
- Extrapolar y generar las secciones obligatorias correctamente (Resumen Ejecutivo, Herramientas, Preguntas).
- El documento final NO debe parecer un compilado. Debe lucir como una obra maestra escrita de un solo tiron.

ESTRUCTURA REQUERIDA (HTML puro):
<h1>📚 [Titulo General que englobe toda la clase]</h1>

<h2>🚀 Resumen Ejecutivo</h2>
<p>[Sintesis general estructurada de todo el arco de conocimiento abarcado]</p>

<h2>📖 Desarrollo Tematico</h2>
[Agrupa o enlaza los temas dispersos en los borradores empleando inteligentemente etiquetas <h3>. Desarrolla las ideas de forma inmersiva y continua con múltiples párrafos <p> bien conectados, conservando anécdotas, nombres y citas. Modera el uso de viñetas para lo estrictamente numerativo.]

<h2>🛠️ Herramientas y Recursos</h2>
[Extrae todas las plataformas o recursos de los borradores en una lista.]

<h2>🧠 Preguntas de Repaso</h2>
[Construye 5-8 preguntas estrategicas cruzando lo más vital enseñado.]

REGLAS DE CALIDAD:
- JAMAS metas comentarios meta como "En el borrador 3 se indico..." Escribe directamente la idea.
- Desarrolla las ideas, evita escribir un solo parrafo largo por tema, dividelo logicamente.

{HTML_FORMAT_RULES}
{EXPANDED_NOTES_RULES}
{materia_ctx}
---
BORRADORES PARCIALES A UNIFICAR:
{combined_summaries}
"""


def _split_transcript(transcript: str, max_chars: int) -> list[str]:
    words = transcript.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        token_len = len(word) + 1
        if length + token_len > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            length = token_len
        else:
            current.append(word)
            length += token_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _parse_retry_after(value: str) -> float:
    try:
        return float(value) + 1.0
    except (TypeError, ValueError):
        return 0.0


def _transient_retry_delay(attempt: int, base: float = 1.2, cap: float = 8.0) -> float:
    # Backoff exponencial con jitter para errores transitorios (5xx/timeouts).
    exp = min(cap, base * (2 ** max(0, attempt)))
    return exp + random.uniform(0.0, 0.8)


def _truncate_prompt_head_tail(prompt_text: str, max_chars: int) -> str:
    marker = "\n\n...[TRUNCADO_INTERMEDIO]...\n\n"
    head_chars = int(max_chars * 0.55)
    tail_chars = max(0, max_chars - head_chars - len(marker))
    return f"{prompt_text[:head_chars]}{marker}{prompt_text[-tail_chars:] if tail_chars else ''}"


def _split_prompt_transcript(prompt_text: str) -> tuple[str, str] | None:
    marker = "TRANSCRIPCION:\n"
    idx = prompt_text.find(marker)
    if idx < 0:
        return None
    prefix = prompt_text[: idx + len(marker)]
    transcript = prompt_text[idx + len(marker):].strip()
    if not transcript:
        return None
    return prefix, transcript


def _is_rate_limited(provider: AIProvider) -> bool:
    until = PROVIDER_RATE_LIMIT_UNTIL.get(provider.value)
    if not until:
        return False
    if time.time() < until:
        return True
    PROVIDER_RATE_LIMIT_UNTIL.pop(provider.value, None)
    return False


def _remember_rate_limit(provider: AIProvider, wait_seconds: float) -> None:
    PROVIDER_RATE_LIMIT_UNTIL[provider.value] = time.time() + max(wait_seconds, 0.0)


def _detect_primary_provider() -> AIProvider:
    if SUMMARY_PROVIDER == "disabled":
        return AIProvider.DISABLED
    if SUMMARY_PROVIDER == "gemini" and GEMINI_API_KEY:
        return AIProvider.GEMINI
    if GEMINI_API_KEY:
        return AIProvider.GEMINI
    return AIProvider.DISABLED


def _build_provider_order(primary: AIProvider) -> list[AIProvider]:
    order: list[AIProvider] = []

    def _add_if_available(provider: AIProvider) -> None:
        if provider in order:
            return
        if provider == AIProvider.GEMINI and not GEMINI_API_KEY:
            return
        if provider == AIProvider.OLLAMA and not OLLAMA_ENABLED:
            return
        if provider != AIProvider.DISABLED:
            order.append(provider)

    _add_if_available(primary)
    _add_if_available(AIProvider.GEMINI)
    _add_if_available(AIProvider.OLLAMA)
    return order


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _model_candidates(provider: AIProvider) -> list[str]:
    if provider == AIProvider.GEMINI:
        candidates = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]

    elif provider == AIProvider.OLLAMA:
        candidates = [OLLAMA_MODEL]
    else:
        candidates = []
    return [model for model in dict.fromkeys(candidates) if model]



async def _call_gemini(
    prompt: str,
    override_model: str = None,
    secondary_model: str = None,
) -> str | None:
    if not GEMINI_API_KEY:
        return None

    if override_model:
        candidates = [override_model]
        if secondary_model:
            candidates.append(secondary_model)
        candidates.extend(_model_candidates(AIProvider.GEMINI))
        candidates = [model for model in dict.fromkeys(candidates) if model]
    else:
        candidates = _model_candidates(AIProvider.GEMINI)
    for model in candidates:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={GEMINI_API_KEY}"
        )

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": AI_TEMPERATURE,
                "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            },
        }

        for attempt in range(max(1, GEMINI_MAX_ATTEMPTS)):
            try:
                http_client = _get_ai_http_client()
                resp = await http_client.post(url, json=payload, headers={"Content-Type": "application/json"})

                if resp.status_code == 429:
                    wait_seconds = _parse_retry_after(resp.headers.get("retry-after", ""))
                    if wait_seconds > 0:
                        _remember_rate_limit(AIProvider.GEMINI, wait_seconds)
                    logger.warning(f"Rate limit [gemini] modelo={model}; intento fallback")
                    break

                if resp.status_code == 404:
                    logger.warning(f"Gemini devolvio 404 en modelo={model}; probando siguiente")
                    break

                if resp.status_code in {500, 502, 503, 504}:
                    if attempt + 1 < max(1, GEMINI_MAX_ATTEMPTS):
                        wait_seconds = _transient_retry_delay(attempt)
                        logger.warning(
                            f"Gemini devolvio {resp.status_code} en modelo={model}; "
                            f"reintento {attempt + 2}/{max(1, GEMINI_MAX_ATTEMPTS)} en {wait_seconds:.1f}s"
                        )
                        await asyncio.sleep(wait_seconds)
                        continue
                    logger.warning(f"Gemini devolvio {resp.status_code} en modelo={model}; probando siguiente")
                    break

                resp.raise_for_status()
                data = resp.json()
                candidates_resp = data.get("candidates", [])
                if not candidates_resp:
                    return None

                selected = candidates_resp[0]
                parts = selected.get("content", {}).get("parts", [])
                text = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
                finish_reason = (selected.get("finishReason") or "").upper()

                cont_attempts = 0
                MAX_CONTINUATIONS = 3
                while text and finish_reason in {"MAX_TOKENS", "LENGTH"} and cont_attempts < MAX_CONTINUATIONS:
                    cont_attempts += 1
                    logger.info(f"Gemini: completando texto cortado por limite (intento {cont_attempts}/{MAX_CONTINUATIONS})...")
                    continuation_payload = {
                        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                        "contents": [
                            {"role": "user", "parts": [{"text": prompt}]},
                            {"role": "model", "parts": [{"text": text[-15000:]}]}, # Solo enviar los ultimos 15000 chars si es muy largo
                            {"role": "user", "parts": [{"text": "Continua exactamente desde donde terminaste en la respuesta anterior, manteniendo el formato HTML sin reiniciar secciones ni repetir texto."}]}
                        ],
                        "generationConfig": {
                            "temperature": AI_TEMPERATURE,
                            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
                        },
                    }
                    cont_client = _get_ai_http_client()
                    cont_resp = await cont_client.post(
                        url,
                        json=continuation_payload,
                        headers={"Content-Type": "application/json"},
                    )
                    if cont_resp.status_code < 400:
                        cont_data = cont_resp.json()
                        cont_candidates = cont_data.get("candidates", [])
                        if cont_candidates:
                            selected_cont = cont_candidates[0]
                            cont_parts = selected_cont.get("content", {}).get("parts", [])
                            cont_text = "\n".join(
                                part.get("text", "") for part in cont_parts if part.get("text")
                            ).strip()
                            finish_reason = (selected_cont.get("finishReason") or "").upper()
                            if cont_text:
                                text = f"{text}\n{cont_text}".strip()
                            else:
                                break
                        else:
                            break
                    else:
                        break

                if text:
                    return text
                return None
            except httpx.TimeoutException:
                if attempt + 1 < max(1, GEMINI_MAX_ATTEMPTS):
                    wait_seconds = _transient_retry_delay(attempt)
                    logger.warning(
                        f"Timeout [gemini] modelo={model} intento {attempt + 1}/{max(1, GEMINI_MAX_ATTEMPTS)}; "
                        f"reintentando en {wait_seconds:.1f}s"
                    )
                    await asyncio.sleep(wait_seconds)
                    continue
                logger.warning(
                    f"Timeout [gemini] modelo={model} intento {attempt + 1}/{max(1, GEMINI_MAX_ATTEMPTS)}"
                )
                continue
            except httpx.HTTPStatusError as exc:
                logger.error(f"HTTP {exc.response.status_code} [gemini]: {exc.response.text[:200]}")
                if 400 <= exc.response.status_code < 500:
                    return None
                break
            except Exception as exc:
                logger.error(f"Error [gemini] modelo={model}: {type(exc).__name__}: {exc}")
                break

    return None

async def _call_ollama(prompt: str) -> str | None:
    base_url = OLLAMA_BASE_URL.rstrip("/")
    prompt_text = prompt

    timeout = httpx.Timeout(
        connect=OLLAMA_CONNECT_TIMEOUT,
        read=OLLAMA_REQUEST_TIMEOUT,
        write=max(30.0, OLLAMA_CONNECT_TIMEOUT),
        pool=OLLAMA_CONNECT_TIMEOUT,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            tags_resp = await http_client.get(f"{base_url}/api/tags")
            tags_resp.raise_for_status()
            tags_data = tags_resp.json() if tags_resp.content else {}
            models = tags_data.get("models", []) if isinstance(tags_data, dict) else []
            available = [m.get("name", "") for m in models if isinstance(m, dict) and m.get("name")]

            model = OLLAMA_MODEL
            if available and model not in available:
                logger.warning(f"Ollama: modelo '{model}' no disponible. Usando '{available[0]}'")
                model = available[0]

            async def _send_ollama(user_prompt: str) -> str | None:
                for attempt in range(max(1, OLLAMA_MAX_ATTEMPTS)):
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {
                            "temperature": AI_TEMPERATURE,
                            "num_predict": OLLAMA_NUM_PREDICT,
                        },
                    }

                    try:
                        resp = await http_client.post(f"{base_url}/api/chat", json=payload)
                        resp.raise_for_status()
                        data = resp.json() if resp.content else {}
                        content = (data.get("message", {}).get("content") or "").strip()
                        if content:
                            return content
                    except httpx.TimeoutException:
                        logger.warning(
                            f"Timeout [ollama] intento {attempt + 1}/{max(1, OLLAMA_MAX_ATTEMPTS)}"
                        )
                        if attempt + 1 < max(1, OLLAMA_MAX_ATTEMPTS):
                            await asyncio.sleep(OLLAMA_RETRY_DELAY)
                            continue
                        return None
                    except httpx.HTTPStatusError as exc:
                        logger.error(f"HTTP {exc.response.status_code} [ollama]: {exc.response.text[:200]}")
                        if 400 <= exc.response.status_code < 500:
                            return None
                        if attempt + 1 < max(1, OLLAMA_MAX_ATTEMPTS):
                            await asyncio.sleep(OLLAMA_RETRY_DELAY)
                            continue
                        return None
                    except Exception as exc:
                        logger.error(f"Error [ollama] intento {attempt + 1}: {type(exc).__name__}: {exc}")
                        if attempt + 1 < max(1, OLLAMA_MAX_ATTEMPTS):
                            await asyncio.sleep(OLLAMA_RETRY_DELAY)
                            continue
                        return None
                return None

            if len(prompt_text) > OLLAMA_SAFE_PROMPT_CHARS:
                split_payload = _split_prompt_transcript(prompt_text)

                if split_payload is not None:
                    prompt_prefix, transcript = split_payload
                    target_chunks = min(6, max(2, (len(transcript) + 15999) // 16000))
                    chunk_chars = max(5000, len(transcript) // target_chunks)
                    chunks = _split_transcript(transcript, max_chars=chunk_chars)

                    logger.warning(
                        f"Prompt grande para Ollama ({len(prompt_text):,} chars). "
                        f"Aplicando map-reduce con {len(chunks)} fragmentos."
                    )

                    partials: list[str] = []
                    for idx, chunk in enumerate(chunks):
                        chunk_prompt = (
                            f"{prompt_prefix}\n"
                            f"[FRAGMENTO {idx + 1}/{len(chunks)}]\n"
                            f"{chunk}\n\n"
                            "INSTRUCCION ADICIONAL: Extrae apuntes parciales fieles en HTML, "
                            "priorizando actividades, entregables y fechas. No inventes informacion."
                        )
                        if len(chunk_prompt) > OLLAMA_SAFE_PROMPT_CHARS:
                            chunk_prompt = _truncate_prompt_head_tail(chunk_prompt, OLLAMA_SAFE_PROMPT_CHARS)

                        partial = await _send_ollama(chunk_prompt)
                        if partial:
                            partials.append(clean_html(partial))

                    if partials:
                        combined = "\n\n".join(
                            f"<!-- Fragmento {i + 1} -->\n{item}" for i, item in enumerate(partials)
                        )
                        unify_prompt = _build_unification_prompt(combined, transcript="", materia="")
                        if len(unify_prompt) > OLLAMA_SAFE_PROMPT_CHARS:
                            unify_prompt = _truncate_prompt_head_tail(unify_prompt, OLLAMA_SAFE_PROMPT_CHARS)
                        unified = await _send_ollama(unify_prompt)
                        if unified:
                            return unified
                        # Si falla unificacion, retornar al menos el consolidado parcial.
                        return combined

                logger.warning(
                    f"Prompt grande para Ollama ({len(prompt_text):,} chars). "
                    f"Sin estructura de transcripcion, recortando a {OLLAMA_SAFE_PROMPT_CHARS:,} chars."
                )
                prompt_text = _truncate_prompt_head_tail(prompt_text, OLLAMA_SAFE_PROMPT_CHARS)

            return await _send_ollama(prompt_text)
    except Exception as exc:
        logger.error(f"Error fatal [ollama]: {type(exc).__name__}: {exc}")
        return None



async def _call_provider(
    provider: AIProvider,
    prompt: str,
    override_model: str = None,
    secondary_model: str = None,
) -> str | None:
    if provider == AIProvider.GEMINI:
        return await _call_gemini(prompt, override_model, secondary_model)

    if provider == AIProvider.OLLAMA:
        return await _call_ollama(prompt)
    return None


async def _call_ai(
    prompt: str,
    override_model: str = None,
    gemini_secondary_model: str = None,
) -> str:
    primary = _detect_primary_provider()
    if primary == AIProvider.DISABLED:
        raise AIServiceUnavailable("No hay proveedores de IA configurados")

    for provider in _build_provider_order(primary):
        if _is_rate_limited(provider):
            continue

        result = await _call_provider(provider, prompt, override_model, gemini_secondary_model)
        if result:
            logger.info(f"Respuesta IA generada con proveedor={provider.value}")
            return result

    raise AIServiceUnavailable("Ningun proveedor pudo generar la respuesta")


async def _summarize_chunks(transcript: str, materia: str, progress_callback=None) -> str:
    primary = _detect_primary_provider()
    chunk_chars = GEMINI_CHUNK_CHARS if primary == AIProvider.GEMINI else AI_CHUNK_MAX_CHARS
    inter_chunk_delay = 0.1


    # Si es muy corto, no vale la pena unificar, hazlo en un solo llamado
    if len(transcript) <= chunk_chars:
        logger.info(f"Resumen en bloque unico (corto) | {len(transcript):,} chars | chunk={chunk_chars}")
        prompt = _build_summary_prompt(transcript, materia=materia)
        result = await _call_ai(prompt)
        return clean_html(result)

    chunks = _split_transcript(transcript, max_chars=chunk_chars)
    total = len(chunks)
    if total == 0:
        raise AIServiceUnavailable("No hay contenido para resumir")

    logger.info(f"Resumen en bloques (Modo Obrero) | {len(transcript):,} chars | {total} secciones | chunk={chunk_chars}")

    partials: list[str] = []
    for idx, chunk in enumerate(chunks):
        section = idx + 1
        if progress_callback:
            await progress_callback((idx / max(total, 1)) * 70, f"Procesando seccion {section} de {total}")

        prompt = _build_chunk_prompt(chunk, section, total, materia=materia)
        partial = await _call_ai(prompt) # Usa el modelo economico (Gemini 3.1 Lite) por defecto
        partials.append(clean_html(partial))

        if idx < total - 1:
            await asyncio.sleep(inter_chunk_delay)

    if progress_callback:
        await progress_callback(85, "Unificando con Modelo Maestro...")

    combined = "\n\n".join(f"<!-- Seccion {i + 1} -->\n{item}" for i, item in enumerate(partials))
    
    # REDUCE: Usamos el modelo principal (configurado en GEMINI_MODEL) para unificar
    unify_prompt = _build_unification_prompt(combined, transcript=transcript, materia=materia)
    override_unifier = GEMINI_MODEL if primary == AIProvider.GEMINI else None
    
    if override_unifier:
        logger.info(f"Ruta chunked - Unificando con modelo maestro: {override_unifier}")

    try:
        # Request con Try-Except
        unified = await _call_ai(unify_prompt, override_model=override_unifier)
        final_doc = clean_html(unified)
    except AIServiceUnavailable:
        logger.warning("Unificacion con modelo maestro fallo, reintentando...")
        await asyncio.sleep(2.0)
        unified = await _call_ai(unify_prompt, override_model=override_unifier)
        final_doc = clean_html(unified)

    return final_doc


def _fallback_summary(transcript: str) -> str:
    words = _word_count(transcript)
    minutes = (words / 150) if words else 0.0

    return f"""<h1>Resumen de la sesion</h1>
<h2>Resumen Ejecutivo</h2>
<p>No fue posible generar el resumen con IA en este intento.</p>
<h2>Desarrollo Tematico</h2>
<p>No se pudo desarrollar el contenido tematico automaticamente.</p>
<h2>Herramientas y Recursos</h2>
<p>No se detectaron herramientas o recursos en esta salida minima.</p>
<h2>{ACADEMIC_LOAD_SECTION_TITLE}</h2>
{NO_TASKS_MESSAGE}
<h2>Preguntas de repaso</h2>
<p>Revisa la transcripcion completa para reconstruir los puntos clave.</p>
<h2>Informacion disponible</h2>
<ul>
  <li><strong>Palabras en transcripcion:</strong> {words:,}</li>
  <li><strong>Duracion estimada:</strong> {minutes:.0f} minutos</li>
</ul>
<h2>Proximos pasos sugeridos</h2>
<p>Reintenta el proceso o revisa conectividad y configuracion del proveedor de IA.</p>
"""


async def summarize(transcript: str, materia: str, progress_callback=None) -> str:
    started_at = time.perf_counter()
    if not transcript or len(transcript.strip()) < 50:
        fallback_html = _fallback_summary(transcript)
        logger.info(
            "Resumen metrics | status=fallback | reason=transcript-corto | "
            f"in_chars={len(transcript or '')} | out_chars={len(fallback_html)} | "
            f"out_words={_word_count(fallback_html)} | duration_ms={(time.perf_counter() - started_at) * 1000:.0f}"
        )
        return fallback_html

    primary = _detect_primary_provider()
    if primary == AIProvider.DISABLED:
        fallback_html = _fallback_summary(transcript)
        logger.info(
            "Resumen metrics | status=fallback | reason=sin-proveedor | "
            f"in_chars={len(transcript)} | out_chars={len(fallback_html)} | "
            f"out_words={_word_count(fallback_html)} | duration_ms={(time.perf_counter() - started_at) * 1000:.0f}"
        )
        return fallback_html

    model = (
        GEMINI_MODEL
        if primary == AIProvider.GEMINI

        else OLLAMA_MODEL
    )

    max_single = GEMINI_MAX_SINGLE_CHARS if primary == AIProvider.GEMINI else DEFAULT_MAX_SINGLE_CHARS
    if MAX_TRANSCRIPT_SIZE_SINGLE > 0:
        max_single = max(20000, MAX_TRANSCRIPT_SIZE_SINGLE)

    route = "chunked" if len(transcript) > max_single else "single"
    override_model = GEMINI_MODEL if route == "single" and primary == AIProvider.GEMINI else None
    gemini_secondary_model = None
    if (
        route == "single"
        and primary == AIProvider.GEMINI
        and len(transcript) >= LONG_AUDIO_THRESHOLD_CHARS
    ):
        gemini_secondary_model = GEMINI_LONG_AUDIO_FALLBACK_MODEL
    
    actual_model = override_model if override_model else model
    logger.info(f"Generando resumen | {len(transcript):,} chars | proveedor={primary.value} | modelo={actual_model}")

    if progress_callback:
        await progress_callback(20, "Analizando transcripción...")

    async def _do_main_summary():
        if len(transcript) > max_single:
            return await _summarize_chunks(transcript, materia, progress_callback=progress_callback)
        else:
            if override_model:
                logger.info(f"Ruta single - Usando modelo mas capaz y limitante para textos cortos: {override_model}")
                if gemini_secondary_model:
                    logger.info(
                        "Ruta single larga - Si falla 2.5 tras reintentos, "
                        f"probando fallback Gemini: {gemini_secondary_model}"
                    )
            prompt = _build_summary_prompt(transcript, materia=materia)
            return clean_html(await _call_ai(prompt, override_model, gemini_secondary_model))

    async def _do_task_extraction():
        if primary != AIProvider.GEMINI:
            return ""
        prompt = f"{TASK_EXTRACTION_PROMPT}\n\nTRANSCRIPCION:\n{transcript}"
        try:
            logger.info(f"Ruta tareas - Extrayendo tareas con modelo especializado: {GEMINI_TASKS_MODEL}")
            result = await _call_ai(prompt, override_model=GEMINI_TASKS_MODEL)
            
            # Limpieza estricta: buscar la etiqueta <h2> para ignorar texto conversacional previo
            result_clean = result.strip()
            for marker in ("```html", "```"):
                result_clean = result_clean.replace(marker, "")
            
            h2_index = result_clean.find("<h2")
            if h2_index != -1:
                result_clean = result_clean[h2_index:]
            
            # Asegurar que siempre devuelva HTML válido y no caiga en el fallback de clean_html
            if not result_clean.startswith("<"):
                result_clean = f"<h2>📝 Carga Academica y Fechas Importantes</h2>\n<p>{result_clean}</p>"
                
            return result_clean.strip()
        except Exception as exc:
            logger.warning(f"Error en extraccion de tareas: {exc}")
            return "<h2>📝 Carga Academica y Fechas Importantes</h2>\n<p><strong>No fue posible analizar la transcripción en busca de tareas debido a un error técnico.</strong></p>"

    # Lanzamos ambos en paralelo
    summary_task = asyncio.create_task(_do_main_summary())
    tasks_task = asyncio.create_task(_do_task_extraction())

    summary_html, tasks_html = await asyncio.gather(summary_task, tasks_task)

    # Verificacion basica (solo si devuelven vacio completo, que es falla catastrofica)
    if not summary_html or not summary_html.strip() or len(summary_html) < 50:
        logger.warning("Resumen IA devolvio salida vacia. Se usa fallback")
        fallback_html = _fallback_summary(transcript)
        logger.info(
            "Resumen metrics | status=fallback | reason=salida_vacia | "
            f"provider={primary.value} | route={route} | "
            f"in_chars={len(transcript)} | out_chars={len(fallback_html)} | "
            f"duration_ms={(time.perf_counter() - started_at) * 1000:.0f}"
        )
        return fallback_html

    final_html = summary_html
    if tasks_html:
        final_html += f"\n\n{tasks_html}"

    logger.info(
        "Resumen metrics | status=ok | "
        f"provider={primary.value} | route={route} | "
        f"in_chars={len(transcript)} | out_chars={len(final_html)} | out_words={_word_count(final_html)} | "
        f"duration_ms={(time.perf_counter() - started_at) * 1000:.0f}"
    )

    return final_html


def clean_html(raw: str) -> str:
    text = raw.strip()
    for marker in ("```html", "```"):
        text = text.replace(marker, "")
    text = text.strip()
    if text and not text.startswith("<"):
        text = f"<h1>Resumen de la sesion</h1>\n<p>{text}</p>"
    return text.strip()

