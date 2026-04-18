import asyncio
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
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODEL,
    GEMINI_MAX_ATTEMPTS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
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

SYSTEM_PROMPT = (
    "Eres un asistente academico hiper-detallista especializado en crear apuntes exhaustivos de clase a partir de "
    "transcripciones de audio. Tu trabajo es EXTRAER, PRESERVAR y ORGANIZAR fielmente el contenido "
    "real de la clase. ESTA PROHIBIDO hacer resúmenes genéricos o superficiales.\n\n"
    "REGLA FUNDAMENTAL: Tienes que mencionar TODOS los ejemplos específicos, anécdotas, casos de estudio, "
    "herramientas, empresas, y datos técnicos que el profesor haya mencionado en la transcripcion.\n"
    "Prioriza una profundidad extrema, estructurando paso a paso mediante listas largas y detalladas. "
)

HTML_FORMAT_RULES = """
REGLAS DE FORMATO (HTML puro, sin Markdown ni bloques de codigo):
- Usa <h1> para el titulo principal, <h2> para secciones y <h3> para subsecciones por cada bloque discutido.
- OBLIGATORIO: Usa listas extensas (<ul> y <li>) para desglosar TODO el detalle, explicaciones y ejemplos puntuales. Protege la granularidad de la información.
- Usa parrafos explicativos (<p>) para introducir conceptos o historias, y luego desglosa los hechos con viñetas.
- Usa <table> para cronogramas o comparaciones.
- Usa <strong>, <em> para resaltar terminos clave.
- NO incluyas ```html, ``` ni estilos inline.
""".strip()

TASK_DETECTION_RULES = """
SECCION DE TAREAS Y ACCIONES (MUY IMPORTANTE):
- Busca minuciosamente menciones a talleres, trabajos, ejercicios, tareas, examenes, parciales, quices, entregas, "para la proxima clase", lecturas, proyectos o "pitch".
- Si hay menciones a evaluaciones, proyectos o sustentaciones, plásmalo con alto detalle en <h2>📝 Carga Academica y Fechas Importantes</h2>.
- Si genuinamente no hay menciones, escribe:
  <p><strong>No se mencionaron tareas, examenes ni fechas de entrega en esta clase.</strong></p>
""".strip()

EXPANDED_NOTES_RULES = """
ESTILO DE APUNTES EXPANDIDOS (NO resumen corto):
- Actua como transcriptor y editor academico: recupera la mayor cantidad posible de la clase.
- Exploralos a fondo: desarrolla el contexto, argumentos, TODOS los ejemplos concretos (nombres de empresas, tecnologías, situaciones históricas) y casos de estudio.
- Enumera fuertemente con <ul> y <li> para desgranar características o componentes, garantizando fidelidad absoluta.
""".strip()


class AIProvider(Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    DISABLED = "disabled"


PROVIDER_RATE_LIMIT_UNTIL: dict[str, float] = {}


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
[Usa OBLIGATORIAMENTE múltiples listas y viñetas detalladas <ul> <li> largas para capturar la esencia completa.]
[NO omitas anécdotas, analogías históricas ni empresas mencionadas.]
<h2>🛠 Herramientas y Recursos</h2>
[Solo si se mencionaron recursos.]
<h2>📝 Carga Academica y Fechas Importantes</h2>
[Instrucciones y plazos detallados a los que el estudiante debe prestar atencion.]
<h2>🧠 Preguntas de Repaso</h2>
[Generar al menos 5 preguntas de formato detallado derivadas de lo visto, para poner a prueba el conocimiento.]

REGLAS DE CALIDAD:
- Desarrolla ideas en parrafos completos y conectados.
- Incluye ejemplos, decisiones, argumentos y matices presentes en la transcripcion.
- En "Carga Academica y Fechas Importantes" incluye tareas, plazos y actividades; si no existen, indicarlo explicitamente.
- Evita frases meta sobre tus limitaciones como modelo.

{HTML_FORMAT_RULES}
{TASK_DETECTION_RULES}
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
- Empuja la descripcion de contextos, matices y ejemplos con varias sub-viñetas <ul> y <li>.
- Usa varios <p> amplios solamente para conectar viñetas.
- Identifica desde ya insumos para la obligatoria seccion final <h2>📝 Carga Academica y Fechas Importantes</h2>.

{HTML_FORMAT_RULES}
{TASK_DETECTION_RULES}
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
- Extrapolar y generar las secciones obligatorias correctamente (Resumen Ejecutivo, Herramientas, Tareas, Preguntas).
- El documento final NO debe parecer un compilado. Debe lucir como una obra maestra escrita de un solo tiron.

ESTRUCTURA REQUERIDA (HTML puro):
<h1>📚 [Titulo General que englobe toda la clase]</h1>

<h2>🚀 Resumen Ejecutivo</h2>
<p>[Sintesis general estructurada de todo el arco de conocimiento abarcado]</p>

<h2>📖 Desarrollo Tematico</h2>
[Agrupa o enlaza los temas dispersos en los borradores empleando inteligentemente etiquetas <h3>. Desarrolla las ideas profundamente con viñetas largas <ul><li>, conservando anécdotas, nombres, y citas dadas en los borradores.]

<h2>🛠️ Herramientas y Recursos</h2>
[Extrae todas las plataformas o recursos de los borradores en una lista.]

<h2>📅 Carga Academica y Fechas Importantes</h2>
[Consolida TODAS las actividades, lecturas o tareas. Si ninguna de las partes aporta algo, pon <p><strong>No se mencionaron tareas, examenes ni fechas de entrega en toda la clase.</strong></p>]

<h2>🧠 Preguntas de Repaso</h2>
[Construye 5-8 preguntas estrategicas cruzando lo más vital enseñado.]

REGLAS DE CALIDAD:
- JAMAS metas comentarios meta como "En el borrador 3 se indico..." Escribe directamente la idea.
- Desarrolla las ideas, evita escribir un solo parrafo largo por tema, dividelo logicamente.

{HTML_FORMAT_RULES}
{TASK_DETECTION_RULES}
{EXPANDED_NOTES_RULES}
{materia_ctx}
---
BORRADORES PARCIALES A UNIFICAR:
{combined_summaries}
"""


def _build_gemini_continuation_prompt(previous_text: str) -> str:
    return (
        "Continua exactamente desde donde termino tu respuesta anterior.\n"
        "No reinicies secciones ni repitas texto.\n"
        "Mantiene el mismo formato HTML y completa el contenido faltante.\n\n"
        "RESPUESTA PREVIA:\n"
        f"{previous_text}"
    )


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


    if "8b" in lower:
        return 5500
    return 7000


def _model_candidates(provider: AIProvider) -> list[str]:
    if provider == AIProvider.GEMINI:
        candidates = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]

    elif provider == AIProvider.OLLAMA:
        candidates = [OLLAMA_MODEL]
    else:
        candidates = []
    return [model for model in dict.fromkeys(candidates) if model]



async def _call_gemini(prompt: str, override_model: str = None) -> str | None:
    if not GEMINI_API_KEY:
        return None

    candidates = [override_model] if override_model else _model_candidates(AIProvider.GEMINI)
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
                async with httpx.AsyncClient(timeout=AI_REQUEST_TIMEOUT) as http_client:
                    resp = await http_client.post(url, json=payload, headers={"Content-Type": "application/json"})

                if resp.status_code == 429:
                    wait_seconds = _parse_retry_after(resp.headers.get("retry-after", ""))
                    if wait_seconds > 0:
                        _remember_rate_limit(AIProvider.GEMINI, wait_seconds)
                    logger.warning(f"Rate limit [gemini] modelo={model}; intento fallback")
                    break

                if resp.status_code in {404, 500, 502, 503, 504}:
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

                if text and finish_reason in {"MAX_TOKENS", "LENGTH"}:
                    continuation_prompt = _build_gemini_continuation_prompt(text)
                    continuation_payload = {
                        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                        "contents": [{"parts": [{"text": continuation_prompt}]}],
                        "generationConfig": {
                            "temperature": AI_TEMPERATURE,
                            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
                        },
                    }
                    async with httpx.AsyncClient(timeout=AI_REQUEST_TIMEOUT) as cont_client:
                        cont_resp = await cont_client.post(
                            url,
                            json=continuation_payload,
                            headers={"Content-Type": "application/json"},
                        )
                    if cont_resp.status_code < 400:
                        cont_data = cont_resp.json()
                        cont_candidates = cont_data.get("candidates", [])
                        if cont_candidates:
                            cont_parts = cont_candidates[0].get("content", {}).get("parts", [])
                            cont_text = "\n".join(
                                part.get("text", "") for part in cont_parts if part.get("text")
                            ).strip()
                            if cont_text:
                                text = f"{text}\n{cont_text}".strip()

                if text:
                    return text
                return None
            except httpx.TimeoutException:
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
    if len(prompt_text) > OLLAMA_SAFE_PROMPT_CHARS:
        logger.warning(
            f"Prompt grande para Ollama ({len(prompt_text):,} chars). "
            f"Recortando a {OLLAMA_SAFE_PROMPT_CHARS:,} chars."
        )
        prompt_text = prompt_text[:OLLAMA_SAFE_PROMPT_CHARS]

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

            for attempt in range(max(1, OLLAMA_MAX_ATTEMPTS)):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
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
    except Exception as exc:
        logger.error(f"Error fatal [ollama]: {type(exc).__name__}: {exc}")
        return None



async def _call_provider(provider: AIProvider, prompt: str, override_model: str = None) -> str | None:
    if provider == AIProvider.GEMINI:
        return await _call_gemini(prompt, override_model)

    if provider == AIProvider.OLLAMA:
        return await _call_ollama(prompt)
    return None


async def _call_ai(prompt: str, override_model: str = None) -> str:
    primary = _detect_primary_provider()
    if primary == AIProvider.DISABLED:
        raise AIServiceUnavailable("No hay proveedores de IA configurados")

    for provider in _build_provider_order(primary):
        if _is_rate_limited(provider):
            continue

        result = await _call_provider(provider, prompt, override_model)
        if result:
            logger.info(f"Respuesta IA generada con proveedor={provider.value}")
            return result

    raise AIServiceUnavailable("Ningun proveedor pudo generar la respuesta")


async def _summarize_chunks(transcript: str, materia: str, progress_callback=None) -> str:
    primary = _detect_primary_provider()
    # Usamos 60k en modo "Obrero" (Map) para dividir la enorme clase a Gemini 3.1 Flash Lite 
    chunk_chars = 60000 if primary == AIProvider.GEMINI else AI_CHUNK_MAX_CHARS
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
    
    # REDUCE: Usamos el modelo supremo para rediseñar unicamente con la informacion procesada
    unify_prompt = _build_unification_prompt(combined, transcript=transcript, materia=materia)
    override_unifier = "gemini-2.5-flash" if primary == AIProvider.GEMINI else None
    
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
<h2>Tareas y acciones</h2>
<p><strong>No se mencionaron tareas, examenes ni fechas de entrega en esta clase.</strong></p>
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

    max_single = 150000 if primary == AIProvider.GEMINI else 30000
    if MAX_TRANSCRIPT_SIZE_SINGLE > 0:
        max_single = max(MAX_TRANSCRIPT_SIZE_SINGLE, 150000)

    route = "chunked" if len(transcript) > max_single else "single"
    
    # User's request: route single to gemini-2.5-flash
    override_model = "gemini-2.5-flash" if route == "single" and primary == AIProvider.GEMINI else None
    
    actual_model = override_model if override_model else model
    logger.info(f"Generando resumen | {len(transcript):,} chars | proveedor={primary.value} | modelo={actual_model}")

    if progress_callback:
        await progress_callback(20, "Generando resumen...")

    if len(transcript) > max_single:
        summary_html = await _summarize_chunks(transcript, materia, progress_callback=progress_callback)
    else:
        # Use single pass with override_model if gemini
        if override_model:
            logger.info(f"Ruta single - Usando modelo mas capaz y limitante para textos cortos: {override_model}")
        prompt = _build_summary_prompt(transcript, materia=materia)
        summary_html = clean_html(await _call_ai(prompt, override_model))

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

    logger.info(
        "Resumen metrics | status=ok | "
        f"provider={primary.value} | route={route} | "
        f"in_chars={len(transcript)} | out_chars={len(summary_html)} | out_words={_word_count(summary_html)} | "
        f"duration_ms={(time.perf_counter() - started_at) * 1000:.0f}"
    )

    return summary_html


def clean_html(raw: str) -> str:
    text = raw.strip()
    for marker in ("```html", "```"):
        text = text.replace(marker, "")
    text = text.strip()
    if text and not text.startswith("<"):
        text = f"<h1>Resumen de la sesion</h1>\n<p>{text}</p>"
    return text.strip()

