import asyncio
import random
import time
from enum import Enum

import httpx
from loguru import logger

from .config import (
    AI_CHUNK_MAX_CHARS,
    AI_REQUEST_TIMEOUT,
    AI_TEMPERATURE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_CHUNK_MAX_CHARS,
    GROQ_MODEL,
    GROQ_REQUEST_DELAY,
    GROQ_SAFE_PROMPT_CHARS,
    MAX_TRANSCRIPT_SIZE_SINGLE,
    SUMMARY_PROVIDER,
)

SYSTEM_PROMPT = (
    "Eres un asistente academico especializado en crear apuntes de clase a partir de "
    "transcripciones de audio. Tu trabajo es EXTRAER y ORGANIZAR fielmente el contenido "
    "real de la clase, NO generar contenido generico de libro de texto.\n\n"
    "REGLA FUNDAMENTAL: Solo incluye temas, conceptos, ejemplos, ejercicios y tareas que "
    "realmente se mencionaron en la transcripcion. Si el profesor menciono un concepto "
    "brevemente, puedes agregar una definicion corta entre parentesis para clarificar, "
    "pero NUNCA inventes secciones enteras con contenido que no se discutio en clase. "
    "La transcripcion proviene de audio y puede contener errores de reconocimiento de voz; "
    "interpreta inteligentemente las palabras mal transcritas segun el contexto academico. "
    "Produce HTML valido y bien estructurado."
)

HTML_FORMAT_RULES = """
REGLAS DE FORMATO (HTML puro, sin Markdown ni bloques de codigo):
- Usa <h1> para el titulo principal, <h2> para secciones, <h3> para subsecciones.
- Escribe parrafos explicativos (<p>) ricos en contenido; evita el abuso de listas.
- Usa <table> solo para datos comparativos o cronogramas.
- Usa <strong>, <em> y <code> para resaltar terminos clave.
- NO incluyas ```html, ```, ni estilos inline (style="...").
""".strip()

TASK_DETECTION_RULES = """
DETECCION DE CARGA ACADEMICA (MUY IMPORTANTE):
- Busca menciones a talleres, trabajos, ejercicios, tareas, examenes, parciales,
  quices, entregas, "para la proxima clase", "tienen que hacer", "van a entregar",
  "quiero que hagan", "me verifican", presentaciones, lecturas asignadas y actividades.
- Si hay carga academica, crea una seccion <h2>📝 Carga Academica y Fechas Importantes</h2>
  con tabla: Actividad | Fecha/Plazo | Descripcion detallada / Instrucciones del profesor.
- Si no hay menciones, escribe:
  <p><strong>No se mencionaron tareas, examenes ni fechas de entrega en esta clase.</strong></p>
""".strip()


class AIProvider(Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    DISABLED = "disabled"


PROVIDER_RATE_LIMIT_UNTIL: dict[AIProvider, float] = {}
RATE_LIMIT_WAIT_THRESHOLD = 120.0


def _build_single_phase_prompt(transcript: str, materia: str = "") -> str:
    materia_ctx = f"\nMATERIA: {materia}\nContextualiza el contenido dentro de esta asignatura.\n" if materia else ""
    return f"""Crea notas de estudio detalladas a partir de la siguiente transcripcion de clase.

REGLA PRINCIPAL - FIDELIDAD AL CONTENIDO REAL:
- SOLO incluye temas, conceptos, ejemplos y ejercicios que aparezcan en la transcripcion.
- NO generes parrafos genericos de libro de texto sobre el tema general.
- Captura detalles especificos: algoritmos, herramientas, software, formulas,
  recursos (URLs, notebooks), ejemplos concretos que uso el profesor.
- La transcripcion puede tener errores de reconocimiento de voz; interpreta segun contexto.
{materia_ctx}
ESTRUCTURA (HTML puro):
<h1>📚 [Titulo especifico basado en temas reales de la clase]</h1>
<h2>🚀 Resumen Ejecutivo</h2>
<p>[2-3 parrafos sobre lo cubierto, actividades y expectativas para el estudiante.]</p>
<h2>📖 Desarrollo Tematico</h2>
[Subcapitulos h3 por cada tema real discutido. Sin inventar contenido.]
<h2>🛠 Herramientas y Recursos</h2>
[Solo si se mencionaron recursos.]
[Seccion de carga academica - ver instrucciones abajo.]
<h2>🧠 Preguntas de Repaso</h2>
[5 preguntas basadas en el contenido real de esta clase.]

{HTML_FORMAT_RULES}

{TASK_DETECTION_RULES}

TRANSCRIPCION:
{transcript}"""


def _build_chunk_prompt(chunk: str, section_num: int, total_sections: int, materia: str = "") -> str:
    materia_ctx = f"\nMATERIA: {materia}\n" if materia else ""
    return f"""Extrae notas de estudio detalladas y fieles de esta seccion de una transcripcion de clase.
{materia_ctx}
INSTRUCCIONES CRITICAS:
1. Solo incluye lo discutido realmente en este fragmento.
2. Preserva detalles especificos (algoritmos, herramientas, ejemplos, ejercicios y recursos).
3. Elimina muletillas y repeticiones, conservando contenido relevante.
4. Si hay tareas/fechas/entregas, extraelas con instrucciones completas del profesor.

{HTML_FORMAT_RULES}

SECCION {section_num}/{total_sections}:
{chunk}"""


def _build_unification_prompt(combined_summaries: str, materia: str = "") -> str:
    materia_ctx = f"\nMATERIA: {materia}\nContextualiza las notas dentro de esta asignatura.\n" if materia else ""
    return f"""Unifica los siguientes borradores de secciones de una clase en un documento HTML cohesivo.
{materia_ctx}
INSTRUCCIONES:
1. El documento final debe reflejar solo lo discutido en clase.
2. Fusiona redundancias y deja narrativa fluida.
3. Preserva nombres, recursos, ejercicios e instrucciones del profesor.

ESTRUCTURA REQUERIDA:
<h1>📚 [Titulo especifico basado en los temas reales de la clase]</h1>
<h2>🚀 Resumen Ejecutivo</h2>
<h2>📖 Desarrollo Tematico</h2>
<h2>🛠 Herramientas y Recursos</h2>
[Seccion de carga academica - ver instrucciones abajo.]
<h2>🧠 Preguntas de Repaso</h2>

{HTML_FORMAT_RULES}

{TASK_DETECTION_RULES}

BORRADORES A UNIFICAR:
{combined_summaries}"""


def _get_provider_model(provider: AIProvider) -> str:
    if provider == AIProvider.GROQ:
        return GROQ_MODEL
    if provider == AIProvider.GEMINI:
        return GEMINI_MODEL
    return ""


def _get_provider_api_key(provider: AIProvider) -> str:
    if provider == AIProvider.GROQ:
        return GROQ_API_KEY
    if provider == AIProvider.GEMINI:
        return GEMINI_API_KEY
    return ""


def _is_rate_limited(provider: AIProvider) -> bool:
    until = PROVIDER_RATE_LIMIT_UNTIL.get(provider)
    if not until:
        return False
    if time.time() < until:
        return True
    PROVIDER_RATE_LIMIT_UNTIL.pop(provider, None)
    return False


def _detect_primary_provider() -> AIProvider:
    if SUMMARY_PROVIDER == "disabled":
        return AIProvider.DISABLED
    if SUMMARY_PROVIDER == "groq" and GROQ_API_KEY:
        return AIProvider.GROQ
    if SUMMARY_PROVIDER == "gemini" and GEMINI_API_KEY:
        return AIProvider.GEMINI
    if GEMINI_API_KEY:
        return AIProvider.GEMINI
    if GROQ_API_KEY:
        return AIProvider.GROQ
    return AIProvider.DISABLED


def _build_fallback_list(primary: AIProvider) -> list[AIProvider]:
    fallbacks: list[AIProvider] = []
    for provider in [AIProvider.GEMINI, AIProvider.GROQ]:
        if provider == primary:
            continue
        if _get_provider_api_key(provider):
            fallbacks.append(provider)
    return fallbacks


def _select_available_provider(primary: AIProvider, fallbacks: list[AIProvider]) -> AIProvider | None:
    if primary != AIProvider.DISABLED and not _is_rate_limited(primary):
        return primary
    for provider in fallbacks:
        if not _is_rate_limited(provider):
            return provider
    return None


async def _handle_rate_limit(
    provider: AIProvider,
    response: httpx.Response,
    attempt: int,
    max_retries: int,
    prompt: str,
    primary: AIProvider,
    fallbacks: list[AIProvider],
) -> str | None:
    retry_after = response.headers.get("retry-after", "")
    try:
        wait_seconds = float(retry_after) + 1.0
    except ValueError:
        wait_seconds = (5 * (2 ** attempt)) + random.uniform(1, 3)

    PROVIDER_RATE_LIMIT_UNTIL[provider] = time.time() + wait_seconds
    logger.warning(
        f"Rate limit [{provider.value}] | espera={wait_seconds:.0f}s | intento {attempt + 1}/{max_retries}"
    )

    if wait_seconds <= RATE_LIMIT_WAIT_THRESHOLD:
        await asyncio.sleep(wait_seconds)
        return None

    alt = _select_available_provider(primary, fallbacks)
    if alt and alt != provider:
        logger.info(f"Cambiando a {alt.value} por espera larga en {provider.value}")
        return await _call_provider(
            alt,
            prompt,
            max_retries=max(1, max_retries - attempt),
            primary=primary,
            fallbacks=fallbacks,
        )

    await asyncio.sleep(RATE_LIMIT_WAIT_THRESHOLD)
    return None


async def _call_groq(prompt: str, max_retries: int, primary: AIProvider, fallbacks: list[AIProvider]) -> str | None:
    api_key = GROQ_API_KEY
    if not api_key:
        return None

    if len(prompt) > GROQ_SAFE_PROMPT_CHARS:
        logger.warning(
            f"Prompt demasiado grande para Groq ({len(prompt):,} chars). "
            f"Saltando Groq para evitar 413."
        )
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": AI_TEMPERATURE,
        "max_tokens": 16384,
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=AI_REQUEST_TIMEOUT) as http_client:
                resp = await http_client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                handled = await _handle_rate_limit(
                    AIProvider.GROQ, resp, attempt, max_retries, prompt, primary, fallbacks
                )
                if handled:
                    return handled
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices"):
                return data["choices"][0]["message"]["content"].strip()
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} [groq]: {e.response.text[:200]}")
            if e.response.status_code == 413 or (400 <= e.response.status_code < 500):
                return None
        except httpx.TimeoutException:
            logger.warning(f"Timeout [groq] intento {attempt + 1}/{max_retries}")
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error [groq]: {type(e).__name__}: {e}")
            await asyncio.sleep(2)

    return None


async def _call_gemini(prompt: str, max_retries: int) -> str | None:
    if not GEMINI_API_KEY:
        return None

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": AI_TEMPERATURE,
            "maxOutputTokens": 32768,
        },
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=AI_REQUEST_TIMEOUT) as http_client:
                resp = await http_client.post(url, json=payload, headers={"Content-Type": "application/json"})

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("retry-after", "60"))
                delay = min(retry_after + 1.0, 60.0)
                PROVIDER_RATE_LIMIT_UNTIL[AIProvider.GEMINI] = time.time() + delay
                logger.warning(f"Rate limit [gemini] - esperando {delay:.0f}s")
                await asyncio.sleep(delay)
                continue

            if resp.status_code == 404:
                logger.error(f"Gemini: modelo '{GEMINI_MODEL}' no encontrado (404)")
                return None

            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "")
                    if text:
                        return text.strip()
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} [gemini]: {e.response.text[:200]}")
            if 400 <= e.response.status_code < 500:
                return None
        except Exception as e:
            logger.error(f"Error [gemini]: {type(e).__name__}: {e}")
            await asyncio.sleep(2)

    return None


async def _call_provider(
    provider: AIProvider,
    prompt: str,
    max_retries: int,
    primary: AIProvider,
    fallbacks: list[AIProvider],
) -> str | None:
    if provider == AIProvider.GROQ:
        return await _call_groq(prompt, max_retries=max_retries, primary=primary, fallbacks=fallbacks)
    if provider == AIProvider.GEMINI:
        return await _call_gemini(prompt, max_retries=max_retries)
    return None


def _split_transcript(transcript: str, max_chars: int = 12000) -> list[str]:
    words = transcript.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        wlen = len(word) + 1
        if length + wlen > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            length = wlen
        else:
            current.append(word)
            length += wlen
    if current:
        chunks.append(" ".join(current))
    return chunks


async def _unify_partials_resilient(partials: list[str], materia: str) -> str | None:
    current = [p for p in partials if p and "[Seccion no procesada]" not in p]
    if not current:
        return None

    combined = "\n\n".join(f"<!-- Seccion {i + 1} -->\n{s}" for i, s in enumerate(current))
    direct_prompt = _build_unification_prompt(combined, materia=materia)
    direct = await _call_ai(direct_prompt)
    if direct:
        return clean_html(direct)

    round_num = 1
    while len(current) > 1 and round_num <= 4:
        next_level: list[str] = []
        batch_size = 3
        logger.warning(
            f"Unificacion directa fallo. Intentando unificacion jerarquica "
            f"(ronda {round_num}, documentos={len(current)})."
        )

        for i in range(0, len(current), batch_size):
            group = current[i : i + batch_size]
            group_combined = "\n\n".join(f"<!-- Bloque {i + j + 1} -->\n{doc}" for j, doc in enumerate(group))
            group_prompt = _build_unification_prompt(group_combined, materia=materia)
            group_summary = await _call_ai(group_prompt)
            if group_summary:
                next_level.append(clean_html(group_summary))
            else:
                next_level.append("\n\n".join(group))

            await asyncio.sleep(min(GROQ_REQUEST_DELAY, 2.0))

        current = next_level
        round_num += 1

    return clean_html(current[0]) if len(current) == 1 else None


async def _call_ai(prompt: str, max_retries: int = 5) -> str | None:
    primary = _detect_primary_provider()
    if primary == AIProvider.DISABLED:
        logger.error("No hay API keys configuradas para proveedores de IA")
        return None

    fallbacks = _build_fallback_list(primary)
    provider = _select_available_provider(primary, fallbacks)

    if not provider and PROVIDER_RATE_LIMIT_UNTIL:
        min_wait = min(PROVIDER_RATE_LIMIT_UNTIL.values()) - time.time()
        if min_wait > 0:
            logger.info(f"Todos los proveedores en rate-limit. Esperando {min_wait:.0f}s...")
            await asyncio.sleep(min_wait + 1)
            provider = _select_available_provider(primary, fallbacks)

    if not provider:
        logger.error("No hay proveedores de IA disponibles")
        return None

    result = await _call_provider(provider, prompt, max_retries=max_retries, primary=primary, fallbacks=fallbacks)
    if result:
        return result

    for alt in fallbacks:
        if alt == provider:
            continue
        result = await _call_provider(alt, prompt, max_retries=max_retries, primary=primary, fallbacks=fallbacks)
        if result:
            return result

    return None


async def _generate_two_phase_summary(
    transcript: str,
    materia: str,
    progress_callback=None,
) -> str:
    primary = _detect_primary_provider()
    chunk_chars = AI_CHUNK_MAX_CHARS
    if primary == AIProvider.GROQ:
        chunk_chars = min(chunk_chars, GROQ_CHUNK_MAX_CHARS)

    chunks = _split_transcript(transcript, max_chars=chunk_chars)
    total = len(chunks)
    partials: list[str] = []

    logger.info(
        f"Resumen IA en dos fases | {len(transcript):,} chars | {total} secciones | "
        f"chunk_chars={chunk_chars}"
    )

    for i, chunk in enumerate(chunks):
        section = i + 1
        if progress_callback:
            pct = (i / max(total, 1)) * 70
            await progress_callback(pct, f"Procesando seccion {section} de {total}")

        prompt = _build_chunk_prompt(chunk, section, total, materia=materia)
        summary = await _call_ai(prompt)
        if summary:
            partials.append(summary)
        else:
            partials.append("<p><em>[Seccion no procesada]</em></p>")

        if i < total - 1:
            await asyncio.sleep(GROQ_REQUEST_DELAY)

    valid_count = sum(1 for s in partials if "[Seccion no procesada]" not in s)
    if valid_count == 0:
        return _fallback_summary(transcript)

    if progress_callback:
        await progress_callback(85, "Combinando resumenes...")

    await asyncio.sleep(min(GROQ_REQUEST_DELAY * 3, 30.0))
    unified = await _unify_partials_resilient(partials, materia=materia)
    if unified:
        return unified

    logger.error("No se pudo unificar el resumen tras multiples intentos; usando fallback completo.")
    return _fallback_summary(transcript)


def _fallback_summary(transcript: str) -> str:
    word_count = len(transcript.split()) if transcript else 0
    duration = word_count / 150 if word_count else 0
    return f"""<h1>📚 Resumen de Clase</h1>

<h2>📋 Informacion</h2>
<ul>
  <li><strong>Duracion estimada:</strong> {duration:.0f} minutos</li>
  <li><strong>Palabras:</strong> {word_count:,}</li>
</ul>

<h2>📝 Acciones Requeridas</h2>
<p>No fue posible generar el resumen con IA. Revisa la transcripcion para:</p>
<ul>
  <li>Identificar tareas y fechas de entrega</li>
  <li>Extraer conceptos clave</li>
  <li>Documentar procedimientos</li>
</ul>

<hr>
<p><em>Resumen automatico - revisar transcripcion para detalles.</em></p>"""


async def summarize(transcript: str, materia: str, progress_callback=None) -> str:
    if not transcript or len(transcript.strip()) < 50:
        return _fallback_summary(transcript)

    primary = _detect_primary_provider()
    if primary == AIProvider.DISABLED:
        return _fallback_summary(transcript)

    provider_model = _get_provider_model(primary)
    max_single = 500000 if primary == AIProvider.GEMINI else 30000
    if MAX_TRANSCRIPT_SIZE_SINGLE > 0:
        max_single = min(max_single, MAX_TRANSCRIPT_SIZE_SINGLE)

    logger.info(
        f"Generando resumen | {len(transcript):,} chars | proveedor={primary.value} | modelo={provider_model}"
    )

    if len(transcript) > max_single:
        return await _generate_two_phase_summary(transcript, materia, progress_callback=progress_callback)

    if progress_callback:
        await progress_callback(20, "Generando resumen...")

    prompt = _build_single_phase_prompt(transcript, materia=materia)
    result = await _call_ai(prompt)
    if result:
        return clean_html(result)
    return _fallback_summary(transcript)


def clean_html(raw: str) -> str:
    text = raw.strip()
    for marker in ("```html", "```"):
        text = text.replace(marker, "")
    text = text.strip()
    if text and not text.startswith("<"):
        text = f"<h1>📚 Resumen de Clase</h1>\n{text}"
    return text.strip()
