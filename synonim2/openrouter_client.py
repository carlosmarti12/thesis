import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError(
        "No se encontró OPENROUTER_API_KEY en el archivo .env"
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=90.0,
    max_retries=3,
)

# OpenRouter a veces reporta fallos del proveedor upstream (p.ej. "Service
# unavailable", 503) como una respuesta HTTP 200 normal con un campo
# `error` embebido en vez de un error HTTP - el cliente de openai-python NO
# reintenta esto automáticamente (su lógica de reintentos solo mira el
# status code HTTP). Confirmado empíricamente con qwen/qwen3.5-9b: ~1 de
# cada 3 llamadas falla así de forma transitoria y la siguiente funciona
# sin cambiar nada. _completar_con_reintentos añade reintentos explícitos
# para este patrón concreto.
MAX_REINTENTOS_ERROR_EMBEBIDO = 3
ESPERA_ENTRE_REINTENTOS_S = 2.0


def _completar_con_reintentos(agente: dict[str, Any], mensaje_usuario: str):
    ultimo_error = None
    for intento in range(MAX_REINTENTOS_ERROR_EMBEBIDO):
        respuesta = client.chat.completions.create(
            model=agente["modelo"],
            messages=[
                {"role": "system", "content": agente["instrucciones"]},
                {"role": "user", "content": mensaje_usuario},
            ],
            temperature=agente.get("temperature", 0.3),
        )
        if respuesta.choices:
            return respuesta
        ultimo_error = getattr(respuesta, "error", None)
        if intento < MAX_REINTENTOS_ERROR_EMBEBIDO - 1:
            time.sleep(ESPERA_ENTRE_REINTENTOS_S)

    raise RuntimeError(f"OpenRouter no devolvió choices tras {MAX_REINTENTOS_ERROR_EMBEBIDO} intentos: {ultimo_error}")


def llamar_modelo(
    agente: dict[str, Any],
    mensaje_usuario: str,
) -> str:
    """
    Envía el agente y el mensaje del usuario a OpenRouter.
    Este es el único lugar que realiza llamadas a la API.
    """

    if not mensaje_usuario.strip():
        raise ValueError("El mensaje no puede estar vacío")

    respuesta = _completar_con_reintentos(agente, mensaje_usuario)

    contenido = respuesta.choices[0].message.content

    if not contenido:
        raise RuntimeError("El modelo devolvió una respuesta vacía")

    return contenido


# Precios publicados por OpenRouter en USD por millón de tokens, verificados
# en openrouter.ai el 2026-07-20 (verificar de nuevo antes de confiar en una
# cifra vieja - esto es solo una estimación para reportar coste aproximado,
# no una factura).
QWEN_3_32B_PRICE_PER_MTOK = {"prompt": 0.08, "completion": 0.28}
QWEN_35_9B_PRICE_PER_MTOK = {"prompt": 0.10, "completion": 0.15}


def llamar_modelo_con_uso(
    agente: dict[str, Any],
    mensaje_usuario: str,
) -> tuple[str, dict[str, Any]]:
    """
    Igual que llamar_modelo, pero además devuelve el uso de tokens
    (prompt_tokens/completion_tokens) para poder estimar coste por corrida -
    no reemplaza a llamar_modelo (que sigue siendo la única función que usan
    los agentes existentes), solo se usa donde se quiere trackear coste."""
    if not mensaje_usuario.strip():
        raise ValueError("El mensaje no puede estar vacío")

    respuesta = _completar_con_reintentos(agente, mensaje_usuario)

    contenido = respuesta.choices[0].message.content
    if not contenido:
        raise RuntimeError("El modelo devolvió una respuesta vacía")

    uso = respuesta.usage.model_dump() if respuesta.usage else {}
    return contenido, uso
