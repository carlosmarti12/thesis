from typing import Any

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

# Cap por defecto de tokens generados por llamada. Sin esto, una generación
# degenerada ocasional puede decodificar miles de tokens en vez de parar en
# la respuesta corta esperada (una lista JSON de candidatos) - observado en
# esta misma carpeta: una sola llamada de rsc/agents/reranker_agent_local.py
# vía tool_agent.py llegó a 2000+ tokens y seguía subiendo, dejando el
# proceso "colgado" hasta el timeout de 120s en vez de responder en
# segundos. Mismo fix ya aplicado en new/src2/run_llm_rerank.py
# (`num_predict=300`) tras encontrar ahí el mismo problema (39k+ tokens en
# un término). Cada agente puede pisarlo con su propio "num_predict" si
# necesita respuestas más largas.
DEFAULT_NUM_PREDICT = 300


def llamar_modelo_mensajes(
    agente: dict[str, Any],
    mensajes: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Variante de bajo nivel de llamar_modelo: recibe la lista de mensajes
    completa (para poder incluir turnos previos de tool-calling) y un
    esquema de `tools` opcional (function calling nativo de Ollama,
    confirmado compatible con llama3.2:3b/qwen3.5:2b/qwen3.5:latest/
    gemma4:latest - todos reportan "tools" en `ollama list`).

    Devuelve el mensaje COMPLETO del modelo (dict con "content" y, si el
    modelo decidió usar una tool, "tool_calls"), no solo el texto - así el
    caller puede inspeccionar tool_calls sin otra llamada a la API. Usado
    por rsc/agents/tool_agent.py para el bucle de tool-calling; llamar_modelo
    (más abajo) sigue siendo el camino simple para agentes de un solo turno.
    """
    payload: dict[str, Any] = {
        "model": agente["modelo"],
        "messages": mensajes,
        "stream": False,
        "options": {
            "temperature": agente.get("temperature", 0.3),
            "num_predict": agente.get("num_predict", DEFAULT_NUM_PREDICT),
        },
    }
    if tools:
        payload["tools"] = tools

    respuesta = requests.post(OLLAMA_URL, json=payload, timeout=120)
    respuesta.raise_for_status()

    return respuesta.json().get("message", {})


def llamar_modelo(
    agente: dict[str, Any],
    mensaje_usuario: str,
) -> str:
    """
    Envía el agente y el mensaje del usuario a un servidor Ollama local.
    Este es el único lugar que realiza llamadas a Ollama. Misma firma que
    openrouter_client.llamar_modelo para que los agentes puedan cambiar de
    backend sin tocar su propia lógica.
    """

    if not mensaje_usuario.strip():
        raise ValueError("El mensaje no puede estar vacío")

    respuesta = requests.post(
        OLLAMA_URL,
        json={
            "model": agente["modelo"],
            "messages": [
                {"role": "system", "content": agente["instrucciones"]},
                {"role": "user", "content": mensaje_usuario},
            ],
            "stream": False,
            "options": {
                "temperature": agente.get("temperature", 0.3),
                "num_predict": agente.get("num_predict", DEFAULT_NUM_PREDICT),
            },
        },
        timeout=120,
    )
    respuesta.raise_for_status()

    contenido = respuesta.json().get("message", {}).get("content")

    if not contenido:
        raise RuntimeError("El modelo devolvió una respuesta vacía")

    return contenido
