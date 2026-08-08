"""
Agente con tool-calling NATIVO para Synonym Finding.

A diferencia del pipeline de la tarea actual (generator_agent ->
embedding_retrieval -> reranker_agent_local, siempre en ese orden fijo),
aquí el LLM decide en tiempo de ejecución cuántas veces llamar a
`retrieve_candidates` (con el término y las paráfrasis que se le ocurran)
antes de llamar a `finalize` con su lista final. Es el patrón "agente que
usa tools" pedido explícitamente, implementado con function calling nativo
de Ollama (no una librería de orquestación tipo LangGraph - los grafos MAS
de new/src2 y synonic fueron consistentemente el peor o segundo peor grupo
de métodos en cada comparación de este proyecto, así que no se reproduce
ese patrón aquí).

Red de seguridad (mismo principio que reranker_agent.py/reranker_agent_local.py
- nunca perder recall por un fallo del LLM):
  - La primera búsqueda con el término original se hace SIEMPRE de forma
    determinista antes de ceder el turno al LLM, para que el agente nunca
    arranque con el pool vacío.
  - Un candidato que `finalize` incluya pero que nunca vino de
    retrieve_candidates se descarta (no se puede "inventar" un candidato).
  - Si el LLM nunca llama a `finalize` con éxito (agota max_turns, responde
    en texto libre, o su lista no contiene ningún candidato real), se
    devuelve el pool acumulado de todo lo visto, en orden de aparición.
"""

from ollama_client import llamar_modelo_mensajes

from ..text_utils import normalize, parse_candidates
from ..tools.embedding_retrieval import EMBEDDER_NAME, retrieve_top_k

AGENTE_TOOL = {
    "modelo": "llama3.2:3b",
    "instrucciones": (
        "You are a synonym-finding agent. You must find the best synonym "
        "for a given term, choosing ONLY from a fixed vocabulary you can "
        "search with the retrieve_candidates tool - you cannot invent "
        "candidates. Call retrieve_candidates one or more times (try the "
        "term itself and, if you think it will help, a paraphrase or a "
        "broader/narrower phrasing) to gather good candidates. When you "
        "are confident, call finalize with your final top 5 candidates "
        "ordered from best to worst, using the exact candidate strings "
        "returned by retrieve_candidates. Do not answer in plain text."
    ),
    "temperature": 0.2,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_candidates",
            "description": (
                "Search the fixed candidate vocabulary for terms "
                "semantically close to a query. Returns up to top_k "
                "candidate strings, best match first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - the term or a paraphrase of it.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many candidates to return (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": (
                "Submit the final ranked list of up to 5 candidate synonyms, "
                "best first. Must use exact strings previously returned by "
                "retrieve_candidates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ranked_candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Up to 5 candidate strings, best synonym first.",
                    }
                },
                "required": ["ranked_candidates"],
            },
        },
    },
]


def buscar_sinonimo_con_tools(
    term: str,
    index_terms: list[str],
    index_embeddings,
    embedder=None,
    embedder_name: str = EMBEDDER_NAME,
    agente: dict = AGENTE_TOOL,
    max_turns: int = 4,
    top_k_final: int = 5,
) -> list[str]:
    """Ejecuta el agente de tool-calling para un término. Devuelve hasta
    top_k_final candidatos ordenados, best-first."""

    # norm -> texto original, en orden de primera aparición (= mejor score
    # visto entre todas las búsquedas hechas en esta llamada).
    vistos_norm: dict[str, str] = {}

    def _registrar(candidatos: list[str]) -> None:
        for c in candidatos:
            norm_c = normalize(c)
            if norm_c and norm_c not in vistos_norm:
                vistos_norm[norm_c] = c

    def _ejecutar_retrieve(query: str, top_k) -> list[str]:
        try:
            top_k = max(1, min(int(top_k or 5), 10))
        except (TypeError, ValueError):
            top_k = 5
        resultados = retrieve_top_k(
            [str(query)], index_terms, index_embeddings, embedder=embedder,
            embedder_name=embedder_name, top_k=top_k, exclude_terms=[term],
        )
        _registrar(resultados)
        return resultados

    primeros = _ejecutar_retrieve(term, top_k_final)

    mensajes = [
        {"role": "system", "content": agente["instrucciones"]},
        {
            "role": "user",
            "content": (
                f'Term: "{term}"\n\n'
                f"Initial candidates already retrieved for the term itself: {primeros}\n\n"
                "Call retrieve_candidates again with paraphrases if you think "
                "it will surface better candidates, then call finalize."
            ),
        },
    ]

    finalizado: list[str] | None = None

    for _ in range(max_turns):
        try:
            mensaje = llamar_modelo_mensajes(agente, mensajes, tools=TOOLS)
        except Exception:
            break

        tool_calls = mensaje.get("tool_calls") or []

        if not tool_calls:
            # Respondió en texto en vez de llamar a finalize - se intenta
            # parsear igual antes de caer al fallback del pool acumulado.
            _registrar(parse_candidates(mensaje.get("content", "")))
            break

        mensajes.append({
            "role": "assistant",
            "content": mensaje.get("content", ""),
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            funcion = call.get("function", {})
            nombre = funcion.get("name")
            argumentos = funcion.get("arguments", {}) or {}

            if nombre == "retrieve_candidates":
                resultado = _ejecutar_retrieve(
                    argumentos.get("query", term), argumentos.get("top_k", 5)
                )
                mensajes.append({"role": "tool", "content": str(resultado)})

            elif nombre == "finalize":
                crudos = argumentos.get("ranked_candidates", []) or []
                # Solo se aceptan candidatos que realmente vinieron de
                # retrieve_candidates - uno inventado por el LLM se descarta.
                finalizado = [
                    vistos_norm[normalize(c)] for c in crudos
                    if normalize(c) in vistos_norm
                ]
                mensajes.append({"role": "tool", "content": "ok"})

            else:
                mensajes.append({"role": "tool", "content": f"unknown tool: {nombre}"})

        if finalizado is not None:
            break

    if finalizado is not None:
        # Cualquier candidato visto que el LLM no incluyó en su finalize se
        # añade al final, en orden de aparición - nunca se pierde recall.
        for texto in vistos_norm.values():
            if texto not in finalizado:
                finalizado.append(texto)
        return finalizado[:top_k_final]

    # El LLM nunca llamó a finalize con éxito: se devuelve el pool
    # acumulado en orden de aparición, nunca una lista vacía si hubo al
    # menos una búsqueda (y siempre la hay, por la búsqueda inicial).
    return list(vistos_norm.values())[:top_k_final]


if __name__ == "__main__":
    from ..tools.embedding_retrieval import build_solution_index

    pool = [
        "recommendation engine", "suggestion system", "filtering system",
        "advice engine", "ranking system", "shares", "stock market",
    ]
    terms, embeddings = build_solution_index(pool)
    resultado = buscar_sinonimo_con_tools("recommender system", terms, embeddings)
    print("Resultado:", resultado)
