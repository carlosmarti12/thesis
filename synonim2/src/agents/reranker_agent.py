"""
Agente 3 (variante API) - recibe el término y los candidatos que ya
seleccionó la herramienta de embeddings, y le pide a un LLM (vía OpenRouter)
que los reordene de mejor a peor. Si el LLM omite algún candidato de la
lista, se añade al final en su orden original - así una respuesta parcial
del LLM nunca hace perder candidatos, solo afecta al orden (mismo patrón que
synonic's llm_ranking_agent con keep_all_via_append=True).

run_experiment.py usa por defecto rsc/agents/reranker_agent_local.py (mismo
comportamiento, pero con qwen3.5:2b en local vía Ollama) - este archivo se
conserva para comparar velocidad/calidad contra la API.

Modelo: qwen/qwen3-32b (no qwen3.5-9b) - se probó primero qwen3.5-9b y
resultó tener fallos transitorios frecuentes en OpenRouter (~2 de cada 3
llamadas devolvían "Service unavailable" / 503 embebido en una respuesta
200, no como error HTTP - ver openrouter_client.py::_completar_con_reintentos
para el porqué el cliente de openai-python no reintentaba esto solo).
qwen3-32b resultó 5/5 fiable en pruebas directas, más rápido (2-6s vs
7-21s), y devuelve JSON limpio con este prompt (sin bloques <think> que
rompan parse_candidates) - ver EXPERIMENTS_LOG.md.
"""

from openrouter_client import llamar_modelo

from ..text_utils import normalize, parse_candidates

AGENTE_RERANKER = {
    "modelo": "qwen/qwen3-32b",
    "instrucciones": (
        "You are a strict semantic ranking agent for domain terminology. "
        "You receive one input term and a list of candidate synonyms. "
        "Your task is to ORDER the candidates from best to worst according to semantic equivalence. "
        "Do not invent new candidates. "
        "Do not remove candidates unless they are completely invalid. "
        "Use the exact candidate strings from the provided list. "
        "Return only a JSON list of candidate strings ordered from best to worst."
    ),
    "temperature": 0.0,
}

# Contador global de fallos - run_eval.py lo lee al final de la corrida y lo
# mete en el summary (fallback_count) para que un fallo sistemático (p.ej.
# créditos agotados) sea visible en las métricas agregadas, no solo en el
# log de consola. Resetear con reset_fallback_count() al inicio de una
# corrida nueva si se quiere contar por-corrida en vez de acumulado.
fallback_count = 0


def reset_fallback_count() -> None:
    global fallback_count
    fallback_count = 0


def reordenar_candidatos(term: str, candidatos: list[str], agente: dict = AGENTE_RERANKER) -> list[str]:
    if not candidatos:
        return []

    lista_texto = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidatos))
    mensaje = f'Term: "{term}"\n\nCandidate synonyms:\n{lista_texto}'

    try:
        respuesta = llamar_modelo(agente, mensaje)
        ordenados_llm = parse_candidates(respuesta)
    except Exception as e:
        # Silencioso a propósito para no perder recall si falla una sola
        # llamada (hiccup transitorio) - pero SIN AVISO esto es peligroso:
        # un fallo sistemático (p.ej. créditos de OpenRouter agotados,
        # 402) produce una corrida que "funciona" pero rerankea CERO filas,
        # dando métricas idénticas a no rerankear sin ningún error visible
        # (pasó de verdad el 2026-07-20 - ver EXPERIMENTS_LOG.md). Imprimir
        # para que un fallo sistemático sea visible en el log de la corrida
        # en vez de mezclarse silenciosamente en las métricas agregadas.
        print(f"WARNING [reordenar_candidatos] fallback a orden original para {term!r}: {e}")
        ordenados_llm = []
        global fallback_count
        fallback_count += 1

    originales_por_norm = {normalize(c): c for c in candidatos if normalize(c)}

    resultado = []
    vistos = set()

    # Primero, el orden que propone el LLM (solo candidatos válidos y no repetidos).
    for item in ordenados_llm:
        norm_item = normalize(item)
        if norm_item in originales_por_norm and norm_item not in vistos:
            resultado.append(originales_por_norm[norm_item])
            vistos.add(norm_item)

    # Después, cualquier candidato que el LLM haya omitido - nunca se pierde recall.
    for c in candidatos:
        norm_c = normalize(c)
        if norm_c and norm_c not in vistos:
            resultado.append(c)
            vistos.add(norm_c)

    return resultado


def reordenar_candidatos_self_consistency(
    term: str, candidatos: list[str], n_calls: int = 3, agente: dict = AGENTE_RERANKER,
) -> list[str]:
    """
    Auto-consistencia: llama a reordenar_candidatos N veces sobre el MISMO
    término+candidatos y agrega por Borda count (posición 1 = len(candidatos)
    puntos, posición 2 = len(candidatos)-1, ..., suma entre las N llamadas,
    ordena descendente). Motivado por un hallazgo del 2026-07-21: el
    reranker de Qwen a temperature=0 NO es perfectamente reproducible entre
    llamadas de API separadas (ver EXPERIMENTS_LOG.md) - en vez de tratar
    esa variabilidad como puro ruido a ignorar, se usa aquí como señal: si
    el modelo se inclina por un candidato la mayoría de las N veces, esa
    mayoría es más fiable que una sola tirada, especialmente en decisiones
    top1-vs-top2 muy reñidas (que es exactamente donde se concentran los
    119 casos "wrong rank" del dataset completo - ver EXPERIMENTS_LOG.md).
    N llamadas a la API en vez de 1 - más caro y más lento, pero el coste
    no es prioridad para esta línea de experimentos (indicado por el
    usuario).
    """
    if not candidatos:
        return []

    n = len(candidatos)
    puntos: dict[str, float] = {normalize(c): 0.0 for c in candidatos}
    display_por_norm = {normalize(c): c for c in candidatos}

    for _ in range(n_calls):
        orden = reordenar_candidatos(term, candidatos, agente=agente)
        for posicion, candidato in enumerate(orden):
            norm_c = normalize(candidato)
            if norm_c in puntos:
                puntos[norm_c] += n - posicion

    ranked_norms = sorted(puntos.keys(), key=lambda nc: puntos[nc], reverse=True)
    return [display_por_norm[nc] for nc in ranked_norms]


if __name__ == "__main__":
    term = "recommender system"
    candidatos = ["recommendation engine", "suggestion system", "filtering system", "advice engine", "ranking system"]
    print("Reordenados:", reordenar_candidatos(term, candidatos))
