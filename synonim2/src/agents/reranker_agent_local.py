"""
Agente 3 (variante local) - mismo rol que reranker_agent.py: recibe el
término y los candidatos que ya seleccionó la herramienta de embeddings, y
le pide a un LLM que los reordene de mejor a peor. Aquí el LLM corre en
local vía Ollama (qwen3.5:2b) en vez de la API de OpenRouter, para comparar
velocidad contra reranker_agent.py con la misma lógica de fallback (si el
LLM omite algún candidato, se añade al final en su orden original).
"""

from ollama_client import llamar_modelo

from ..text_utils import normalize, parse_candidates

AGENTE_RERANKER_LOCAL = {
    "modelo": "qwen3.5:2b",
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


def reordenar_candidatos(term: str, candidatos: list[str], agente: dict = AGENTE_RERANKER_LOCAL) -> list[str]:
    if not candidatos:
        return []

    lista_texto = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidatos))
    mensaje = f'Term: "{term}"\n\nCandidate synonyms:\n{lista_texto}'

    try:
        respuesta = llamar_modelo(agente, mensaje)
        ordenados_llm = parse_candidates(respuesta)
    except Exception:
        ordenados_llm = []

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


if __name__ == "__main__":
    term = "recommender system"
    candidatos = ["recommendation engine", "suggestion system", "filtering system", "advice engine", "ranking system"]
    print("Reordenados:", reordenar_candidatos(term, candidatos))
