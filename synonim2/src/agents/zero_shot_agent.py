"""
Agente zero-shot: le pide al LLM su(s) mejor(es) sinónimo(s) para el
término, sin retrieval previo ni acceso al pool de candidatos - el LLM
nunca ve la lista de candidatos válidos, solo el término. Su respuesta
libre se ancla después al pool real vía la herramienta de embeddings
(rsc/tools/embedding_retrieval.py::retrieve_top_k, usando sus propuestas
como queries), para que sea evaluable como Synonym Finding (elegir de la
lista de en_synonym) sin dejar de ser generación libre por debajo - mismo
principio que new/src2/run_llm_expansion_retrieval.py y
synonic/src/baselines.py::llm_expansion_baseline, con una sola generación
en vez de varias paráfrasis.
"""

from ollama_client import llamar_modelo

from ..text_utils import parse_candidates

AGENTE_ZERO_SHOT = {
    "modelo": "llama3.2:3b",
    "instrucciones": (
        "You are a domain terminology expert. Given a term, propose up to "
        "5 of the best synonyms or closely related phrasings for it. "
        "Return only a JSON list, best first. Do not explain."
    ),
    "temperature": 0.3,
}

# Backend Qwen (OpenRouter, qwen3-32b) para el mismo rol - se usa como
# fuente ADICIONAL de candidatos (conocimiento paramétrico del LLM, sin ver
# el pool) en llm_expansion_rerank_qwen_llmguess (ver src/methods.py):
# apunta a los mismos casos de "retrieval miss" que ampliar el pool de
# embeddings, pero por una vía distinta (memoria del modelo en vez de
# vecinos más cercanos).
AGENTE_ZERO_SHOT_QWEN = {
    "modelo": "qwen/qwen3-32b",
    "instrucciones": (
        "You are a domain terminology expert. Given a term, propose up to "
        "3 of the best synonyms or closely related phrasings for it. "
        "Return only a JSON list, best first. Do not explain."
    ),
    "temperature": 0.0,
}


def generar_respuesta_zero_shot(term: str, agente: dict = AGENTE_ZERO_SHOT, llamar_fn=llamar_modelo) -> list[str]:
    """Devuelve hasta 5 propuestas libres del LLM, o [] si la llamada falla
    o no se puede parsear nada útil (el caller decide el fallback, p.ej.
    usar el propio término como query)."""
    try:
        respuesta = llamar_fn(agente, f'Term: "{term}"')
        return parse_candidates(respuesta)
    except Exception:
        return []


if __name__ == "__main__":
    print(generar_respuesta_zero_shot("recommender system"))
