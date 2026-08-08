"""
Agente 1 - generador: le pide al LLM (local, vía Ollama) variantes/sinónimos
del término de entrada. Devuelve el término original + hasta 9 variantes
(10 candidatos por defecto). Esta lista se usa como conjunto de QUERIES
para la herramienta de retrieval (rsc/tools/embedding_retrieval.py), no
como respuesta final - mismo rol que generate_query_variants en
new/src2/run_llm_expansion_retrieval.py.

El reranker que usa el pipeline por defecto (rsc/agents/reranker_agent_local.py)
también corre en local vía Ollama - la variante que llama a la API de
OpenRouter (rsc/agents/reranker_agent.py) sigue existiendo para comparar
velocidad/calidad, pero run_experiment.py ya no la usa.
"""

from ollama_client import llamar_modelo

from ..text_utils import normalize, parse_candidates

AGENTE_GENERADOR = {
    "modelo": "llama3.2:3b",
    "instrucciones": (
        "You are a domain terminology expert. "
        "Generate English synonyms or closely related semantic variants "
        "for the given academic/domain term. These variants will be used "
        "as search queries against a vocabulary index, so include "
        "broader, narrower, and near-equivalent phrasings. "
        "Return only a JSON list of short variants. "
        "Do not explain."
    ),
    "temperature": 0.3,
}

# temperature=0 variant: AGENTE_GENERADOR's 0.3 causes measured run-to-run
# hit@1 variance (~1-2pt across repeats) that confounded the original
# fusion-weight sweep (see EXPERIMENTS_LOG.md). This is used to get a clean,
# low-noise weight retune - AGENTE_GENERADOR itself is left untouched since
# other call sites/results still reference it.
AGENTE_GENERADOR_T0 = {**AGENTE_GENERADOR, "temperature": 0.0}

# Backend Qwen vía OpenRouter (API de pago) en vez de Ollama local - mismo
# rol que AGENTE_GENERADOR_T0, pero para comparar si un modelo más grande
# genera mejores variantes. No importa openrouter_client aquí (este módulo
# no debe depender de tener OPENROUTER_API_KEY configurado solo para poder
# generar con Ollama) - el caller pasa `llamar_fn` explícitamente, ver
# src/methods.py.
AGENTE_GENERADOR_QWEN = {
    "modelo": "qwen/qwen3-32b",  # ver src/agents/reranker_agent.py para el porqué (qwen3.5-9b era poco fiable)
    "instrucciones": AGENTE_GENERADOR["instrucciones"],
    "temperature": 0.0,
}


def generar_variantes(term: str, n: int = 9, agente: dict = AGENTE_GENERADOR, llamar_fn=llamar_modelo) -> list[str]:
    """Pide al LLM hasta n variantes del término. Devuelve [] si la llamada
    falla o no se puede parsear nada útil (el caller decide el fallback)."""
    mensaje = f'Term: "{term}"'

    try:
        respuesta = llamar_fn(agente, mensaje)
        variantes = parse_candidates(respuesta)[:n]
    except Exception:
        variantes = []

    return variantes


def generar_candidatos(term: str, n: int = 9, agente: dict = AGENTE_GENERADOR, llamar_fn=llamar_modelo) -> list[str]:
    """
    Devuelve el término original en primera posición + hasta n variantes
    generadas por el LLM, deduplicadas por texto normalizado (10 elementos
    por defecto). Si la generación falla por completo, devuelve al menos
    [term] para que la herramienta de retrieval siga teniendo una query
    con la que buscar.
    """
    variantes = generar_variantes(term, n=n, agente=agente, llamar_fn=llamar_fn)

    candidatos = [term]
    vistos = {normalize(term)}

    for variante in variantes:
        norm_variante = normalize(variante)
        if norm_variante and norm_variante not in vistos:
            candidatos.append(variante)
            vistos.add(norm_variante)

    return candidatos


if __name__ == "__main__":
    term = "recommender system"
    candidatos = generar_candidatos(term)
    print(f"Candidatos ({len(candidatos)}):")
    for c in candidatos:
        print(" -", c)
