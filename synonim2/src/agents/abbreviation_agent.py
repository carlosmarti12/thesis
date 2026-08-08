"""
Expansión de abreviaturas/siglas - idea nueva del ciclo 2026-07-21 para
atacar específicamente los casos de "retrieval miss" (gold fuera del
top-5) que resultaron ser abreviaturas (p.ej. "EEG" -> gold "brainwave
recording", "evs" -> gold "European Values Study", ver
EXPERIMENTS_LOG.md). A diferencia de los intentos ya descartados de
ensanchar el pool o mezclar sugerencias directas del LLM como CANDIDATOS
(que tocaban TODAS las filas y metían distractores), esto:

1. Solo se aplica a términos que YA parecen una abreviatura
   (text_utils.is_abbreviation_like) - el resto de términos no se toca en
   absoluto, cero riesgo de degradar los casos que ya funcionan.
2. La expansión se usa como una QUERY MÁS para el retrieval por embeddings
   (junto al término original y las variantes del generador), no como
   candidato directo - sigue sin haber invención de respuestas, solo
   mejora el texto de búsqueda para términos donde el término crudo
   ("EEG") es un mal match léxico/semántico para el candidato correcto
   ("brainwave recording").

Corre en local vía Ollama (gratis) - es un paso de preprocesado barato,
no necesita el modelo fuerte de pago.
"""

from ollama_client import llamar_modelo

AGENTE_ABREVIATURA = {
    "modelo": "llama3.2:3b",
    "instrucciones": (
        "You are a domain terminology expert. Given a short term that may "
        "be an abbreviation or acronym, expand it to its most likely full "
        "form. If it is not an abbreviation, or you don't know a plausible "
        "expansion, return the term unchanged. Return ONLY the expanded "
        "term (or the original term), nothing else. Do not explain."
    ),
    "temperature": 0.0,
}


def expandir_abreviatura(term: str, agente: dict = AGENTE_ABREVIATURA) -> str | None:
    """Devuelve la expansión propuesta por el LLM, o None si la llamada
    falla o la respuesta viene vacía (el caller decide el fallback, p.ej.
    no añadir ninguna query extra)."""
    try:
        respuesta = llamar_modelo(agente, f'Term: "{term}"')
        expansion = respuesta.strip().strip('"').strip()
        return expansion or None
    except Exception:
        return None


if __name__ == "__main__":
    for term in ["EEG", "EVS", "GDP", "recommender system"]:
        print(f"{term!r} -> {expandir_abreviatura(term)!r}")
