"""
Reranking con cross-encoder: a diferencia de un bi-encoder (embedding_retrieval.py,
que embebe query y candidato POR SEPARADO y compara por coseno), un
cross-encoder recibe el par (query, candidato) JUNTO y predice directamente
un score de relevancia/similitud - más caro por candidato (no se puede
precomputar un índice), pero más preciso porque el modelo puede atender a
ambas frases a la vez. Se usa aquí como segunda pasada sobre un conjunto ya
acotado (top-N del bi-encoder), nunca sobre el pool completo, para mantener
el coste manejable - patrón estándar "retrieve-then-rerank" en IR.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

CROSS_ENCODER_NAME = "cross-encoder/stsb-roberta-base"


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str = CROSS_ENCODER_NAME) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(
    term: str,
    candidates: list[str],
    cross_encoder: CrossEncoder | None = None,
    top_k: int = 5,
) -> list[str]:
    """Reordena `candidates` (ya recuperados por otro método) por score de
    similitud semántica par-a-par del cross-encoder, best-first. Si
    `candidates` está vacío devuelve []."""
    if not candidates:
        return []

    cross_encoder = cross_encoder or get_cross_encoder()
    pairs = [(term, c) for c in candidates]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
