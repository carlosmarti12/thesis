"""
Reciprocal Rank Fusion (RRF) - combina varias listas rankeadas (p.ej.
embeddings semánticos + BM25 léxico) sin necesitar recalibrar sus scores a
la misma escala, que es justo el problema al mezclar coseno (embeddings) con
BM25 (léxico): sus rangos numéricos no son comparables directamente, pero
sus RANKINGS sí. Referencia: Cormack, Clarke & Buettcher (2009), ya citada
en REFERENCES.md para este mismo propósito.

RRF(d) = sum_por_lista 1 / (k + rank_en_esa_lista(d)), rank empieza en 1.
Un documento que no aparece en una lista simplemente no suma nada de esa
lista (no penaliza, solo no aporta).
"""

from ..text_utils import normalize


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 5,
) -> list[str]:
    """
    ranked_lists: cada elemento es una lista de candidatos ya ordenada
    (mejor primero) de una fuente distinta (p.ej. [emb_ranked, bm25_ranked]).
    weights: peso opcional por lista (default 1.0 cada una) - permite dar
    más importancia a una señal que a otra sin romper la escala RRF.
    """
    if not ranked_lists:
        return []

    weights = weights or [1.0] * len(ranked_lists)
    assert len(weights) == len(ranked_lists)

    fused_score: dict[str, float] = {}
    display_by_norm: dict[str, str] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, candidate in enumerate(ranked_list, start=1):
            norm_c = normalize(candidate)
            if not norm_c:
                continue
            display_by_norm.setdefault(norm_c, candidate)
            fused_score[norm_c] = fused_score.get(norm_c, 0.0) + weight / (k + rank)

    ranked_norms = sorted(fused_score.keys(), key=lambda n: fused_score[n], reverse=True)
    return [display_by_norm[n] for n in ranked_norms[:top_k]]


def weighted_score_fusion(
    bm25_scored: list[tuple[str, float]],
    emb_scored: list[tuple[str, float]],
    bm25_weight: float = 1.0,
    emb_weight: float = 1.0,
    top_k: int = 5,
) -> list[str]:
    """
    Fusión por SCORE (no por rango como RRF) - min-max normaliza cada fuente
    a [0, 1] por separado (los scores BM25 crudos y el coseno de embeddings
    no están en la misma escala) y combina con una suma ponderada. A
    diferencia de RRF, esto conserva CUÁNTO mejor es un candidato que el
    siguiente dentro de cada fuente, no solo su posición - no probado antes
    para BM25+e5 (solo se había probado RRF, con varios pesos, y perdió
    siempre frente a e5 solo).
    """
    def _minmax(scored: list[tuple[str, float]]) -> dict[str, float]:
        if not scored:
            return {}
        values = [s for _, s in scored]
        lo, hi = min(values), max(values)
        span = hi - lo
        if span == 0:
            return {normalize(c): 1.0 for c, _ in scored}
        return {normalize(c): (s - lo) / span for c, s in scored}

    bm25_norm = _minmax(bm25_scored)
    emb_norm = _minmax(emb_scored)

    display_by_norm: dict[str, str] = {}
    for c, _ in bm25_scored:
        display_by_norm.setdefault(normalize(c), c)
    for c, _ in emb_scored:
        display_by_norm.setdefault(normalize(c), c)

    all_norms = set(bm25_norm) | set(emb_norm)
    fused_score = {
        n: bm25_weight * bm25_norm.get(n, 0.0) + emb_weight * emb_norm.get(n, 0.0)
        for n in all_norms
    }

    ranked_norms = sorted(fused_score.keys(), key=lambda n: fused_score[n], reverse=True)
    return [display_by_norm[n] for n in ranked_norms[:top_k]]
