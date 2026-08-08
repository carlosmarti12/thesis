"""
Combinación de rankings para retrieval multi-query (Método 1 pedido por el
usuario: LLM genera variantes del término -> se recupera para cada variante
-> se combinan los rankings). Cuatro estrategias, todas operando sobre la
MISMA matriz de scores (queries x pool) para que comparar entre ellas no
introduzca ninguna diferencia salvo la fórmula de combinación:

- "max": el score de cada candidato es el máximo coseno visto entre todas
  las queries (candidato favorecido si CUALQUIER query lo acerca mucho,
  aunque sea con una sola). Es, sin el bono de multi-hit, lo que ya hace
  embedding_retrieval.retrieve_top_k.
- "mean": el score es el promedio sobre todas las queries (penaliza
  candidatos que solo una query acerca y el resto ignora - más conservador
  que max).
- "weighted": promedio ponderado, dando más peso a la query del término
  ORIGINAL que a las variantes generadas por el LLM (razonamiento: el
  término original es la entrada real, las variantes son parafraseos
  potencialmente ruidosos - no deberían poder "ganarle" al término original
  solo por casualidad estadística de estar más cerca de un candidato
  incorrecto).
- "rrf": Reciprocal Rank Fusion (Cormack et al. 2009, ya citado en
  REFERENCES.md) - cada query aporta un ranking, se fusionan por rango en
  vez de por score bruto. Reutiliza fusion.reciprocal_rank_fusion.

`query_embeddings` DEBE tener el término original en la posición 0 - lo
exige generar_candidatos() (ver src/agents/generator_agent.py), que siempre
antepone el término original a las variantes generadas.
"""

import numpy as np

from ..text_utils import normalize
from .fusion import reciprocal_rank_fusion

FUSION_METHODS = ("max", "mean", "weighted", "rrf")


def combine_scores(
    fusion_method: str,
    query_embeddings: np.ndarray,
    index_terms: list[str],
    index_embeddings: np.ndarray,
    exclude_terms: list[str] | None = None,
    top_k: int = 5,
    original_weight: float = 3.0,
    other_weight: float = 1.0,
    rrf_k: int = 60,
    rrf_top_n_per_query: int = 20,
) -> list[str]:
    if fusion_method not in FUSION_METHODS:
        raise ValueError(f"Unknown fusion method: {fusion_method!r} (expected one of {FUSION_METHODS})")

    exclude_norms = {normalize(t) for t in (exclude_terms or [])}
    scores = np.dot(query_embeddings, index_embeddings.T)  # (n_queries, n_index)

    if fusion_method == "rrf":
        ranked_lists = []
        for row in scores:
            order = np.argsort(row)[::-1][:rrf_top_n_per_query]
            ranked_lists.append([
                index_terms[i] for i in order if normalize(index_terms[i]) not in exclude_norms
            ])
        return reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_k=top_k)

    if fusion_method == "max":
        agg = scores.max(axis=0)
    elif fusion_method == "mean":
        agg = scores.mean(axis=0)
    else:  # weighted - query 0 es siempre el término original (ver docstring)
        n_queries = scores.shape[0]
        weights = np.full(n_queries, other_weight, dtype=float)
        weights[0] = original_weight
        agg = (scores * weights[:, None]).sum(axis=0) / weights.sum()

    order = np.argsort(agg)[::-1]
    result = []
    for i in order:
        if normalize(index_terms[i]) in exclude_norms:
            continue
        result.append(index_terms[i])
        if len(result) >= top_k:
            break
    return result


def combine_scores_scored(
    fusion_method: str,
    query_embeddings: np.ndarray,
    index_terms: list[str],
    index_embeddings: np.ndarray,
    exclude_terms: list[str] | None = None,
    top_k: int = 5,
    original_weight: float = 3.0,
    other_weight: float = 1.0,
    rrf_k: int = 60,
    rrf_top_n_per_query: int = 20,
) -> tuple[list[str], float]:
    """
    Igual que combine_scores, pero además devuelve el MARGEN entre el rank1
    y el rank2 del top-k (score_rank1 - score_rank2, entre los candidatos NO
    excluidos) - usado por el rerank condicionado por confianza (Fase C.5):
    un margen pequeño indica una decisión ambigua entre los dos mejores
    candidatos, donde vale la pena invertir en una segunda pasada de LLM;
    un margen grande indica que el top-1 ya es una elección clara, donde
    rerankear solo puede introducir ruido (razón por la que rerankear TODO
    el top-5, sin condicionar, no dio beneficio en los métodos 2 y 3 - ver
    EXPERIMENTS_LOG.md). Para "rrf" el margen se calcula sobre el score RRF
    fusionado, no sobre el coseno crudo (las escalas no son comparables).
    """
    top_k_terms = combine_scores(
        fusion_method, query_embeddings, index_terms, index_embeddings,
        exclude_terms=exclude_terms, top_k=max(top_k, 2), original_weight=original_weight,
        other_weight=other_weight, rrf_k=rrf_k, rrf_top_n_per_query=rrf_top_n_per_query,
    )

    exclude_norms = {normalize(t) for t in (exclude_terms or [])}
    scores = np.dot(query_embeddings, index_embeddings.T)

    if fusion_method == "rrf":
        ranked_lists = []
        for row in scores:
            order = np.argsort(row)[::-1][:rrf_top_n_per_query]
            ranked_lists.append([
                index_terms[i] for i in order if normalize(index_terms[i]) not in exclude_norms
            ])
        fused = {}
        for ranked_list in ranked_lists:
            for rank, candidate in enumerate(ranked_list, start=1):
                norm_c = normalize(candidate)
                fused[norm_c] = fused.get(norm_c, 0.0) + 1.0 / (rrf_k + rank)
        score_by_norm = fused
    else:
        if fusion_method == "max":
            agg = scores.max(axis=0)
        elif fusion_method == "mean":
            agg = scores.mean(axis=0)
        else:  # weighted
            n_queries = scores.shape[0]
            weights = np.full(n_queries, other_weight, dtype=float)
            weights[0] = original_weight
            agg = (scores * weights[:, None]).sum(axis=0) / weights.sum()
        score_by_norm = {normalize(index_terms[i]): float(agg[i]) for i in range(len(index_terms))}

    top2_scores = [score_by_norm.get(normalize(t), 0.0) for t in top_k_terms[:2]]
    margin = (top2_scores[0] - top2_scores[1]) if len(top2_scores) == 2 else float("inf")

    return top_k_terms[:top_k], margin
