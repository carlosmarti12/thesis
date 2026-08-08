"""
Métricas de evaluación para la tarea actual (open synonym discovery: un
único término correcto en en_synonym por fila). Cada método devuelve una
lista ordenada de candidatos; se compara contra ese único ground truth.

hit@k y ndcg@k reemplazan a exact_top1 / top_k_accuracy usadas en
iteraciones anteriores (new/, synonic/) - mismo cálculo, nomenclatura de
information retrieval estándar para que sea comparable con literatura.
"""

import math

from .text_utils import normalize


def _rank_of_gold(ground_truth: str, candidates: list[str]) -> int | None:
    """Posición (1-indexada) del ground truth en candidates, o None si no
    aparece."""
    norm_gt = normalize(ground_truth)
    for idx, candidate in enumerate(candidates, start=1):
        if normalize(candidate) == norm_gt:
            return idx
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    """True si el ground truth aparece en las primeras k posiciones."""
    return rank is not None and rank <= k


def ndcg_at_k(rank: int | None, k: int) -> float:
    """NDCG@k con relevancia binaria y un único item relevante.

    DCG@k = 1/log2(rank+1) si el item relevante cae dentro de las primeras
    k posiciones, si no 0. IDCG@k (orden ideal: el relevante en la posición
    1) = 1/log2(2) = 1, así que NDCG@k se reduce a 1/log2(rank+1) cuando
    rank <= k, y 0 en otro caso - penaliza gradualmente según cuánto se
    aleja el acierto de la posición 1, a diferencia de hit@k que es binario.
    """
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def compute_metrics(ground_truth: str, candidates: list[str]) -> dict:
    rank = _rank_of_gold(ground_truth, candidates)

    return {
        "hit@1": hit_at_k(rank, 1),
        "hit@3": hit_at_k(rank, 3),
        "hit@5": hit_at_k(rank, 5),
        "ndcg@3": ndcg_at_k(rank, 3),
        "ndcg@5": ndcg_at_k(rank, 5),
        "mrr": (1.0 / rank) if rank is not None else 0.0,
        "rank": rank if rank is not None else -1,
    }


def summarize_results(results_df, method: str) -> dict:
    n = len(results_df)
    return {
        "method": method,
        "rows": n,
        "hit@1": float(results_df["hit@1"].mean()),
        "hit@3": float(results_df["hit@3"].mean()),
        "hit@5": float(results_df["hit@5"].mean()),
        "n_hit@1": int(results_df["hit@1"].sum()),
        "n_hit@3": int(results_df["hit@3"].sum()),
        "n_hit@5": int(results_df["hit@5"].sum()),
        "ndcg@3": float(results_df["ndcg@3"].mean()),
        "ndcg@5": float(results_df["ndcg@5"].mean()),
        "mrr": float(results_df["mrr"].mean()),
        "avg_time_seconds": float(results_df["time_seconds"].mean()),
    }
