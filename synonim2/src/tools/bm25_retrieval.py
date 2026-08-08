"""
Herramienta de retrieval léxico (BM25) sobre el mismo pool de candidatos que
embedding_retrieval.py. Complementa a los embeddings semánticos: BM25 premia
solapamiento de tokens/subpalabras, por lo que puede rescatar variantes
morfológicas, abreviaturas o términos compuestos donde el embedder falla por
tokenizar la frase completa de forma demasiado global (p.ej. "ADHD" vs
"attention deficit hyperactivity disorder", o "accountancy" vs "accounting").
No depende de ningún LLM ni de sentence-transformers.
"""

import numpy as np
from rank_bm25 import BM25Okapi

from ..text_utils import normalize


def _tokenize(text: str) -> list[str]:
    return normalize(text).split()


def build_bm25_index(pool_terms: list[str]) -> tuple[list[str], BM25Okapi]:
    """Deduplica (mismo criterio que embedding_retrieval.build_solution_index)
    y construye un índice BM25 sobre el pool tokenizado."""
    seen_norm = set()
    terms = []
    for raw in pool_terms:
        text = str(raw)
        norm = normalize(text)
        if norm and norm not in seen_norm:
            seen_norm.add(norm)
            terms.append(text)

    tokenized = [_tokenize(t) for t in terms]
    bm25 = BM25Okapi(tokenized)
    return terms, bm25


def retrieve_bm25_scored(
    query: str,
    index_terms: list[str],
    bm25: BM25Okapi,
    top_n: int = 20,
    exclude_terms: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Devuelve las top_n parejas (candidato, score BM25) para una única
    query. A diferencia de embedding_retrieval, no se hace agregación
    multi-query aquí - BM25 se usa como señal complementaria de una sola
    query (el término original), no como motor de expansión de queries."""
    if not query or not index_terms:
        return []

    exclude_norms = {normalize(t) for t in (exclude_terms or [])}

    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = np.argsort(scores)[::-1]

    results = []
    for idx in ranked_indices:
        candidate = index_terms[idx]
        if normalize(candidate) in exclude_norms:
            continue
        score = float(scores[idx])
        if score <= 0:
            break  # el resto también será <=0 (orden descendente); no aportan señal
        results.append((candidate, score))
        if len(results) >= top_n:
            break

    return results
