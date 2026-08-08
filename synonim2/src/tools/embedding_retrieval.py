"""
Herramienta 2 - retrieval por embeddings.

Dada una lista de candidatos/queries y una lista de posibles soluciones
(pool), calcula embeddings de ambas con sentence-transformers y devuelve
las top_k soluciones del pool más cercanas semánticamente. La similitud de
cada solución del pool se agrega como el máximo (coseno) obtenido entre
todas las queries, con un pequeño bono por aparecer entre los mejores
resultados de varias queries distintas.

No depende de ningún LLM ni de este dataset concreto - solo necesita dos
listas de texto - para poder reutilizarla en otros proyectos, tal y como
se pidió. Mismo patrón de agregación que
new/src2/run_llm_expansion_retrieval.py::expansion_retrieval_baseline /
synonic/src/baselines.py::llm_expansion_baseline, pero extraído aquí como
pieza independiente en vez de estar empotrado en un baseline.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ..text_utils import normalize

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"

# Algunos embedders (familia e5/bge) se entrenaron con prefijos distintos
# para "query" (lo que se busca) y "passage"/"documento" (lo que se indexa) -
# usarlos sin el prefijo correcto degrada notablemente su calidad respecto a
# como fueron entrenados. all-MiniLM-L6-v2 (el default de este proyecto) no
# usa prefijos. Mapeo explícito por nombre de modelo en vez de adivinar.
EMBEDDER_PREFIXES = {
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "BAAI/bge-base-en-v1.5": (
        "Represent this sentence for searching relevant passages: ", "",
    ),
    "BAAI/bge-small-en-v1.5": (
        "Represent this sentence for searching relevant passages: ", "",
    ),
    # gte-base-en-v1.5 no usa prefijos de query/passage (entrenado
    # simétricamente, a diferencia de e5/bge).
    "Alibaba-NLP/gte-base-en-v1.5": ("", ""),
    # mxbai-embed-large-v1 solo usa prefijo en la query, no en los
    # documentos (mismo patrón asimétrico que bge).
    "mixedbread-ai/mxbai-embed-large-v1": (
        "Represent this sentence for searching relevant passages: ", "",
    ),
}

# Modelos que requieren trust_remote_code=True para cargar su código de
# modelado personalizado (gte-base-en-v1.5 usa atención rotatoria/unpadding
# no incluida en transformers estándar).
TRUST_REMOTE_CODE_EMBEDDERS = {"Alibaba-NLP/gte-base-en-v1.5"}


def _query_prefix(embedder_name: str) -> str:
    return EMBEDDER_PREFIXES.get(embedder_name, ("", ""))[0]


def _passage_prefix(embedder_name: str) -> str:
    return EMBEDDER_PREFIXES.get(embedder_name, ("", ""))[1]


@lru_cache(maxsize=4)
def get_embedder(model_name: str = EMBEDDER_NAME) -> SentenceTransformer:
    """Instancia cacheada del embedder - cargar el modelo es lo caro, no
    usarlo, así que se reutiliza entre llamadas dentro del mismo proceso.
    maxsize=4 (en vez de 1) para poder comparar varios embedders en el mismo
    proceso de evaluación sin recargar cada vez que se alterna entre ellos."""
    trust_remote_code = model_name in TRUST_REMOTE_CODE_EMBEDDERS
    return SentenceTransformer(model_name, trust_remote_code=trust_remote_code)


def build_solution_index(
    pool_terms: list[str],
    embedder: SentenceTransformer | None = None,
    embedder_name: str = EMBEDDER_NAME,
) -> tuple[list[str], np.ndarray]:
    """
    Deduplica (por texto normalizado) y embebe la lista de posibles
    soluciones. Este índice se calcula UNA VEZ por lista de soluciones y
    se reutiliza para cada término procesado - es la parte cara de este
    módulo, igual que el índice de WordNet en synonic.

    Aplica el prefijo "passage"/"documento" del embedder si aplica (ver
    EMBEDDER_PREFIXES) - necesario para que modelos e5/bge rindan como
    fueron entrenados.
    """
    embedder = embedder or get_embedder(embedder_name)

    seen_norm = set()
    terms = []
    for raw in pool_terms:
        text = str(raw)
        norm = normalize(text)
        if norm and norm not in seen_norm:
            seen_norm.add(norm)
            terms.append(text)

    prefix = _passage_prefix(embedder_name)
    to_encode = [prefix + t for t in terms] if prefix else terms
    embeddings = embedder.encode(to_encode, normalize_embeddings=True, show_progress_bar=False)
    return terms, embeddings


def get_solution_index(
    pool_terms: list[str],
    embedder: SentenceTransformer | None = None,
    embedder_name: str = EMBEDDER_NAME,
    cache_name: str = "solution_pool",
) -> tuple[list[str], np.ndarray]:
    """
    Cached wrapper around build_solution_index: embedding ~1000 pool terms
    only takes a couple seconds, but this still avoids repeating it on every
    run of run_experiment.py, and matters more if the pool grows later.
    Cached to disk under data/cache/, keyed by embedder name + pool size
    (same pattern as synonic/src/open_vocab.py::get_wordnet_index) - if the
    dataset changes row count, the cache key changes and it rebuilds.
    """
    safe_name = hashlib.sha1(embedder_name.encode("utf-8")).hexdigest()[:10]

    # Dedup count first (cheap) so the cache key matches what build_solution_index
    # will actually produce, without needing to embed anything yet.
    seen_norm = set()
    for raw in pool_terms:
        norm = normalize(str(raw))
        if norm:
            seen_norm.add(norm)
    n_terms = len(seen_norm)

    emb_path = CACHE_DIR / f"{cache_name}_{safe_name}_{n_terms}.npy"
    terms_path = CACHE_DIR / f"{cache_name}_{safe_name}_{n_terms}_terms.json"

    if emb_path.exists() and terms_path.exists():
        embeddings = np.load(emb_path)
        with open(terms_path, "r", encoding="utf-8") as f:
            terms = json.load(f)
        return terms, embeddings

    embedder = embedder or get_embedder(embedder_name)
    terms, embeddings = build_solution_index(pool_terms, embedder=embedder, embedder_name=embedder_name)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    with open(terms_path, "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False)

    return terms, embeddings


def encode_queries(
    queries: list[str],
    embedder: SentenceTransformer,
    embedder_name: str = EMBEDDER_NAME,
) -> np.ndarray:
    """Embebe una lista de queries aplicando el prefijo "query: " del
    embedder si aplica. Expuesto por separado (no solo dentro de
    retrieve_top_k) para que multi_query_fusion.py pueda construir su propia
    matriz de scores completa (query x pool) y probar varias estrategias de
    combinación sobre la MISMA matriz, en vez de recalcular embeddings por
    estrategia."""
    prefix = _query_prefix(embedder_name)
    to_encode = [prefix + q for q in queries] if prefix else queries
    return embedder.encode(to_encode, normalize_embeddings=True, show_progress_bar=False)


def _aggregate_scores(
    queries: list[str],
    index_terms: list[str],
    index_embeddings: np.ndarray,
    embedder: SentenceTransformer,
    embedder_name: str,
    per_query_k: int,
    exclude_terms: list[str] | None,
    multi_hit_bonus: float,
) -> list[tuple[str, float]]:
    """Lógica de agregación compartida por retrieve_top_k y retrieve_scored:
    embebe las queries, compara contra el índice, agrega por máximo + bono
    de multi-hit, y devuelve TODOS los candidatos vistos ordenados
    (candidato, score) - el caller decide cuántos cortar."""
    exclude_norms = {normalize(t) for t in (exclude_terms or [])}

    prefix = _query_prefix(embedder_name)
    to_encode = [prefix + q for q in queries] if prefix else queries
    query_embeddings = embedder.encode(to_encode, normalize_embeddings=True, show_progress_bar=False)
    scores = np.dot(query_embeddings, index_embeddings.T)  # (n_queries, n_index)

    best_score: dict[str, float] = {}
    hit_count: dict[str, int] = {}
    display_by_norm: dict[str, str] = {}

    for query_scores in scores:
        ranked_indices = np.argsort(query_scores)[::-1][:per_query_k]

        for idx in ranked_indices:
            candidate = index_terms[idx]
            norm_candidate = normalize(candidate)

            if not norm_candidate or norm_candidate in exclude_norms:
                continue

            score = float(query_scores[idx])
            display_by_norm.setdefault(norm_candidate, candidate)

            if norm_candidate not in best_score or score > best_score[norm_candidate]:
                best_score[norm_candidate] = score

            hit_count[norm_candidate] = hit_count.get(norm_candidate, 0) + 1

    ranked_norms = sorted(
        best_score.keys(),
        key=lambda n: best_score[n] + multi_hit_bonus * (hit_count[n] - 1),
        reverse=True,
    )

    return [(display_by_norm[n], best_score[n]) for n in ranked_norms]


def retrieve_top_k(
    queries: list[str],
    index_terms: list[str],
    index_embeddings: np.ndarray,
    embedder: SentenceTransformer | None = None,
    embedder_name: str = EMBEDDER_NAME,
    top_k: int = 5,
    per_query_k: int = 10,
    exclude_terms: list[str] | None = None,
    multi_hit_bonus: float = 0.05,
) -> list[str]:
    """
    Embebe cada query y la compara (producto punto sobre embeddings
    normalizados == coseno) contra el índice de soluciones. Devuelve los
    top_k candidatos del pool con mayor score agregado.
    """
    if not queries or index_embeddings is None or len(index_terms) == 0:
        return []

    embedder = embedder or get_embedder(embedder_name)
    ranked = _aggregate_scores(
        queries, index_terms, index_embeddings, embedder, embedder_name,
        per_query_k, exclude_terms, multi_hit_bonus,
    )
    return [c for c, _ in ranked[:top_k]]


def retrieve_scored(
    queries: list[str],
    index_terms: list[str],
    index_embeddings: np.ndarray,
    embedder: SentenceTransformer | None = None,
    embedder_name: str = EMBEDDER_NAME,
    top_n: int = 20,
    per_query_k: int = 10,
    exclude_terms: list[str] | None = None,
    multi_hit_bonus: float = 0.05,
) -> list[tuple[str, float]]:
    """
    Igual que retrieve_top_k pero devuelve las top_n parejas (candidato,
    score) en vez de solo los nombres - pensado para alimentar pasos
    posteriores (fusión con BM25, reranking por cross-encoder) que necesitan
    más candidatos y sus scores, no solo el top_k final.
    """
    if not queries or index_embeddings is None or len(index_terms) == 0:
        return []

    embedder = embedder or get_embedder(embedder_name)
    ranked = _aggregate_scores(
        queries, index_terms, index_embeddings, embedder, embedder_name,
        per_query_k, exclude_terms, multi_hit_bonus,
    )
    return ranked[:top_n]


if __name__ == "__main__":
    pool = ["shares", "stock market", "bonds", "recommendation engine", "financial accounting"]
    index_terms, index_embeddings = build_solution_index(pool)

    top5 = retrieve_top_k(["stocks", "equity"], index_terms, index_embeddings, top_k=3)
    print("Top:", top5)
