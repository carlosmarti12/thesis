"""
MAS agent functions - each one is a standalone, directly-callable function
(no LangGraph state, no closures) so they can be tested or reused in
isolation. graph.py wires these into the three pipeline variants
(base / llm_ranker / safe_hybrid) using LangGraph nodes.

This is a node-for-node port of new/src2's run_mas_langgraph.py /
run_mas_v2_langgraph.py / run_mas_v3_safe_hybrid.py. The only substantive
change: retrieval_agent searches a precomputed WordNet vocabulary index
(open_vocab.get_wordnet_index) instead of an index built from
df['en_synonym'] - that was the leakage the advisor flagged. Every other
agent (prompts, merge/verification/ranking logic, score formulas) is
unchanged from the original.
"""

import numpy as np
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

from .evaluation import normalize
from .llm import ask_llm, parse_candidates


def generator_agent(term: str, model: str | None = None, n: int = 8) -> list[str]: # Este agente pide al LLM que genere candidatos a sinónimos directamente.
    """LLM proposes up to `n` candidate synonyms directly from the term."""
    system_prompt = (
        "You are a domain terminology expert. "
        "Generate English synonyms or near-equivalent academic/domain terms. "
        f"Return only a JSON list of {n} candidates. "
        "Do not explain."
    )

    try:
        response = ask_llm(system_prompt, f'Term: "{term}"', model=model)
        candidates = parse_candidates(response)
    except Exception:
        candidates = []

    return candidates[:n]


def retrieval_agent( # Este agente busca los términos más parecidos dentro del índice de WordNet. Compara el término de entrada con todas las palabras previamente almacenadas
    term: str,
    embedder,
    candidate_pool_terms: list[str],
    candidate_pool_embeddings,
    top_k: int = 8,
) -> list[str]:
    """Embed `term` and retrieve the top_k nearest neighbors from the
    precomputed WordNet candidate pool (never df['en_synonym'])."""
    term_emb = embedder.encode([term], normalize_embeddings=True) # creamos el embeding del termino 
    scores = cosine_similarity(term_emb, candidate_pool_embeddings)[0] # el termino y cada  uno de la lista
    ranked_indices = np.argsort(scores)[::-1]

    norm_term = normalize(term)
    results = []
    seen = set()

    for idx in ranked_indices:
        candidate = candidate_pool_terms[idx]
        norm_candidate = normalize(candidate)

        if norm_candidate == norm_term:
            continue

        if norm_candidate not in seen:
            results.append(candidate)
            seen.add(norm_candidate)

        if len(results) >= top_k:
            break

    return results # devuelce la lista de candidatos más cercanos al término de entrada, ordenados por similitud.


def merge_agent( # convina los candidatos generados y recuperados, eliminando duplicados y limitando la lista a un número máximo de candidatos
    term: str,
    generated: list[str],
    retrieved: list[str],
    order: str = "generated_first",
    cap: int | None = None,
) -> list[str]:
    """Deduplicate and combine generator + retrieval output. `order`
    controls which source is prioritized when both agree on a candidate's
    first occurrence; `cap` truncates the merged list (v2/v3 use cap=20 so
    the downstream LLM ranker doesn't see an unbounded list)."""
    seen = set()
    merged = []

    all_candidates = (
        generated + retrieved if order == "generated_first" else retrieved + generated
    )

    norm_term = normalize(term)

    for candidate in all_candidates:
        norm_candidate = normalize(candidate)
        if not norm_candidate or norm_candidate == norm_term:
            continue
        if norm_candidate not in seen:
            merged.append(candidate)
            seen.add(norm_candidate)

    if cap is not None:
        merged = merged[:cap]

    return merged


def verification_agent(term: str, candidates: list[str], embedder) -> list[dict]: # evalua cada candidato calculando una puntuación final basada en la similitud semántica y la similitud difusa, y devuelve una lista de diccionarios con los candidatos verificados y sus puntuaciones
    """v1 (base) only: score each merged candidate by
    0.75*semantic_similarity + 0.25*fuzzy_similarity."""
    if not candidates:
        return []

    texts = [term] + candidates # convinad termino y candidatos 
    embeddings = embedder.encode(texts, normalize_embeddings=True) # haces embeding de todos los terminos y candidatos
    term_emb = embeddings[0:1] # separas el embeding del termino de entrada
    cand_embs = embeddings[1:] # separas los embeding de los candidatos
    semantic_scores = cosine_similarity(term_emb, cand_embs)[0] # calculas la similitud coseno entre el embeding del termino y los embeding de los candidatos

    verified = []
    for candidate, semantic_score in zip(candidates, semantic_scores): # fuzz similarity compara la forma ttextual de los candidatos con el termino de entrada, y calcula una puntuación final basada en la similitud semántica y la similitud difusa
        fuzzy_score = fuzz.ratio(normalize(term), normalize(candidate)) / 100.0 # puntuacion de 0 a 100, dividida entre 100 para normalizar a 0-1
        final_score = 0.75 * float(semantic_score) + 0.25 * float(fuzzy_score) # calamo la puntuacion final 
        verified.append({ # creamos n dicionario con el candidato y sus puntuaciones
            "candidate": candidate,
            "semantic_score": float(semantic_score),
            "fuzzy_score": float(fuzzy_score),
            "final_score": float(final_score),
        })

    return verified


def ranking_agent(verified: list[dict], top_k: int = 5) -> list[str]: # Este agente recibe los resultados de verification_agent, los ordena y devuelve los mejores.
    """v1 (base) only: sort verified candidates by final_score, take top_k."""
    ranked = sorted(verified, key=lambda x: x["final_score"], reverse=True)
    return [x["candidate"] for x in ranked[:top_k]]


def llm_ranking_agent( # v2 (llm_ranker) / v3 (safe_hybrid) only: ask the LLM to order the merged candidate list.
    term: str,
    candidates: list[str],
    model: str | None = None,
    keep_all_via_append: bool = True,
) -> list[str]:
    """
    v2/v3: ask the LLM to order the merged candidate list.

    keep_all_via_append=True (v2 'llm_ranker'): appends any candidate the LLM
    omitted after its ordering, so a flaky ranker can't destroy recall by
    silently dropping valid candidates. Returns the final top-5 directly.

    keep_all_via_append=False (v3 'safe_hybrid'): returns only the LLM's
    valid, ordered subset (no top-5 cut, no append) - safe_finalizer_agent
    handles blending in anything the LLM left out.
    """
    if not candidates:
        return []

    candidate_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates)) # preparamos la lista para el pront

    system_prompt = (
        "You are a strict semantic ranking agent for domain terminology. "
        "You receive one input term and a list of candidate synonyms. "
        "Your task is to ORDER the candidates from best to worst according to semantic equivalence. "
        "Do not invent new candidates. "
        "Do not remove candidates unless they are completely invalid. "
        "Use the exact candidate strings from the provided list. "
        "Return only a JSON list of candidate strings ordered from best to worst."
    )
    user_prompt = f'Term: "{term}"\n\nCandidate synonyms:\n{candidate_text}'

    try:
        response = ask_llm(system_prompt, user_prompt, model=model)
        parsed = parse_candidates(response)
    except Exception:
        parsed = []

    original_by_norm = {normalize(c): c for c in candidates if normalize(c)}

    ordered = []
    seen = set()
    for item in parsed:
        norm_item = normalize(item)
        if norm_item in original_by_norm and norm_item not in seen:
            ordered.append(original_by_norm[norm_item])
            seen.add(norm_item)

    if keep_all_via_append:
        for candidate in candidates:
            norm_candidate = normalize(candidate)
            if norm_candidate and norm_candidate not in seen:
                ordered.append(candidate)
                seen.add(norm_candidate)
        return ordered[:5]

    return ordered


def safe_finalizer_agent(
    term: str,
    candidates: list[str],
    llm_ranked: list[str],
    embedder,
    top_k: int = 5,
) -> list[str]:
    """v3 (safe_hybrid) only: 0.9*embedding_sim + 0.1*llm_rank_bonus - the
    LLM ranking can only add a bonus, never penalize a candidate it left
    out, which protects recall from an unreliable LLM ranker."""
    if not candidates:
        return []

    texts = [term] + candidates
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    term_emb = embeddings[0:1]
    cand_embs = embeddings[1:]
    embedding_scores = cosine_similarity(term_emb, cand_embs)[0]

    llm_position = {normalize(c): rank for rank, c in enumerate(llm_ranked)}

    scored = []
    for candidate, emb_score in zip(candidates, embedding_scores):
        norm_candidate = normalize(candidate)
        llm_bonus = 1.0 / (llm_position[norm_candidate] + 1) if norm_candidate in llm_position else 0.0
        final_score = 0.90 * float(emb_score) + 0.10 * float(llm_bonus)
        scored.append({"candidate": candidate, "final_score": final_score})

    ranked = sorted(scored, key=lambda x: x["final_score"], reverse=True)

    final_candidates = []
    seen = set()
    for item in ranked:
        candidate = item["candidate"]
        norm_candidate = normalize(candidate)
        if norm_candidate not in seen:
            final_candidates.append(candidate)
            seen.add(norm_candidate)
        if len(final_candidates) >= top_k:
            break

    return final_candidates


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    from .open_vocab import get_wordnet_index

    term = "stocks"
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index_terms, index_embeddings = get_wordnet_index(embedder, "sentence-transformers/all-MiniLM-L6-v2")

    generated = generator_agent(term, n=8)
    print("Generated:", generated)

    retrieved = retrieval_agent(term, embedder, index_terms, index_embeddings, top_k=8)
    print("Retrieved:", retrieved)

    merged = merge_agent(term, generated, retrieved)
    print("Merged:", merged)

    verified = verification_agent(term, merged, embedder)
    print("Ranked (v1):", ranking_agent(verified))
