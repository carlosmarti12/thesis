"""
Non-agentic baseline methods for open synonym discovery.

Every function here takes ONLY the input `term` (plus shared, term-independent
resources like an embedder or a prebuilt WordNet index) and returns a ranked
list of candidate strings. None of them receive the dataframe or
`en_synonym` - that column is read for the first time in evaluation.py's
metrics step, in run_experiment.py, strictly after a method has already
returned its candidates. This is the fix for the leakage found in
new/src2, where `build_output_vector_index` built its search index directly
from df['en_synonym'].
"""

import numpy as np

from .evaluation import normalize
from .llm import ask_llm, parse_candidates


def wordnet_direct_baseline(term: str, max_words: int = 4) -> list[str]: # esta funcion busca en WordNet todos los sinónimos de un término dado y devuelve una lista de ellos, excluyendo el término original y limitando el número de palabras en los sinónimos a max_words
    """
    Direct WordNet synonym lookup: find every synset that contains `term` as
    one of its lemmas, and return the OTHER lemma names in those synsets.
    Pure symbolic method, no ML, no dataset vocabulary of any kind - a
    classical NLP baseline for comparison against embedding/LLM methods.
    """
    from nltk.corpus import wordnet as wn

    query = term.strip().lower().replace(" ", "_")
    norm_term = normalize(term)

    seen = set()
    results = []

    for synset in wn.synsets(query):
        for lemma in synset.lemmas():
            phrase = lemma.name().replace("_", " ")
            if len(phrase.split()) > max_words:
                continue
            norm_phrase = normalize(phrase)
            if norm_phrase == norm_term or not norm_phrase:
                continue
            if norm_phrase not in seen:
                seen.add(norm_phrase)
                results.append(phrase)

    return results


def embedding_wordnet_baseline( # esta funcion toma un termino, un embedder, una lista de terminos indexados y sus embeddings, y devuelve los top_k terminos mas cercanos al termino dado usando embeddings y busqueda en el indice de WordNet
    term: str,
    embedder,
    index_terms: list[str],
    index_embeddings,
    top_k: int = 5,
) -> list[str]:
    """
    Encode `term` and retrieve the top_k nearest neighbors from the
    precomputed WordNet vocabulary index (see open_vocab.get_wordnet_index).
    Embeddings are L2-normalized so dot product == cosine similarity.
    """
    term_emb = embedder.encode([term], normalize_embeddings=True)[0]
    scores = np.dot(index_embeddings, term_emb)
    ranked_indices = np.argsort(scores)[::-1]

    norm_term = normalize(term)
    results = []
    seen = set()

    for idx in ranked_indices:
        candidate = index_terms[idx]
        norm_candidate = normalize(candidate)

        if norm_candidate == norm_term:
            continue

        if norm_candidate not in seen:
            results.append(candidate)
            seen.add(norm_candidate)

        if len(results) >= top_k:
            break 

    return results


def llm_zero_shot_baseline(term: str, model: str | None = None, top_k: int = 5) -> list[str]: # esat funcion pregunta directamente a un modelo de lenguaje por sinónimos de un término dado, sin recuperación involucrada, y devuelve los top_k candidatos
    """Ask the LLM directly for synonyms of `term`, no retrieval involved."""
    system_prompt = (
        "You are a domain terminology expert. "
        "Your task is to propose English synonyms for a given academic/domain term. "
        f"Return only a JSON list of {top_k} short English candidate synonyms. "
        "Do not explain."
    )

    try:
        response = ask_llm(system_prompt, f'Term: "{term}"', model=model)
        candidates = parse_candidates(response)
    except Exception:
        candidates = []

    return candidates[:top_k]


def _generate_query_variants(term: str, model: str | None, n: int = 6) -> list[str]: # esta funcion genera generar variantes del término. devuelve una lista de variantes cortas del término dado, que pueden ser sinónimos o variantes semánticas cercanas, para ser usadas como consultas de búsqueda
    system_prompt = (
        "You are a domain terminology expert. "
        "Generate English synonyms or closely related semantic variants "
        "for the given academic/domain term. These variants will be used "
        "as search queries against a vocabulary index, so include "
        "broader, narrower, and near-equivalent phrasings. "
        f"Return only a JSON list of {n} short variants. "
        "Do not explain."
    )

    try:
        response = ask_llm(system_prompt, f'Term: "{term}"', model=model)
        variants = parse_candidates(response)[:n]
    except Exception:
        variants = []

    if not variants:
        variants = [term]

    return variants


def llm_expansion_baseline( # esta funcion combina la generacion de variantes, te quedas con la mejor a paricion, de los diez que genera el llm, y que buscan en worndnet de todos los que buscan , y cuantas veces aparecen
    term: str,
    embedder,
    index_terms: list[str],
    index_embeddings,
    model: str | None = None,
    top_k: int = 5,
    n_variants: int = 6,
    per_variant_k: int = 10,
) -> list[str]:
    """
    Generative query expansion + embedding retrieval over the WordNet index:

    1. Ask the LLM for several synonym/semantic-variant queries of `term`.
    2. Embed each variant.
    3. Search each variant embedding against the precomputed WordNet vector
       index (never df['en_synonym']).
    4. Aggregate across variants: a candidate's score is the best (max)
       cosine similarity it achieved across variant queries, with a small
       bonus for appearing in multiple variants' top-k results.
    5. Return the top-k aggregated candidates.
    """
    variants = _generate_query_variants(term, model, n=n_variants) # generas variantes del término usando el llm

    variant_embeddings = embedder.encode(variants, normalize_embeddings=True) # creas embeding para cada variable generada
    scores = np.dot(variant_embeddings, index_embeddings.T) # comparamos con worndet index embeddings para obtener scores de similitud

    norm_term = normalize(term)
    best_score = {}
    hit_count = {}

    for variant_scores in scores:
        ranked_indices = np.argsort(variant_scores)[::-1][:per_variant_k] # por cada variante generada cogemos los top_k candidatos de la busqueda en el indice de WordNet

        for idx in ranked_indices:
            candidate = index_terms[idx]
            norm_candidate = normalize(candidate)

            if norm_candidate == norm_term:
                continue

            score = float(variant_scores[idx])

            if norm_candidate not in best_score or score > best_score[norm_candidate]:
                best_score[norm_candidate] = score

            hit_count[norm_candidate] = hit_count.get(norm_candidate, 0) + 1

    display_by_norm = {}
    for term_text in index_terms:
        norm_candidate = normalize(term_text)
        if norm_candidate in best_score and norm_candidate not in display_by_norm:
            display_by_norm[norm_candidate] = term_text

    ranked_norms = sorted(
        best_score.keys(),
        key=lambda n: best_score[n] + 0.05 * (hit_count[n] - 1),
        reverse=True,
    )

    return [display_by_norm[n] for n in ranked_norms[:top_k]]
    # la bonificación de 0.05 * (hit_count[n] - 1) recompensa a los candidatos que aparecen en múltiples variantes, pero no es demasiado grande para que un candidato con una sola aparición y una puntuación alta supere a uno con varias apariciones y una puntuación más baja.


def llm_generate_rerank_baseline( # generamos candidatos con llm, luego los reordenamos por similitud de embedding con el término original, y devolvemos los top_k candidatos
    term: str,
    embedder,
    model: str | None = None,
    n_generate: int = 10,
    top_k: int = 5,
) -> list[str]:
    """
    Ask the LLM for `n_generate` candidate synonyms, then rerank them
    numerically instead of trusting the LLM's own order:

    1. Ask the LLM directly for `n_generate` candidates of `term` (no
       retrieval, no vocabulary - just generation).
    2. Embed the original `term` and all `n_generate` candidates with the
       same embedder.
    3. Score each candidate by cosine similarity to the term's embedding
       (embeddings are L2-normalized, so cosine similarity = dot product).
    4. Keep the `top_k` candidates with the highest similarity.

    Unlike llm_zero_shot_baseline (which just takes the LLM's own top-5 as
    given) or llm_expansion_baseline (which uses the LLM to generate search
    *queries* against the WordNet index), this method never touches
    WordNet - the only candidate pool is whatever the LLM itself proposed,
    and only the ranking is numeric.
    """
    system_prompt = (
        "You are a domain terminology expert. "
        "Generate English synonyms or near-equivalent academic/domain terms. "
        f"Return only a JSON list of {n_generate} candidates. "
        "Do not explain."
    )

    try:
        response = ask_llm(system_prompt, f'Term: "{term}"', model=model)
        candidates = parse_candidates(response)[:n_generate]
    except Exception:
        candidates = []

    if not candidates:
        return []

    texts = [term] + candidates
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    term_emb = embeddings[0]
    cand_embs = embeddings[1:]
    scores = np.dot(cand_embs, term_emb)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

    norm_term = normalize(term)
    results = []
    seen = set()
    for candidate, _ in ranked:
        norm_candidate = normalize(candidate)
        if norm_candidate == norm_term or norm_candidate in seen:
            continue
        results.append(candidate)
        seen.add(norm_candidate)
        if len(results) >= top_k:
            break

    return results


def hybrid_fusion_baseline( # 
    term: str,
    embedder,
    index_terms: list[str],
    index_embeddings,
    model: str | None = None,
    top_k: int = 5,
    n_samples: int = 3,
    temperature: float = 0.7,
    n_generate: int = 10,
    pool_k: int = 10,
    wordnet_k: int = 10,
    rrf_k: int = 60,
    weight_llm: float = 1.0,
    weight_wordnet: float = 0.5,
) -> list[str]:
    """
    Self-consistency LLM generation, fused with WordNet retrieval via weighted
    Reciprocal Rank Fusion (RRF) instead of either constraining every candidate
    to the WordNet vocabulary (like the MAS variants) or trusting a single LLM
    call (like llm_zero_shot_baseline/llm_generate_rerank_baseline):

    1. Ask the LLM for n_generate candidates, n_samples times at temperature>0
       (each call in its own try/except so one bad call doesn't wipe the pool).
    2. Pool + dedupe the n_samples responses, rerank by cosine similarity to
       the term -> list A (LLM-generation signal).
    3. If the LLM pool is entirely empty (every call failed), fall back
       directly to embedding_wordnet_baseline instead of returning [].
    4. Retrieve list B from the WordNet index via embedding_wordnet_baseline
       (retrieval signal, always available since it needs no LLM).
    5. Fuse A and B with weighted RRF - weight_llm > weight_wordnet because
       free-form LLM generation empirically outperforms WordNet-constrained
       retrieval on this dataset's academic-phrase gold answers, so an
       unweighted RRF would let the weaker signal dilute the stronger one.
    """
    system_prompt = (
        "You are a domain terminology expert. "
        "Generate English synonyms or near-equivalent academic/domain terms. "
        f"Return only a JSON list of {n_generate} candidates. "
        "Do not explain."
    )

    norm_term = normalize(term)
    pooled: dict[str, str] = {}

    for _ in range(n_samples): # hacemos tres llamadas al LLM para generar candidatos, y los vamos agregando a un diccionario para evitar duplicados
        try:
            response = ask_llm(system_prompt, f'Term: "{term}"', model=model, temperature=temperature)
            for candidate in parse_candidates(response)[:n_generate]:
                norm_candidate = normalize(candidate)
                if norm_candidate and norm_candidate != norm_term:
                    pooled.setdefault(norm_candidate, candidate)
        except Exception:
            continue

    if not pooled:
        return embedding_wordnet_baseline(term, embedder, index_terms, index_embeddings, top_k=top_k)

    candidates = list(pooled.values())
    texts = [term] + candidates
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    scores = np.dot(embeddings[1:], embeddings[0]) # Los candidatos generados se reordenan mediante embeddings:
    list_a = [c for c, _ in sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)][:pool_k]

    list_b = embedding_wordnet_baseline(term, embedder, index_terms, index_embeddings, top_k=wordnet_k) # crealista con embeding y comparando con worndet

    rrf_scores: dict[str, float] = {} #vamos a combinar mediante : Reciprocal Rank Fusion
    display_by_norm: dict[str, str] = {}
    for candidate_list, weight in ((list_a, weight_llm), (list_b, weight_wordnet)):
        for rank, candidate in enumerate(candidate_list, start=1):
            norm_candidate = normalize(candidate)
            if not norm_candidate or norm_candidate == norm_term:
                continue
            display_by_norm.setdefault(norm_candidate, candidate)
            rrf_scores[norm_candidate] = rrf_scores.get(norm_candidate, 0.0) + weight / (rrf_k + rank) # weight_llm = 1.0 weight_wordnet = 0.5 , k es constante y rak es donde esta en la lista

    ranked = sorted(rrf_scores, key=lambda n: rrf_scores[n], reverse=True)
    return [display_by_norm[n] for n in ranked[:top_k]]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    term = "stocks"
    print("wordnet_direct:", wordnet_direct_baseline(term))

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("llm_generate_rerank:", llm_generate_rerank_baseline(term, embedder))
