import re
from typing import List, Dict, Any

from rapidfuzz import fuzz as fuzz_scorer
from rapidfuzz import process as fuzz_process
from sentence_transformers import SentenceTransformer, util

from src.llm import ask_llm, clean_list_response, is_refusal


_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def domain_inference_agent(term: str) -> str:
    """
    Agent 1:
    Infers the academic domain of the given term.
    This is needed because the dataset only provides opaque topic IDs,
    not descriptive domain names.
    """
    prompt = f"""You are a domain classification agent for academic taxonomy research.

Task: Identify the academic field or discipline that the following term belongs to.

Term: {term}

Rules:
- Return ONLY the domain name (e.g., "finance", "computer science", "law", "medicine", "sociology")
- Be specific but concise (1-4 words)
- Return only the domain name, nothing else."""

    response = ask_llm(prompt, temperature=0.1)

    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\•\d\.\)\s]+", "", line).strip().strip("\"'` ")
        if line and len(line) <= 60 and not is_refusal(line):
            return line

    return "academic taxonomy"


def synonym_generator_agent(term: str, domain: str = "academic taxonomy", n: int = 8) -> List[str]:
    """
    Agent 2:
    Generates candidate synonyms for a term within a specific domain.
    Retries once with higher temperature if the first attempt yields nothing.
    """
    prompt = f"""You are a synonym generation agent for academic taxonomy research.

Task: Generate {n} English synonyms for the given term.

Term: {term}
Domain: {domain}

Rules:
- Return ONLY the synonyms, one per line.
- No bullet points, numbers, or explanations.
- Do not include the original term unless there is truly no alternative.
- Synonyms must be appropriate for the "{domain}" domain."""

    for attempt in range(2):
        response = ask_llm(prompt, temperature=0.2 + attempt * 0.2)
        if is_refusal(response):
            prompt = f"""List {n} academic synonyms for the scholarly term "{term}" used in the field of {domain}.
One synonym per line, no explanations."""
            continue
        candidates = clean_list_response(response)
        if candidates:
            return candidates[:n]

    return []


def domain_critic_agent(
    term: str,
    candidates: List[str],
    domain: str = "academic taxonomy",
) -> List[str]:
    """
    Agent 3:
    Filters candidates that are not valid synonyms in the inferred domain.
    Uses fuzzy matching to map LLM output back to the original candidate list.
    """
    if not candidates:
        return []

    candidate_text = "\n".join(f"- {c}" for c in candidates)

    prompt = f"""You are a domain critic agent for academic taxonomy research.

Task: From the candidates below, keep only those that are valid synonyms in the given domain.

Original term: {term}
Domain: {domain}

Candidates:
{candidate_text}

Rules:
- Return ONLY items from the list above, one per line.
- No explanations or new terms.
- Keep the closest matches if none are perfect."""

    response = ask_llm(prompt)
    selected = clean_list_response(response)

    candidate_map = {c.lower().strip(): c for c in candidates}
    filtered: List[str] = []
    seen: set = set()

    for item in selected:
        key = item.lower().strip()
        if key in candidate_map and key not in seen:
            filtered.append(candidate_map[key])
            seen.add(key)
        else:
            match = fuzz_process.extractOne(
                key,
                list(candidate_map.keys()),
                scorer=fuzz_scorer.ratio,
                score_cutoff=85,
            )
            if match and match[0] not in seen:
                filtered.append(candidate_map[match[0]])
                seen.add(match[0])

    if not filtered:
        return candidates

    return filtered


def semantic_ranker_agent(term: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """
    Agent 4:
    Ranks candidates by embedding cosine similarity to the original term.
    Deterministic — no LLM call.
    """
    if not candidates:
        return []

    texts = [term] + candidates
    embeddings = _embedding_model.encode(texts, convert_to_tensor=True)

    term_emb = embeddings[0]
    cand_embs = embeddings[1:]
    scores = util.cos_sim(term_emb, cand_embs)[0]

    ranked = sorted(
        [{"candidate": c, "score": float(s.item())} for c, s in zip(candidates, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )

    return ranked


def llm_ranking_agent(
    term: str,
    ranked_candidates: List[Dict[str, Any]],
    domain: str = "academic taxonomy",
) -> List[str]:
    """
    Agent 5:
    Agentic re-ranking stage. Takes the embedding-ranked candidates (retrieval)
    and asks an LLM to re-order them best-to-worst as synonyms for the term,
    using the embedding scores as extra context but exercising domain judgment
    on top of them. This retrieval + generation + LLM-ranking combination is
    the one that produced the best experimental results, so it must stay
    LLM-based rather than being collapsed into a purely deterministic step.

    Falls back to the embedding-based order if the LLM output can't be parsed
    into valid candidates.
    """
    if not ranked_candidates:
        return []

    candidate_text = "\n".join(
        f"- {item['candidate']} (semantic_score={item['score']:.3f})"
        for item in ranked_candidates
    )

    prompt = f"""You are a ranking agent for academic taxonomy research.

Task: Re-rank the candidates below from BEST to WORST synonym for the original term,
considering domain fit in addition to the semantic similarity scores shown.

Original term: {term}
Domain: {domain}

Candidates (with embedding similarity scores):
{candidate_text}

Rules:
- Return ALL candidates from the list above, one per line.
- Order them from best match (line 1) to worst match (last line).
- Do not add new terms or explanations."""

    valid_map = {item["candidate"].lower().strip(): item["candidate"] for item in ranked_candidates}

    for attempt in range(2):
        response = ask_llm(prompt, temperature=0.1 + attempt * 0.1)
        selected = clean_list_response(response)

        ordered: List[str] = []
        seen: set = set()
        for item in selected:
            key = item.lower().strip()
            if key in valid_map and key not in seen:
                ordered.append(valid_map[key])
                seen.add(key)
            else:
                match = fuzz_process.extractOne(
                    key, list(valid_map.keys()), scorer=fuzz_scorer.ratio, score_cutoff=85
                )
                if match and match[0] not in seen:
                    ordered.append(valid_map[match[0]])
                    seen.add(match[0])

        if ordered:
            # Keep any candidates the LLM omitted, appended in embedding order.
            for item in ranked_candidates:
                if item["candidate"].lower().strip() not in seen:
                    ordered.append(item["candidate"])
            return ordered

    return [item["candidate"] for item in ranked_candidates]


def select_final_candidates(
    llm_ranked_candidates: List[str],
    ranked_candidates: List[Dict[str, Any]],
    filtered_candidates: List[str],
    candidates: List[str],
    k: int = 5,
) -> List[str]:
    """
    Final step:
    Deterministically takes up to k candidates, preferring the LLM's ranked
    order and padding from earlier pipeline stages (embedding ranking, then
    filtered candidates, then raw generated candidates) whenever an earlier
    stage left fewer than k candidates. No LLM call here, and no access to
    the ground truth at any point in this function or the ones before it.
    """
    final: List[str] = []
    seen: set = set()

    def _extend(pool: List[str]) -> None:
        for item in pool:
            if len(final) >= k:
                return
            key = item.lower().strip()
            if key and key not in seen:
                final.append(item)
                seen.add(key)

    _extend(llm_ranked_candidates)
    _extend([c["candidate"] for c in ranked_candidates])
    _extend(filtered_candidates)
    _extend(candidates)

    return final[:k]


if __name__ == "__main__":
    term = "stocks"

    domain = domain_inference_agent(term)
    print("Domain:", domain)

    candidates = synonym_generator_agent(term, domain)
    print("Generated:", candidates)

    filtered = domain_critic_agent(term, candidates, domain)
    print("Filtered:", filtered)

    ranked = semantic_ranker_agent(term, filtered)
    print("Ranked:", ranked)

    llm_ranked = llm_ranking_agent(term, ranked, domain)
    print("LLM ranked:", llm_ranked)

    final = select_final_candidates(llm_ranked, ranked, filtered, candidates, k=5)
    print("Final top-5:", final)
