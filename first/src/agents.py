from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer, util

from src.llm import ask_llm, clean_list_response


_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def synonym_generator_agent(term: str, topic: str | None = None, n: int = 8) -> List[str]:
    """
    Agent 1:
    Generates possible synonyms for a given term.
    """

    prompt = f"""
You are a synonym generation agent.

Task:
Generate {n} possible English synonyms for the given term.

Term: {term}
Topic/domain: {topic if topic else "academic taxonomy"}

Rules:
- Return only the synonyms.
- One synonym per line.
- Do not explain.
- Do not include the original term unless there is no better option.
"""

    response = ask_llm(prompt)
    candidates = clean_list_response(response)

    return candidates[:n]


def domain_critic_agent(
    term: str,
    candidates: List[str],
    topic: str | None = None,
) -> List[str]:
    """
    Agent 2:
    Filters candidates that are not good domain-specific synonyms.
    """

    if not candidates:
        return []

    candidate_text = "\n".join(f"- {candidate}" for candidate in candidates)

    prompt = f"""
You are a domain critic agent.

Task:
Given an original term and a list of candidate synonyms, keep only the candidates
that are valid synonyms or near-synonyms in the given domain.

Original term: {term}
Topic/domain: {topic if topic else "academic taxonomy"}

Candidate synonyms:
{candidate_text}

Rules:
- Return only candidates from the provided list.
- One candidate per line.
- Do not explain.
- If several are valid, keep the best ones.
- If none are perfect, keep the closest candidates.
"""

    response = ask_llm(prompt)
    selected = clean_list_response(response)

    # Keep only candidates that were actually in the original candidate list.
    candidate_map = {candidate.lower().strip(): candidate for candidate in candidates}
    filtered = []

    for item in selected:
        key = item.lower().strip()
        if key in candidate_map:
            filtered.append(candidate_map[key])

    # Fallback: if the critic fails to follow the format, keep original candidates.
    if not filtered:
        return candidates

    return filtered


def semantic_ranker_agent(term: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """
    Agent 3:
    Ranks candidate synonyms by semantic similarity to the original term.
    This agent is deterministic and uses embeddings.
    """

    if not candidates:
        return []

    texts = [term] + candidates

    embeddings = _embedding_model.encode(
        texts,
        convert_to_tensor=True,
    )

    term_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    scores = util.cos_sim(term_embedding, candidate_embeddings)[0]

    ranked = []

    for candidate, score in zip(candidates, scores):
        ranked.append(
            {
                "candidate": candidate,
                "score": float(score.item()),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return ranked


def final_selector_agent(
    term: str,
    ranked_candidates: List[Dict[str, Any]],
    topic: str | None = None,
) -> str:
    """
    Agent 4:
    Selects the final synonym from the ranked list.
    """

    if not ranked_candidates:
        return ""

    candidate_text = "\n".join(
        f"- {item['candidate']} | semantic_score={item['score']:.3f}"
        for item in ranked_candidates
    )

    prompt = f"""
You are a final decision agent.

Task:
Choose the single best synonym for the original term.

Original term: {term}
Topic/domain: {topic if topic else "academic taxonomy"}

Ranked candidates:
{candidate_text}

Rules:
- Choose exactly one candidate from the list.
- Return only the chosen candidate.
- Do not explain.
"""

    response = ask_llm(prompt)
    selected = clean_list_response(response)

    valid_candidates = {
        item["candidate"].lower().strip(): item["candidate"]
        for item in ranked_candidates
    }

    for item in selected:
        key = item.lower().strip()
        if key in valid_candidates:
            return valid_candidates[key]

    # Fallback: if the LLM fails to return a valid candidate, choose the top-ranked one.
    return ranked_candidates[0]["candidate"]


if __name__ == "__main__":
    term = "stocks"
    topic = "finance"

    candidates = synonym_generator_agent(term, topic)
    print("Generated:", candidates)

    filtered = domain_critic_agent(term, candidates, topic)
    print("Filtered:", filtered)

    ranked = semantic_ranker_agent(term, filtered)
    print("Ranked:", ranked)

    final = final_selector_agent(term, ranked, topic)
    print("Final:", final)