from typing import List
from src.llm import ask_llm, parse_list, is_refusal


def generator_agent(term: str) -> List[str]:
    """
    Agent 1 — Synonym Generator.
    Given a term, returns up to 10 synonym candidates.
    No domain inference needed — the model determines context itself.
    """
    prompt = f"""You are a synonym expert for academic research databases.

Task: Generate exactly 10 English synonyms or semantically equivalent terms for the word below.
Include a mix of: single-word synonyms, multi-word equivalents, and domain-specific labels.

Term: "{term}"

Rules:
- Return exactly 10 synonyms, one per line.
- No bullet points, numbers, or explanations.
- Do not repeat the original term.
- Keep each entry under 60 characters."""

    for attempt in range(2):
        response = ask_llm(prompt, temperature=0.3 + attempt * 0.15)
        if not is_refusal(response):
            candidates = parse_list(response)
            candidates = [c for c in candidates if c.lower().strip() != term.lower().strip()]
            if candidates:
                return candidates[:10]
    return []


def filter_agent(term: str, candidates: List[str]) -> List[str]:
    """
    Agent 2 — Synonym Filter.
    Reviews the candidates and keeps the best 8 that genuinely make sense
    as synonyms for the given term.
    """
    if not candidates:
        return []

    candidate_text = "\n".join(f"- {c}" for c in candidates)

    prompt = f"""You are a synonym validation expert for academic research databases.

Task: From the list below, select the best 8 terms that are genuine synonyms or semantically equivalent to "{term}".

Term: "{term}"

Candidates:
{candidate_text}

Rules:
- Return ONLY items from the list above, one per line (copy them exactly).
- Select the 8 most accurate and useful synonyms.
- If fewer than 8 are valid, return as many as are genuinely correct.
- No explanations, no new terms, no bullet points."""

    from rapidfuzz import process as fuzz_process, fuzz as fuzz_scorer

    response = ask_llm(prompt, temperature=0.1)
    selected = parse_list(response)

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
                key, list(candidate_map.keys()),
                scorer=fuzz_scorer.token_set_ratio, score_cutoff=85,
            )
            if match and match[0] not in seen:
                filtered.append(candidate_map[match[0]])
                seen.add(match[0])

    # Fallback: if filter is too aggressive, return all candidates
    return filtered if filtered else candidates
