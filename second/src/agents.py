import re
from typing import List, Dict, Any

from rapidfuzz import fuzz as fuzz_scorer
from rapidfuzz import process as fuzz_process
from sentence_transformers import SentenceTransformer, util

from src.llm import ask_llm, clean_list_response, is_refusal


_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def domain_inference_agent(term: str) -> str:
    """
    Agent 1 — Domain Inference:
    Infers the academic domain of the input term. The dataset provides only
    opaque topic IDs, so this agent supplies the domain context that all
    downstream agents need to generate and filter relevant synonyms.
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
    Agent 2 — Synonym Generator (dual-strategy):
    Runs two focused generation passes to maximise candidate diversity:
      - Strategy A (lexical): single-word synonyms, near-synonyms, spelling
        variants, acronym expansions.
      - Strategy B (compound): domain-prefixed compound forms and multi-word
        academic labels.
    Results from both passes are merged and deduplicated before being passed
    to the self-reflection agent. Generating more diverse candidates at this
    stage increases the probability that the ground-truth synonym is present
    somewhere in the list.
    """
    prompt_lexical = f"""You are a synonym generation agent for academic taxonomy research.

Task: Generate {n} simple English synonyms or near-synonyms for the term below.
Focus on: single-word alternatives, near-synonyms, spelling variants (British/American),
and full expansions if the term is an acronym or abbreviation.

Term: {term}
Domain: {domain}

Examples:
- "stocks" → shares, equities, securities, holdings
- "anxiety" → worry, apprehension, distress, nervousness
- "adhd" → attention deficit disorder, attention deficit hyperactivity disorder
- "aggression" → hostility, combativeness, belligerence

Rules:
- Return ONLY the synonyms, one per line. No bullets, numbers, or explanations.
- Do not repeat the original term."""

    prompt_compound = f"""You are a synonym generation agent for academic taxonomy research.

Task: Generate {n} compound or domain-specific alternative labels for the term below.
Focus on: multi-word phrases, domain-prefixed forms (e.g. "financial X", "X management"),
and formal academic or institutional labels used in taxonomies and databases.

Term: {term}
Domain: {domain}

Examples:
- "auditors" → financial examiners, account inspectors, fiscal controllers
- "accounting" → financial accounting, financial record-keeping, bookkeeping practice
- "labour economics" → employment economics, work economics, labor market studies
- "recommender system" → recommendation engine, personalized recommendation system

Rules:
- Return ONLY the synonyms, one per line. No bullets, numbers, or explanations.
- Do not repeat the original term."""

    seen: set = set()
    all_candidates: List[str] = []
    original_key = term.lower().strip()

    for prompt in [prompt_lexical, prompt_compound]:
        for attempt in range(2):
            response = ask_llm(prompt, temperature=0.2 + attempt * 0.15)
            if is_refusal(response):
                continue
            batch = clean_list_response(response)
            for c in batch:
                key = c.lower().strip()
                if key and key != original_key and key not in seen:
                    all_candidates.append(c)
                    seen.add(key)
            if batch:
                break

    return all_candidates


def domain_critic_agent(
    term: str,
    candidates: List[str],
    domain: str = "academic taxonomy",
) -> List[str]:
    """
    Agent 3 — Domain Critic:
    Filters candidates that are not valid synonyms in the inferred domain.
    Complements self-reflection: while self-reflection checks synonym quality,
    the critic checks domain relevance (removes terms that are synonyms in
    general language but wrong for this specific academic domain).
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
    Agent 5 — Semantic Ranker:
    Ranks candidates by embedding cosine similarity to the original term.
    Deterministic — no LLM call. Places the most semantically close synonyms
    at the top of the list, which maximises top-k evaluation accuracy and
    ensures the best candidates are presented first.
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


if __name__ == "__main__":
    term = "stocks"

    domain = domain_inference_agent(term)
    print("Domain:", domain)

    candidates = synonym_generator_agent(term, domain)
    print(f"Generated ({len(candidates)}):", candidates)

    filtered = domain_critic_agent(term, candidates, domain)
    print(f"Filtered ({len(filtered)}):", filtered)

    ranked = semantic_ranker_agent(term, filtered)
    print("Ranked:", ranked)
