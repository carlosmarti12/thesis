import re
import warnings
from typing import List, Tuple

from rapidfuzz import fuzz as fuzz_scorer
from rapidfuzz import process as fuzz_process

from src.llm import ask_llm, clean_list_response, is_refusal, MODEL_FAST, MODEL_STRONG, MODEL_CONTEXTUAL

FUZZY_THRESHOLD = 85
_MIN_CRITIC_KEEP = 3


def domain_classifier_agent(term: str) -> Tuple[str, str]:
    """
    Agent 1 — Domain & Type Classifier.
    Infers the academic domain and classifies the term type.
    Returns (domain, term_type) where term_type is one of:
    'simple', 'compound', 'acronym', 'ambiguous'.
    """
    prompt = f"""You are a domain classification agent for academic taxonomy research.

Task: For the term below, provide two things:
1. The academic field or discipline it belongs to (1-4 words)
2. Its type: 'simple' (single common word), 'compound' (multi-word phrase), 'acronym' (abbreviation/initialism), or 'ambiguous' (unclear or multi-domain)

Term: {term}

Return your answer in exactly this format (two lines, nothing else):
domain: <domain name>
type: <simple|compound|acronym|ambiguous>"""

    response = ask_llm(prompt, temperature=0.1)

    domain = "academic taxonomy"
    term_type = "simple"

    for line in response.splitlines():
        line = line.strip().lower()
        if line.startswith("domain:"):
            raw = line[len("domain:"):].strip().strip("\"'` ")
            if raw and len(raw) <= 60 and not is_refusal(raw):
                domain = raw
        elif line.startswith("type:"):
            raw = line[len("type:"):].strip().strip("\"'` ")
            if raw in ("simple", "compound", "acronym", "ambiguous"):
                term_type = raw

    return domain, term_type


def lexical_generator_agent(term: str, domain: str, term_type: str, n: int = 8) -> List[str]:
    """
    Agent 2a — Lexical Generator (parallel branch).
    Focuses on surface-level variants: near-synonyms, spelling differences,
    acronym expansions. Explicitly avoids multi-word compound phrases.
    """
    prompt = f"""You are a lexical synonym agent for academic taxonomy research.

Task: Generate {n} English synonyms for the term below, focusing exclusively on LEXICAL alternatives.

Term: {term}
Domain: {domain}
Term type: {term_type}

Focus only on:
- Single-word near-synonyms and morphological variants
- British/American spelling differences (e.g., "labour" / "labor")
- Full expansions if the term is an acronym (e.g., "adhd" → "attention deficit hyperactivity disorder")
- Common abbreviations if the term is a long phrase
- Plural or singular reformulations used as labels in taxonomies

Do NOT generate multi-word compound phrases — that is handled by a separate agent.

Rules:
- Return ONLY the synonyms, one per line
- No bullet points, numbers, or explanations
- Do not include the original term
- Each synonym should be a plausible entry in an academic taxonomy or database"""

    return _run_generator(prompt, term, temperature=0.15)


def semantic_generator_agent(term: str, domain: str, n: int = 8) -> List[str]:
    """
    Agent 2b — Semantic Generator (parallel branch).
    Focuses on conceptual equivalents: technical jargon, cross-disciplinary
    names, hypernyms/hyponyms that practitioners treat as interchangeable.
    """
    prompt = f"""You are a semantic synonym agent for academic taxonomy research.

Task: Generate {n} English synonyms for the term below, focusing exclusively on SEMANTIC equivalents.

Term: {term}
Domain: {domain}

Focus only on:
- Conceptually equivalent terms used in the {domain} domain
- Technical jargon and formal academic labels for the same concept
- Hypernyms or hyponyms that practitioners treat as interchangeable
- Cross-disciplinary names for the same concept when used in {domain}
- Formal institutional labels (e.g., from standards bodies, government databases, textbooks)

Do NOT generate simple spelling variants — that is handled by a separate agent.

Rules:
- Return ONLY the synonyms, one per line
- No bullet points, numbers, or explanations
- Do not include the original term
- Prefer terms as they appear in academic literature, not in colloquial speech"""

    return _run_generator(prompt, term, temperature=0.30)


def contextual_generator_agent(term: str, domain: str, n: int = 8) -> List[str]:
    """
    Agent 2c — Contextual Generator (parallel branch).
    Reasons through a domain context sentence to find multi-word labels,
    formal/institutional terms, and compound phrases.
    """
    prompt = f"""You are a contextual synonym agent for academic taxonomy research.

Task: Generate {n} English synonyms for the term below by analyzing it in a real domain context.

Term: {term}
Domain: {domain}

Step 1 (internal reasoning only, do not output): Mentally construct a sentence using "{term}" in a {domain} research paper or database entry.
Step 2: Identify which other terms or phrases could replace "{term}" in that sentence while preserving the meaning.

Focus on:
- Domain-specific compound phrases and multi-word labels (e.g., "financial securities", "investment holdings")
- Formal or bureaucratic equivalents used by institutions in the {domain} field
- Historical or legacy terminology still used in academic databases and taxonomies
- Both short and multi-word forms are welcome

Rules:
- Return ONLY the synonyms from Step 2, one per line — do not output the sentence from Step 1
- No bullet points, numbers, or explanations
- Do not include the original term"""

    model = MODEL_CONTEXTUAL
    return _run_generator(prompt, term, temperature=0.50, model=model)


def critic_agent(
    term: str,
    candidates: List[str],
    domain: str,
    use_strong_model: bool = False,
) -> List[str]:
    """
    Agent 3 — Domain Critic.
    Filters the candidate pool, keeping only valid synonyms for the term in
    the given domain. Biased toward recall: always keeps at least _MIN_CRITIC_KEEP
    candidates. Uses fuzzy matching to map LLM output back to canonical candidates.
    """
    if not candidates:
        return []

    model = MODEL_STRONG if use_strong_model else MODEL_FAST
    candidate_text = "\n".join(f"- {c}" for c in candidates)

    prompt = f"""You are a domain expert critic for academic synonym validation.

Task: Review the candidates below. Keep ONLY those that are valid synonyms for the original term when used in the given academic domain.

A candidate is VALID if:
1. It has the same or very similar meaning as the original term, AND
2. It would be a plausible label for the same concept in an academic database or taxonomy in the {domain} field

Original term: {term}
Domain: {domain}

Candidates:
{candidate_text}

Rules:
- Return ONLY the candidates you are keeping, one per line
- Do not add new terms that were not in the list
- Do not explain your decisions
- When in doubt, KEEP the candidate (prefer recall over precision)
- You MUST keep at least {_MIN_CRITIC_KEEP} candidates; if fewer are valid, keep the {_MIN_CRITIC_KEEP} most plausible ones"""

    response = ask_llm(prompt, model=model, temperature=0.1)
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
                scorer=fuzz_scorer.token_set_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if match and match[0] not in seen:
                filtered.append(candidate_map[match[0]])
                seen.add(match[0])

    if not filtered:
        warnings.warn(f"[critic] Fallback: returning all {len(candidates)} candidates for '{term}'")
        return candidates

    return filtered


def reflection_agent(
    term: str,
    domain: str,
    top_candidates: List[str],
    existing_candidates: List[str],
    n: int = 5,
) -> List[str]:
    """
    Agent 4 (optional) — Reflection / Gap Filler.
    Triggered when the top consensus score is below a threshold.
    Looks at the current best candidates and tries to generate alternatives
    that might be a better fit, avoiding what's already in the pool.
    """
    existing_text = "\n".join(f"- {c}" for c in existing_candidates[:10])
    top_text = ", ".join(top_candidates[:3]) if top_candidates else "(none yet)"

    prompt = f"""You are a synonym reflection agent for academic taxonomy research.

The current best synonym candidates for "{term}" (domain: {domain}) are: {top_text}

These candidates are already in the pool and do not need to be repeated:
{existing_text}

Task: Generate {n} NEW English synonyms for "{term}" that are NOT in the list above.
Focus on alternatives that might be a better or more precise match in the {domain} domain.

Rules:
- Return ONLY new synonyms, one per line
- No bullet points, numbers, or explanations
- Do not repeat any term already listed above
- Do not include the original term "{term}" """

    return _run_generator(prompt, term, temperature=0.4)


def _run_generator(
    prompt: str,
    original_term: str,
    temperature: float = 0.2,
    model: str | None = None,
) -> List[str]:
    """Shared retry logic for all generator agents."""
    original_key = original_term.lower().strip()

    for attempt in range(2):
        response = ask_llm(prompt, model=model, temperature=temperature + attempt * 0.15)
        if is_refusal(response):
            continue
        batch = clean_list_response(response)
        result = [c for c in batch if c.lower().strip() != original_key]
        if result:
            return result

    return []


if __name__ == "__main__":
    term = "stocks"
    domain, term_type = domain_classifier_agent(term)
    print(f"Domain: {domain}, Type: {term_type}")

    lex = lexical_generator_agent(term, domain, term_type)
    sem = semantic_generator_agent(term, domain)
    ctx = contextual_generator_agent(term, domain)
    print("Lexical:", lex)
    print("Semantic:", sem)
    print("Contextual:", ctx)

    all_candidates = list(dict.fromkeys(lex + sem + ctx))
    filtered = critic_agent(term, all_candidates, domain)
    print("After critic:", filtered)
