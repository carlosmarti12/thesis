from src.llm import ask_llm, clean_list_response, MODEL_FAST, MODEL_STRONG


_SINGLE_PROMPT = """You are a synonym generation system for academic taxonomy research.

Task: Given an English term, return the best English synonym.

Term: {term}

Rules:
- Return exactly one synonym.
- Return only the synonym, nothing else.
- Do not explain or add bullet points."""


def same_term_baseline(term: str, topic: str | None = None) -> str:
    """Returns the original term — serves as the lower bound."""
    return term


def llm_single_prompt_baseline(term: str, topic: str | None = None) -> str:
    """Single LLM call with llama3.2:3b — no multi-agent pipeline."""
    prompt = _SINGLE_PROMPT.format(term=term)
    response = ask_llm(prompt, model=MODEL_FAST, temperature=0.2)
    cleaned = clean_list_response(response)
    if cleaned:
        return cleaned[0]
    return response.strip().splitlines()[0].strip() if response.strip() else term


def llm_single_prompt_large_baseline(term: str, topic: str | None = None) -> str:
    """Single LLM call with deepseek-r1:8b — stronger model, no pipeline."""
    prompt = _SINGLE_PROMPT.format(term=term)
    response = ask_llm(prompt, model=MODEL_STRONG, temperature=0.1)
    cleaned = clean_list_response(response)
    if cleaned:
        return cleaned[0]
    return response.strip().splitlines()[0].strip() if response.strip() else term


if __name__ == "__main__":
    term = "stocks"
    print("Same-term:  ", same_term_baseline(term))
    print("LLM small:  ", llm_single_prompt_baseline(term))
    print("LLM large:  ", llm_single_prompt_large_baseline(term))
