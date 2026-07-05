from src.llm import ask_llm, clean_list_response


def same_term_baseline(term: str, topic: str | None = None) -> str:
    """
    Baseline 1:
    Returns the original term as the prediction.
    Intentionally trivial — serves as the lower bound.
    """
    return term


def llm_single_prompt_baseline(term: str, topic: str | None = None) -> str:
    """
    Baseline 2:
    Single LLM call with no multi-agent pipeline.
    Does not use the topic ID since it is an opaque dataset identifier.
    """
    prompt = f"""You are a synonym generation system for academic taxonomy research.

Task: Given an English term, return the best English synonym.

Term: {term}

Rules:
- Return exactly one synonym.
- Return only the synonym, nothing else.
- Do not explain or add bullet points."""

    response = ask_llm(prompt)
    cleaned = clean_list_response(response)

    if cleaned:
        return cleaned[0]

    return response.strip().splitlines()[0].strip()


if __name__ == "__main__":
    term = "stocks"

    print("Same-term baseline:", same_term_baseline(term))
    print("LLM baseline:", llm_single_prompt_baseline(term))
