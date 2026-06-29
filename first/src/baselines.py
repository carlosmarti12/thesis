from src.llm import ask_llm, clean_list_response


def same_term_baseline(term: str, topic: str | None = None) -> str:
    """
    Baseline 1:
    Returns the original term as the prediction.
    This is intentionally simple and weak.
    """

    return term


def llm_single_prompt_baseline(term: str, topic: str | None = None) -> str:
    """
    Baseline 2:
    Uses a single LLM call to generate one synonym.
    No multi-agent architecture.
    """

    prompt = f"""
You are a synonym generation system.

Task:
Given an English term, return the best English synonym.

Term: {term}
Topic/domain: {topic if topic else "academic taxonomy"}

Rules:
- Return exactly one synonym.
- Return only the synonym.
- Do not explain.
- Do not include bullet points.
"""

    response = ask_llm(prompt)
    cleaned = clean_list_response(response)

    if cleaned:
        return cleaned[0]

    return response.strip().splitlines()[0].strip()


if __name__ == "__main__":
    term = "stocks"
    topic = "finance"

    print("Same-term baseline:", same_term_baseline(term, topic))
    print("LLM baseline:", llm_single_prompt_baseline(term, topic))