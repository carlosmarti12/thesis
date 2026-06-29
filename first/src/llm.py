import os
import re
from typing import List

import ollama


MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def ask_llm(prompt: str, model: str | None = None, temperature: float = 0.2) -> str:
    """
    Sends a prompt to an Ollama model and returns the text response.
    """

    response = ollama.chat(
        model=model or MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": temperature,
        },
    )

    return response["message"]["content"].strip()


def clean_list_response(response: str) -> List[str]:
    """
    Cleans LLM responses that should contain one item per line.
    """

    items = []

    for line in response.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove bullets, numbers, markdown symbols, quotes
        line = re.sub(r"^[\-\*\•\d\.\)\s]+", "", line).strip()
        line = line.strip("\"'` ")

        # Remove very long lines, explanations, or empty junk
        if not line:
            continue

        if len(line) > 80:
            continue

        lower = line.lower()
        if lower.startswith("here") or lower.startswith("sure") or lower.startswith("note"):
            continue

        items.append(line)

    # Deduplicate while preserving order
    seen = set()
    unique_items = []

    for item in items:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    return unique_items


if __name__ == "__main__":
    answer = ask_llm(
        'Give me one English synonym for the term "stocks". Return only the synonym.'
    )
    print(answer)