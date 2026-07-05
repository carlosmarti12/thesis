import os
import re
from typing import List

import ollama


MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

_REFUSAL_PHRASES = [
    "i can't", "i cannot", "i'm unable", "i'm sorry", "i apologize",
    "as an ai", "as a language model", "self-harm", "harmful behavior",
    "inappropriate", "i don't think i can", "i won't",
    "is there anything", "can i help", "let me know if",
    "please note", "it's worth noting", "keep in mind",
]

_SKIP_STARTERS = ("here", "sure", "note", "below", "following", "certainly", "of course")


def ask_llm(prompt: str, model: str | None = None, temperature: float = 0.2, retries: int = 2) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = ollama.chat(
                model=model or MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": min(temperature + attempt * 0.15, 1.0)},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            last_exc = e
    raise last_exc  # type: ignore[misc]


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def clean_list_response(response: str) -> List[str]:
    if is_refusal(response):
        return []

    items: List[str] = []

    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue

        # Strip "original → synonym" or "original -> synonym" patterns produced by few-shot examples
        for arrow in (" → ", " -> ", "→", "->"):
            if arrow in line:
                line = line.split(arrow, 1)[1].strip()
                break

        line = re.sub(r"^[\-\*\•\d\.\)\s]+", "", line).strip()
        line = line.strip("\"'` ")

        if not line:
            continue

        lower = line.lower()

        if any(lower.startswith(s) for s in _SKIP_STARTERS):
            continue

        if is_refusal(line):
            continue

        if "?" in line:
            continue

        if len(line) > 80:
            # Handle comma-separated inline lists like "shares, equities, securities"
            parts = [
                re.sub(r"^[\-\*\•\d\.\)\s]+", "", p).strip().strip("\"'` ")
                for p in line.split(",")
            ]
            items.extend(p for p in parts if p and 0 < len(p) <= 60 and not is_refusal(p))
            continue

        items.append(line)

    seen: set = set()
    unique: List[str] = []
    for item in items:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


if __name__ == "__main__":
    answer = ask_llm(
        'Give me one English synonym for the term "stocks". Return only the synonym.'
    )
    print(answer)
