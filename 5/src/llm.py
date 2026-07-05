import os
import re
from typing import List

import ollama

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

_REFUSAL_PHRASES = [
    "i can't", "i cannot", "i'm unable", "i'm sorry", "i apologize",
    "as an ai", "as a language model", "i won't", "please note",
]
_SKIP_STARTERS = ("here", "sure", "note", "below", "following", "certainly", "of course")


def ask_llm(prompt: str, temperature: float = 0.3, retries: int = 2) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": min(temperature + attempt * 0.15, 1.0)},
            )
            return r["message"]["content"].strip()
        except Exception as e:
            last_exc = e
    raise last_exc  # type: ignore[misc]


def is_refusal(text: str) -> bool:
    return any(p in text.lower() for p in _REFUSAL_PHRASES)


def parse_list(response: str) -> List[str]:
    if is_refusal(response):
        return []
    items: List[str] = []
    for line in response.splitlines():
        line = re.sub(r"^[\-\*\•\d\.\)\s]+", "", line.strip()).strip().strip("\"'` ")
        if not line or any(line.lower().startswith(s) for s in _SKIP_STARTERS) or is_refusal(line) or "?" in line:
            continue
        if len(line) > 80:
            items.extend(p.strip().strip("\"'` ") for p in line.split(",") if 0 < len(p.strip()) <= 60)
            continue
        items.append(line)
    seen: set = set()
    return [x for x in items if not (x.lower() in seen or seen.add(x.lower()))]  # type: ignore[func-returns-value]
