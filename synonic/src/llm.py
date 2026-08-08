"""
Centralized LLM access: one place that knows how to talk to Ollama and how
to parse its output into a clean candidate list. baselines.py and agents.py
both depend on this module instead of importing langchain_ollama directly.
"""

import json
import os
import re
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


@lru_cache(maxsize=None)
def get_llm(model: str | None = None, temperature: float = 0.0):
    """Cached ChatOllama instance per (model, temperature) pair, so repeated
    calls in a single run don't reconnect for every term."""
    return ChatOllama(model=model or MODEL_NAME, temperature=temperature)


def ask_llm(system_prompt: str, user_prompt: str, model: str | None = None, temperature: float = 0.0) -> str:
    """Send one system+user message pair to the LLM, return the raw text response."""
    llm = get_llm(model, temperature)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    return response.content


def parse_candidates(text: str) -> list[str]: # convierte el texto de salida del LLM en una lista de candidatos
    """
    Robustly parse LLM output into a clean list of candidate strings.

    Handles:
    - Valid JSON lists
    - Markdown code blocks
    - Bullet lists
    - Numbered lists
    - Broken JSON-like lists
    - Quoted strings with trailing commas
    """
    if text is None:
        return []

    text = str(text).strip()

    text = text.replace("```json", "").replace("```python", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return clean_candidate_list(parsed)
    except Exception:
        pass

    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_part = text[start:end + 1]
            parsed = json.loads(json_part)
            if isinstance(parsed, list):
                return clean_candidate_list(parsed)
    except Exception:
        pass

    quoted_items = re.findall(r'"([^"]+)"', text)
    if quoted_items:
        return clean_candidate_list(quoted_items)

    raw_items = []
    for line in text.splitlines():
        line = line.strip()

        if line in {"[", "]", "{", "}"}:
            continue

        line = re.sub(r"^\s*[\-\*\•]\s*", "", line)
        line = re.sub(r"^\s*\d+[\).\-\:]\s*", "", line)

        if line:
            raw_items.append(line)

    if len(raw_items) <= 1 and "," in text:
        raw_items = text.split(",")

    return clean_candidate_list(raw_items)


def clean_candidate_list(items) -> list[str]: # limpia la lista de candidatos eliminando duplicados, vacíos y normalizando
    from .evaluation import normalize

    cleaned = []
    seen = set()

    for item in items:
        candidate = str(item).strip()

        candidate = candidate.strip()
        candidate = candidate.strip(",")
        candidate = candidate.strip()
        candidate = candidate.strip('"')
        candidate = candidate.strip("'")
        candidate = candidate.strip()
        candidate = candidate.strip(",")

        if candidate in {"[", "]", "{", "}"}:
            continue

        if not candidate:
            continue

        if len(candidate.split()) > 8:
            continue

        norm = normalize(candidate)
        if not norm:
            continue

        if norm not in seen:
            cleaned.append(candidate)
            seen.add(norm)

    return cleaned


if __name__ == "__main__":
    answer = ask_llm(
        "You are a domain terminology expert.",
        'Give one English synonym for the term "stocks". Return only a JSON list of 1 item.',
    )
    print(answer)
    print(parse_candidates(answer))
