"""
Utilidades de texto compartidas por agentes y herramientas: normalización
para comparar strings de forma robusta, y parseo de la salida de un LLM
(que puede venir como JSON válido, JSON roto, lista con viñetas, etc.) a
una lista limpia de candidatos.
"""

import json
import re

_NON_ALNUM = re.compile(r"[^a-z0-9\s\-]")
_MULTI_SPACE = re.compile(r"\s+")
_BULLET_PREFIX = re.compile(r"^\s*[\-\*\•]\s*")
_NUMBER_PREFIX = re.compile(r"^\s*\d+[\).\-\:]\s*")


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = _NON_ALNUM.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def clean_candidate_list(items) -> list[str]:
    cleaned = []
    seen = set()

    for item in items:
        candidate = str(item).strip().strip(",").strip().strip('"').strip("'").strip().strip(",")

        if candidate in {"[", "]", "{", "}"} or not candidate:
            continue

        # Descarta explicaciones largas que el LLM cuela como si fueran un candidato.
        if len(candidate.split()) > 8:
            continue

        norm = normalize(candidate)
        if not norm or norm in seen:
            continue

        cleaned.append(candidate)
        seen.add(norm)

    return cleaned


def is_abbreviation_like(term: str) -> bool:
    """Heurística barata (mismo criterio que scripts_error_breakdown.py):
    un solo token, todo mayúsculas o <=5 caracteres - usada para decidir
    cuándo vale la pena pedirle a un LLM que expanda el término antes de
    generar queries de retrieval (ver src/agents/abbreviation_agent.py)."""
    words = str(term).strip().split()
    if len(words) != 1:
        return False
    token = words[0]
    return token.isupper() or len(token) <= 5


_INITIALISM_STOPWORDS = {"of", "and", "the", "in", "for", "a", "an", "on", "to"}


def generate_initialism(term: str) -> str | None:
    """
    Genera el acrónimo determinista de un término multi-palabra (primera
    letra de cada palabra de contenido, saltando conectores comunes) - p.ej.
    "information technology" -> "IT", "quality of worklife" -> "QW".

    Motivado por el failure-pattern analysis del 2026-07-22: los intentos
    previos de expansión de abreviaturas solo atacaban el caso "el INPUT es
    la abreviatura" (EEG, evs). El análisis del dataset completo mostró el
    caso inverso, sin atacar: el GOLD es la abreviatura y el input es la
    forma expandida (`information technology` -> gold `IT`, `identity
    management` -> gold `IdM`) - esos casos son retrieval-miss porque el
    embedder nunca acerca la forma larga a una entrada de 2-4 caracteres en
    el pool. Devuelve None si el término es de una sola palabra (no hay
    nada que abreviar) o si el acrónimo resultante tiene menos de 2 letras
    (¬útil como query de búsqueda).
    """
    words = [w for w in str(term).strip().split() if w.lower() not in _INITIALISM_STOPWORDS]
    if len(words) < 2:
        return None
    letters = "".join(w[0].upper() for w in words if w)
    if len(letters) < 2:
        return None
    return letters


def morphological_variants(term: str) -> list[str]:
    """
    Variantes morfológicas deterministas y baratas de un término (sin LLM):
    singular/plural naive, guión<->espacio, quitar contenido entre
    paréntesis, colapsar "of"/"and" en forma compacta. Pensado como queries
    EXTRA para retrieve_top_k junto al término original - no reemplaza al
    término, lo complementa para casos donde el embedder puntúa peor la
    forma exacta del dataset que una variante equivalente (p.ej. plural vs
    singular, o "e-commerce" vs "ecommerce").
    """
    text = str(term).strip()
    variants: list[str] = []
    seen = {normalize(text)}

    def add(candidate: str) -> None:
        norm_c = normalize(candidate)
        if norm_c and norm_c not in seen:
            variants.append(candidate)
            seen.add(norm_c)

    # Quitar contenido entre paréntesis, p.ej. "term (abbr)" -> "term".
    no_parens = re.sub(r"\s*\([^)]*\)", "", text).strip()
    if no_parens != text:
        add(no_parens)

    # Guión <-> espacio (e-commerce <-> e commerce <-> ecommerce).
    if "-" in text:
        add(text.replace("-", " "))
        add(text.replace("-", ""))
    if " " in text:
        add(text.replace(" ", "-"))

    # Singular/plural naive sobre la ÚLTIMA palabra (donde suele marcarse
    # el número en inglés) - heurística simple, no un lematizador completo.
    words = text.split()
    if words:
        last = words[-1]
        if last.endswith("ies") and len(last) > 3:
            words2 = words[:-1] + [last[:-3] + "y"]
            add(" ".join(words2))
        elif last.endswith("es") and len(last) > 2:
            words2 = words[:-1] + [last[:-2]]
            add(" ".join(words2))
        elif last.endswith("s") and not last.endswith("ss") and len(last) > 1:
            words2 = words[:-1] + [last[:-1]]
            add(" ".join(words2))
        else:
            words2 = words[:-1] + [last + "s"]
            add(" ".join(words2))

    return variants


def parse_candidates(text: str) -> list[str]:
    """Parsea la salida cruda de un LLM a una lista de candidatos, con
    varios niveles de fallback (JSON directo, JSON embebido, strings entre
    comillas, líneas con viñetas/numeración, split por comas)."""
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
            parsed = json.loads(text[start:end + 1])
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
        line = _BULLET_PREFIX.sub("", line)
        line = _NUMBER_PREFIX.sub("", line)
        if line:
            raw_items.append(line)

    if len(raw_items) <= 1 and "," in text:
        raw_items = text.split(",")

    return clean_candidate_list(raw_items)
