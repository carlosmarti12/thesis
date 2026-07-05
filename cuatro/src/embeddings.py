from typing import List, Dict

import numpy as np
import ollama


EMBED_MODEL = "nomic-embed-text"

_embed_cache: Dict[str, np.ndarray] = {}


def embed(text: str) -> np.ndarray:
    """Return the embedding vector for a single text string (cached)."""
    key = text.lower().strip()
    if key in _embed_cache:
        return _embed_cache[key]
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    vec = np.array(response["embedding"], dtype=np.float32)
    _embed_cache[key] = vec
    return vec


def embed_batch(texts: List[str]) -> np.ndarray:
    """Return a 2-D array of shape (N, D) for a list of texts."""
    return np.stack([embed(t) for t in texts], axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def batch_cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and each row of a matrix."""
    norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(matrix))
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = (matrix @ query) / (norms * query_norm)
    sims = np.nan_to_num(sims, nan=0.0)
    return sims


if __name__ == "__main__":
    a = embed("stocks")
    b = embed("shares")
    print(f"stocks vs shares: {cosine_sim(a, b):.4f}")
