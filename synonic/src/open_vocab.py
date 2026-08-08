"""
Open-world candidate vocabulary.

This is the fix for the data-leakage bug found in new/src2: several methods
there built their retrieval index from df['en_synonym'] - i.e. they searched
directly inside the answer column, which is not available in a realistic
setting where only the input term is known.

Here the candidate vocabulary comes from WordNet (via nltk) instead: every
lemma phrase across all synsets, independent of this project's dataset or
its ground-truth column. Methods that need a fixed "vocabulary to search"
(embedding_wordnet, the MAS retrieval_agent) use this module, never
df['en_synonym'].
"""

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def _normalize(text: str) -> str:
    import re

    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def build_wordnet_vocab(max_words: int = 4) -> list[str]:
    """
    Collect every lemma phrase from WordNet (nltk.corpus.wordnet), replacing
    '_' with spaces, deduplicated by normalized text. Phrases longer than
    `max_words` words are dropped (WordNet has some very long technical
    phrases that are rarely useful as short synonym candidates).
    """
    from nltk.corpus import wordnet as wn

    seen_norm = set()
    vocab = []

    for lemma_name in wn.all_lemma_names():
        phrase = lemma_name.replace("_", " ").strip()

        if not phrase or len(phrase.split()) > max_words:
            continue

        norm = _normalize(phrase)
        if not norm or norm in seen_norm:
            continue

        seen_norm.add(norm)
        vocab.append(phrase)

    return sorted(vocab)


def get_wordnet_vocab(max_words: int = 4) -> list[str]:
    """Cached wrapper around build_wordnet_vocab (rebuilding is ~seconds but
    this avoids repeating the sort/dedup pass across every run)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"wordnet_vocab_maxwords{max_words}.json"

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    vocab = build_wordnet_vocab(max_words=max_words)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    return vocab


def get_wordnet_index(embedder, embedder_name: str, max_words: int = 4) -> tuple[list[str], np.ndarray]:
    """
    Return (terms, embeddings) for the full WordNet vocabulary, embedded with
    `embedder` (a SentenceTransformer instance). Embeddings are cached to
    disk keyed by embedder name + vocab size, since encoding ~100k phrases
    is the expensive part and the vocabulary/model don't change between runs.
    """
    vocab = get_wordnet_vocab(max_words=max_words)

    safe_name = hashlib.sha1(embedder_name.encode("utf-8")).hexdigest()[:10]
    emb_cache_path = CACHE_DIR / f"wordnet_embeddings_{safe_name}_{len(vocab)}.npy"
    terms_cache_path = CACHE_DIR / f"wordnet_embeddings_{safe_name}_{len(vocab)}_terms.json"

    if emb_cache_path.exists() and terms_cache_path.exists():
        embeddings = np.load(emb_cache_path)
        with open(terms_cache_path, "r", encoding="utf-8") as f:
            terms = json.load(f)
        return terms, embeddings

    embeddings = embedder.encode(
        vocab, normalize_embeddings=True, show_progress_bar=True, batch_size=256
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_cache_path, embeddings)
    with open(terms_cache_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    return vocab, embeddings
