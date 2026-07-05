import re
from typing import List, Dict, Any

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util


_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def normalize_text(text: str) -> str:
    """
    Lowercase, remove punctuation, and normalize spaces.
    """

    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


def exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Strict normalized exact match.
    """

    return normalize_text(prediction) == normalize_text(ground_truth)


def fuzzy_similarity(prediction: str, ground_truth: str) -> float:
    """
    String-level similarity between 0 and 1.
    Useful when outputs are almost the same but not exact.
    """

    pred = normalize_text(prediction)
    gold = normalize_text(ground_truth)

    if not pred or not gold:
        return 0.0

    return fuzz.ratio(pred, gold) / 100.0


def semantic_similarity(prediction: str, ground_truth: str) -> float:
    """
    Embedding-based semantic similarity between prediction and ground truth.
    """

    pred = normalize_text(prediction)
    gold = normalize_text(ground_truth)

    if not pred or not gold:
        return 0.0

    embeddings = _embedding_model.encode(
        [pred, gold],
        convert_to_tensor=True,
    )

    score = util.cos_sim(embeddings[0], embeddings[1])
    return float(score.item())


def top_k_contains_ground_truth(
    candidates: List[str],
    ground_truth: str,
    k: int = 3,
) -> bool:
    """
    Checks whether the ground truth appears in the top-k candidate list.
    """

    gold = normalize_text(ground_truth)
    top_k = candidates[:k]

    for candidate in top_k:
        if normalize_text(candidate) == gold:
            return True

    return False


def exact_top1(candidates: List[str], ground_truth: str) -> bool:
    """
    Strict exact match between the first (top-ranked) candidate and the ground truth.
    """

    if not candidates:
        return False

    return exact_match(candidates[0], ground_truth)


def exact_any(candidates: List[str], ground_truth: str) -> bool:
    """
    True if the ground truth exactly matches any candidate in the list.
    """

    gold = normalize_text(ground_truth)
    return any(normalize_text(c) == gold for c in candidates)


def mrr(candidates: List[str], ground_truth: str) -> float:
    """
    Mean Reciprocal Rank of the first exact match (1/rank), 0.0 if absent.
    """

    gold = normalize_text(ground_truth)
    for rank, candidate in enumerate(candidates, start=1):
        if normalize_text(candidate) == gold:
            return 1.0 / rank

    return 0.0


def evaluate_candidates(
    candidates: List[str],
    ground_truth: str,
) -> Dict[str, Any]:
    """
    Returns all evaluation metrics for a final candidate list, comparing it
    against `ground_truth` (e.g. row["en_synonym"]).

    `candidates` must be the system's *output* (already generated without any
    access to the ground truth) — this function is only called afterwards,
    purely for scoring.
    """

    top1 = candidates[0] if candidates else ""

    return {
        "exact_top1": exact_top1(candidates, ground_truth),
        "exact_any": exact_any(candidates, ground_truth),
        "top_3_accuracy": top_k_contains_ground_truth(candidates, ground_truth, k=3),
        "top_5_accuracy": top_k_contains_ground_truth(candidates, ground_truth, k=5),
        "fuzzy_similarity": fuzzy_similarity(top1, ground_truth),
        "semantic_similarity": semantic_similarity(top1, ground_truth),
        "mrr": mrr(candidates, ground_truth),
    }


if __name__ == "__main__":
    gold = "shares"
    candidates = ["equities", "shares", "stocks"]

    print(evaluate_candidates(candidates, gold))