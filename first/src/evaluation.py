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


def evaluate_prediction(
    prediction: str,
    ground_truth: str,
    candidates: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Returns all evaluation metrics for one prediction.
    """

    if candidates is None:
        candidates = [prediction] if prediction else []

    return {
        "exact_match": exact_match(prediction, ground_truth),
        "fuzzy_similarity": fuzzy_similarity(prediction, ground_truth),
        "semantic_similarity": semantic_similarity(prediction, ground_truth),
        "top_3_accuracy": top_k_contains_ground_truth(candidates, ground_truth, k=3),
        "top_5_accuracy": top_k_contains_ground_truth(candidates, ground_truth, k=5),
    }


if __name__ == "__main__":
    pred = "shares"
    gold = "shares"

    print(evaluate_prediction(pred, gold, candidates=["equities", "shares", "stocks"]))