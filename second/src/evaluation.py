import re
from typing import List, Dict, Any

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util


_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Minimum token_set_ratio score (0–100) to count a fuzzy match as correct.
FUZZY_THRESHOLD = 80


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def fuzzy_similarity(prediction: str, ground_truth: str) -> float:
    pred = normalize_text(prediction)
    gold = normalize_text(ground_truth)
    if not pred or not gold:
        return 0.0
    return fuzz.ratio(pred, gold) / 100.0


def semantic_similarity(prediction: str, ground_truth: str) -> float:
    pred = normalize_text(prediction)
    gold = normalize_text(ground_truth)
    if not pred or not gold:
        return 0.0
    embeddings = _embedding_model.encode([pred, gold], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return float(score.item())


def top_k_contains_ground_truth(
    candidates: List[str],
    ground_truth: str,
    k: int = 8,
    fuzzy: bool = False,
) -> bool:
    """
    Returns True if ground_truth appears in the top-k candidates.
    With fuzzy=True, also accepts candidates whose token_set_ratio with
    the ground truth is >= FUZZY_THRESHOLD (catches partial/reordered matches
    like "Examiners" vs "financial examiners").
    """
    gold = normalize_text(ground_truth)
    top_k = candidates[:k]

    for candidate in top_k:
        pred = normalize_text(candidate)
        if pred == gold:
            return True
        if fuzzy and fuzz.token_set_ratio(pred, gold) >= FUZZY_THRESHOLD:
            return True

    return False


def evaluate_prediction(
    prediction: str,
    ground_truth: str,
    candidates: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Returns all evaluation metrics for one prediction.
    Candidates should be the full ranked list (up to 8 items for MAS).
    For single-prediction baselines pass candidates=[prediction].
    """
    if candidates is None:
        candidates = [prediction] if prediction else []

    return {
        "exact_match": exact_match(prediction, ground_truth),
        "fuzzy_similarity": fuzzy_similarity(prediction, ground_truth),
        "semantic_similarity": semantic_similarity(prediction, ground_truth),
        "top_3_accuracy": top_k_contains_ground_truth(candidates, ground_truth, k=3, fuzzy=False),
        "top_5_accuracy": top_k_contains_ground_truth(candidates, ground_truth, k=5, fuzzy=False),
        "top_8_accuracy": top_k_contains_ground_truth(candidates, ground_truth, k=8, fuzzy=False),
        "top_8_fuzzy_accuracy": top_k_contains_ground_truth(candidates, ground_truth, k=8, fuzzy=True),
    }


if __name__ == "__main__":
    pred = "examiners"
    gold = "financial examiners"
    print("Exact:", exact_match(pred, gold))
    print("Fuzzy top-8:", top_k_contains_ground_truth([pred], gold, k=8, fuzzy=True))
    print(evaluate_prediction(pred, gold, candidates=["equities", "examiners", "auditors"]))
