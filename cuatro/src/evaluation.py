import re
from typing import List, Dict, Any, Union

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util


# Use all-MiniLM-L6-v2 for evaluation metrics (same as first/ and second/)
# so results are directly comparable across experiments.
# nomic-embed-text is used internally by the pipeline for ranking only.
_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
    ground_truths: Union[List[str], str],
    k: int,
    fuzzy: bool = False,
) -> bool:
    """
    Returns True if any of the ground truths appears in the top-k candidates.
    With fuzzy=True, also accepts near-matches via token_set_ratio.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    top_k = candidates[:k]
    gold_norms = [normalize_text(g) for g in ground_truths]

    for cand in top_k:
        pred_norm = normalize_text(cand)
        for gold_norm in gold_norms:
            if pred_norm == gold_norm:
                return True
            if fuzzy and fuzz.token_set_ratio(pred_norm, gold_norm) >= FUZZY_THRESHOLD:
                return True

    return False


def mrr(
    candidates: List[str],
    ground_truths: Union[List[str], str],
    fuzzy: bool = False,
) -> float:
    """
    Mean Reciprocal Rank: returns 1/rank of the first correct answer in the
    candidate list, or 0.0 if no correct answer is found.
    Higher is better; a correct top-1 prediction gives MRR = 1.0.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    gold_norms = [normalize_text(g) for g in ground_truths]

    for rank, cand in enumerate(candidates, start=1):
        pred_norm = normalize_text(cand)
        for gold_norm in gold_norms:
            if pred_norm == gold_norm:
                return 1.0 / rank
            if fuzzy and fuzz.token_set_ratio(pred_norm, gold_norm) >= FUZZY_THRESHOLD:
                return 1.0 / rank

    return 0.0


def evaluate_prediction(
    prediction: str,
    ground_truths: Union[List[str], str],
    candidates: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for one prediction.

    - Point metrics (exact_match, fuzzy_similarity, semantic_similarity): best
      score across all ground truths.
    - Set metrics (top-k, MRR): True/nonzero if ANY ground truth is matched.
    - candidates should be the full ranked list; for baselines pass [prediction].
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    if candidates is None:
        candidates = [prediction] if prediction else []

    # Point metrics — best match across all ground truths
    best_exact = any(exact_match(prediction, g) for g in ground_truths)
    best_fuzzy = max((fuzzy_similarity(prediction, g) for g in ground_truths), default=0.0)
    best_sem = max((semantic_similarity(prediction, g) for g in ground_truths), default=0.0)

    return {
        "exact_match": best_exact,
        "fuzzy_similarity": best_fuzzy,
        "semantic_similarity": best_sem,
        "top_3_accuracy": top_k_contains_ground_truth(candidates, ground_truths, k=3),
        "top_5_accuracy": top_k_contains_ground_truth(candidates, ground_truths, k=5),
        "top_10_accuracy": top_k_contains_ground_truth(candidates, ground_truths, k=10),
        "top_5_fuzzy_accuracy": top_k_contains_ground_truth(candidates, ground_truths, k=5, fuzzy=True),
        "top_10_fuzzy_accuracy": top_k_contains_ground_truth(candidates, ground_truths, k=10, fuzzy=True),
        "mrr": mrr(candidates, ground_truths),
        "mrr_fuzzy": mrr(candidates, ground_truths, fuzzy=True),
    }


if __name__ == "__main__":
    pred = "examiners"
    golds = ["financial examiners", "auditors"]
    print(evaluate_prediction(pred, golds, candidates=["equities", "examiners", "auditors"]))
