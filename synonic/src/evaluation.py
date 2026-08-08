"""
Evaluation metrics for the open synonym-discovery task.

IMPORTANT: this module is the *only* place in `synonic/` that is allowed to
read `en_synonym`, together with the `ground_truth` column each run script
attaches to its output rows. Every method in methods.py / mas_pipeline.py
receives only the input term and produces candidates without ever seeing
`en_synonym` - see README.md for the full leakage writeup this fixes.

For each (term, ground_truth) pair, a method returns an ordered list of
`num_candidates` predicted synonym strings (index 0 = most confident). All
metrics below are computed from that single ordered list against the single
ground-truth string.

Because this dataset provides exactly ONE gold synonym per term (not a set
of acceptable synonyms), several standard IR metrics degenerate or need a
caveat - noted per metric below.

Naming: exact_top1/recall_at_3/recall_at_5 were renamed to hit@1/hit@3/hit@5
(standard IR naming, per advisor feedback, matching 18_7/rsc/evaluation.py) -
same computation as before, just renamed. ndcg@3/ndcg@5 are new.
"""

import math
import re

import numpy as np
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

SEMANTIC_SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace, drop punctuation other than
    hyphens. Used to compare strings for exact/rank-based metrics so that
    e.g. 'Stock Market' == 'stock market'."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_metrics(ground_truth: str, candidates: list[str], embedder) -> dict:
    """
    Compute every metric for one (ground_truth, candidates) pair.

    `candidates` is the ranked list of predictions actually returned by the
    method for this term - its length is `num_candidates` (recorded
    separately by the caller as `num_final_candidates`, since exact_any/
    recall@K only mean something once you know how many candidates were
    allowed to compete for a hit).

    Metrics returned:

    - hit@1: candidates[0] normalizes to exactly the same string as
      ground_truth. The strictest metric - only the #1 prediction counts.
      (Was `exact_top1`.)

    - exact_any: ground_truth appears anywhere in `candidates` (i.e. within
      the `num_candidates` returned - always report num_final_candidates
      alongside this number, since exact_any trivially increases the more
      candidates a method is allowed to return).

    - hit@3 / hit@5: ground_truth appears within the first 3 / 5 candidates.
      (Were `recall_at_3` / `recall_at_5` - same computation, renamed to
      standard IR naming to match 18_7/rsc/evaluation.py.)

    - ndcg@3 / ndcg@5: NDCG with binary relevance and a single relevant item.
      DCG@k = 1/log2(rank+1) if the gold answer falls within the first k
      positions, else 0. IDCG@k (ideal order: gold in position 1) = 1, so
      NDCG@k reduces to 1/log2(rank+1) when rank <= k, 0 otherwise - unlike
      hit@k (binary), this penalizes gradually the further the hit is from
      position 1.

    - precision_at_5: (# of the top-5 candidates that exactly match
      ground_truth) / 5. CAVEAT (per advisor feedback): because there is
      only one gold synonym, at most 1 of the 5 slots can ever be correct,
      so precision@5 is mechanically capped at 0.2 and mostly just measures
      whether recall@5 succeeded, divided by 5. It does NOT measure whether
      the *other 4* candidates are "wrong" (some may be valid synonyms the
      dataset simply didn't record). Prefer recall@5 / exact_any for this
      dataset; precision@5 is reported only for completeness.

    - mrr (Mean Reciprocal Rank, computed per-row here, meaned by
      summarize_results): 1 / rank_of_correct_answer, where rank is the
      1-indexed position of ground_truth in `candidates` (0.0 if absent).
      E.g. correct answer in position 1 -> 1.0, position 2 -> 0.5, position
      4 -> 0.25, not found -> 0. Rewards ranking the correct synonym near
      the top even when it isn't first.

    - fuzzy_similarity: max over candidates of
      rapidfuzz.fuzz.ratio(normalize(ground_truth), normalize(candidate)) / 100.
      fuzz.ratio is normalized indel (edit) similarity: 100 * (1 -
      edit_distance / (len(a) + len(b))), i.e. it scores lexical/spelling
      closeness, not meaning - useful for catching near-miss spellings or
      morphological variants ("optimise" vs "optimize") that exact-match
      metrics treat as complete misses.

    - semantic_similarity: max over candidates of cosine similarity between
      the ground_truth embedding and the candidate embedding, both encoded
      with `sentence-transformers/all-MiniLM-L6-v2` (SEMANTIC_SIMILARITY_MODEL)
      with L2-normalized embeddings - so cosine similarity is just the dot
      product. This is a *meaning*-based score: near-1.0 for paraphrases
      that share no words with ground_truth, unlike fuzzy_similarity.
    """
    if not candidates:
        candidates = [""]

    norm_gt = normalize(ground_truth)
    norm_candidates = [normalize(c) for c in candidates]

    rank = None
    for idx, cand in enumerate(norm_candidates, start=1):
        if cand == norm_gt:
            rank = idx
            break

    hit_at_1 = rank == 1
    exact_any = norm_gt in norm_candidates
    hit_at_3 = rank is not None and rank <= 3
    hit_at_5 = rank is not None and rank <= 5

    ndcg_at_3 = (1.0 / math.log2(rank + 1)) if hit_at_3 else 0.0
    ndcg_at_5 = (1.0 / math.log2(rank + 1)) if hit_at_5 else 0.0

    top5_hits = sum(1 for c in norm_candidates[:5] if c == norm_gt)
    precision_at_5 = top5_hits / 5.0

    mrr = (1.0 / rank) if rank is not None else 0.0

    fuzzy_scores = [fuzz.ratio(norm_gt, normalize(c)) / 100.0 for c in candidates]
    max_fuzzy = max(fuzzy_scores) if fuzzy_scores else 0.0

    texts = [ground_truth] + list(candidates)
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    gt_emb = embeddings[0:1]
    cand_embs = embeddings[1:]
    semantic_scores = cosine_similarity(gt_emb, cand_embs)[0]
    max_semantic = float(np.max(semantic_scores)) if len(semantic_scores) else 0.0

    return {
        "hit@1": hit_at_1,
        "exact_any": exact_any,
        "hit@3": hit_at_3,
        "hit@5": hit_at_5,
        "ndcg@3": ndcg_at_3,
        "ndcg@5": ndcg_at_5,
        "precision_at_5": precision_at_5,
        "mrr": mrr,
        "fuzzy_similarity": max_fuzzy,
        "semantic_similarity": max_semantic,
    }


def summarize_results(results_df) -> dict:
    """Average every per-row metric across the whole run, plus how many
    candidates were returned on average (num_final_candidates) - report this
    number next to exact_any/hit@5 whenever citing them, since both
    depend on how many candidates the method was allowed to return."""
    return {
        "rows": len(results_df),
        "avg_num_candidates_returned": float(results_df["num_final_candidates"].mean()),
        "hit@1": float(results_df["hit@1"].mean()),
        "exact_any": float(results_df["exact_any"].mean()),
        "hit@3": float(results_df["hit@3"].mean()),
        "hit@5": float(results_df["hit@5"].mean()),
        "ndcg@3": float(results_df["ndcg@3"].mean()),
        "ndcg@5": float(results_df["ndcg@5"].mean()),
        "precision_at_5": float(results_df["precision_at_5"].mean()),
        "mean_mrr": float(results_df["mrr"].mean()),
        "mean_fuzzy_similarity": float(results_df["fuzzy_similarity"].mean()),
        "mean_semantic_similarity": float(results_df["semantic_similarity"].mean()),
        "avg_time_seconds": float(results_df["time_seconds"].mean()),
    }
