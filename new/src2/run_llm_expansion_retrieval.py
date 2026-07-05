import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


DATA_PATH = Path("data/topic_synonyms_clean_with_duplicates.csv")
RESULTS_DIR = Path("thesis_results")


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_candidates(text: str) -> list[str]:
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

    # Remove markdown fences first
    text = text.replace("```json", "").replace("```python", "").replace("```", "").strip()

    # Try direct JSON parsing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return clean_candidate_list(parsed)
    except Exception:
        pass

    # Try extracting JSON array substring
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

    # Try extracting quoted strings from broken JSON-like output
    quoted_items = re.findall(r'"([^"]+)"', text)
    if quoted_items:
        return clean_candidate_list(quoted_items)

    # Fallback: line-based parsing
    raw_items = []
    for line in text.splitlines():
        line = line.strip()

        # Skip JSON/list structure characters
        if line in {"[", "]", "{", "}"}:
            continue

        # Remove bullets or numbering
        line = re.sub(r"^\s*[\-\*\•]\s*", "", line)
        line = re.sub(r"^\s*\d+[\).\-\:]\s*", "", line)

        if line:
            raw_items.append(line)

    # Final fallback: comma split if everything was one line
    if len(raw_items) <= 1 and "," in text:
        raw_items = text.split(",")

    return clean_candidate_list(raw_items)


def clean_candidate_list(items) -> list[str]:
    cleaned = []
    seen = set()

    for item in items:
        candidate = str(item).strip()

        # Remove common JSON/list artifacts
        candidate = candidate.strip()
        candidate = candidate.strip(",")
        candidate = candidate.strip()
        candidate = candidate.strip('"')
        candidate = candidate.strip("'")
        candidate = candidate.strip()
        candidate = candidate.strip(",")

        # Remove accidental brackets
        if candidate in {"[", "]", "{", "}"}:
            continue

        # Remove empty or meaningless items
        if not candidate:
            continue

        # Avoid long explanations
        if len(candidate.split()) > 8:
            continue

        norm = normalize(candidate)
        if not norm:
            continue

        if norm not in seen:
            cleaned.append(candidate)
            seen.add(norm)

    return cleaned


def build_output_vector_index(df: pd.DataFrame, embedder) -> tuple[list[str], np.ndarray]:
    """
    Build a local in-memory vector index of candidate output terms from
    df['en_synonym'] only (closed-world output vocabulary).

    Deduplicated by normalized text so identical candidates are embedded
    once. This index is built a single time, before the main loop.
    """
    raw_terms = df["en_synonym"].dropna().astype(str).tolist()

    seen_norm = set()
    index_terms = []

    for term in sorted(raw_terms):
        norm_term = normalize(term)
        if norm_term not in seen_norm:
            seen_norm.add(norm_term)
            index_terms.append(term)

    index_embeddings = embedder.encode(index_terms, normalize_embeddings=True)

    return index_terms, index_embeddings


def generate_query_variants(term: str, llm, n: int = 6) -> list[str]:
    """
    Ask the local LLM for several English synonyms / semantic variants of
    `term`. These variants do not need to exist in the output vocabulary -
    they are only used as extra search queries against the en_synonym vector
    index (generative query expansion).
    """
    messages = [
        SystemMessage(
            content=(
                "You are a domain terminology expert. "
                "Generate English synonyms or closely related semantic variants "
                "for the given academic/domain term. These variants will be used "
                "as search queries against a vocabulary index, so include "
                "broader, narrower, and near-equivalent phrasings. "
                f"Return only a JSON list of {n} short variants. "
                "Do not explain."
            )
        ),
        HumanMessage(content=f'Term: "{term}"'),
    ]

    try:
        response = llm.invoke(messages)
        variants = parse_candidates(response.content)[:n]
    except Exception:
        variants = []

    if not variants:
        variants = [term]

    return variants


def expansion_retrieval_baseline(
    term: str,
    llm,
    embedder,
    index_terms: list[str],
    index_embeddings: np.ndarray,
    top_k: int = 5,
    n_variants: int = 6,
    per_variant_k: int = 10,
) -> list[str]:
    """
    Generative query expansion + embedding retrieval:

    1. Ask the LLM for several synonym/semantic-variant queries of `term`.
    2. Embed each variant.
    3. Search each variant embedding against the precomputed en_synonym
       vector index.
    4. Aggregate across variants: an index term's score is the best
       (max) cosine similarity it achieved across all variant queries, with
       a small bonus for appearing in multiple variants' top-k results
       (embedding similarity remains the dominant signal; cross-variant
       agreement is only a tie-breaker).
    5. Return the top-5 aggregated candidates.
    """
    variants = generate_query_variants(term, llm, n=n_variants)

    variant_embeddings = embedder.encode(variants, normalize_embeddings=True)
    scores = np.dot(variant_embeddings, index_embeddings.T)  # (n_variants, n_index)

    norm_term = normalize(term)
    best_score = {}
    hit_count = {}

    for variant_scores in scores:
        ranked_indices = np.argsort(variant_scores)[::-1][:per_variant_k]

        for idx in ranked_indices:
            candidate = index_terms[idx]
            norm_candidate = normalize(candidate)

            if norm_candidate == norm_term:
                continue

            score = float(variant_scores[idx])

            if norm_candidate not in best_score or score > best_score[norm_candidate]:
                best_score[norm_candidate] = score

            hit_count[norm_candidate] = hit_count.get(norm_candidate, 0) + 1

    # Recover a display string per normalized candidate (first occurrence).
    display_by_norm = {}
    for term_text in index_terms:
        norm_candidate = normalize(term_text)
        if norm_candidate in best_score and norm_candidate not in display_by_norm:
            display_by_norm[norm_candidate] = term_text

    ranked_norms = sorted(
        best_score.keys(),
        key=lambda n: best_score[n] + 0.05 * (hit_count[n] - 1),
        reverse=True,
    )

    results = [display_by_norm[n] for n in ranked_norms[:top_k]]

    return results


def compute_metrics(ground_truth, candidates, embedder):
    if not candidates:
        candidates = [""]

    norm_gt = normalize(ground_truth)
    norm_candidates = [normalize(c) for c in candidates]

    exact_top1 = norm_candidates[0] == norm_gt
    exact_any = norm_gt in norm_candidates
    top_3_accuracy = norm_gt in norm_candidates[:3]
    top_5_accuracy = norm_gt in norm_candidates[:5]

    fuzzy_scores = [
        fuzz.ratio(norm_gt, normalize(c)) / 100.0
        for c in candidates
    ]
    max_fuzzy = max(fuzzy_scores) if fuzzy_scores else 0.0

    texts = [ground_truth] + candidates
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    gt_emb = embeddings[0:1]
    cand_embs = embeddings[1:]
    semantic_scores = cosine_similarity(gt_emb, cand_embs)[0]
    max_semantic = float(np.max(semantic_scores)) if len(semantic_scores) else 0.0

    mrr = 0.0
    for idx, cand in enumerate(norm_candidates, start=1):
        if cand == norm_gt:
            mrr = 1.0 / idx
            break

    return {
        "exact_top1": exact_top1,
        "exact_any": exact_any,
        "top_3_accuracy": top_3_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "fuzzy_similarity": max_fuzzy,
        "semantic_similarity": max_semantic,
        "mrr": mrr,
    }


def summarize_results(results_df):
    return {
        "rows": len(results_df),
        "exact_top1": float(results_df["exact_top1"].mean()),
        "exact_any": float(results_df["exact_any"].mean()),
        "top_3_accuracy": float(results_df["top_3_accuracy"].mean()),
        "top_5_accuracy": float(results_df["top_5_accuracy"].mean()),
        "mean_fuzzy_similarity": float(results_df["fuzzy_similarity"].mean()),
        "mean_semantic_similarity": float(results_df["semantic_similarity"].mean()),
        "mean_mrr": float(results_df["mrr"].mean()),
        "avg_time_seconds": float(results_df["time_seconds"].mean()),
    }


def run(limit, model_name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH) 

    if limit is not None:
        df = df.head(limit)

    print(f"Rows: {len(df)}")
    print("Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"Loading Ollama model: {model_name}")
    llm = ChatOllama(model=model_name, temperature=0)

    print("Building output vector index from en_synonym...")
    index_terms, index_embeddings = build_output_vector_index(df, embedder)

    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    limit_label = f"limit_{limit}" if limit is not None else "full"
    run_id = f"llm_expansion_retrieval_{safe_model_name}_{limit_label}"

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        term = row["en"]
        ground_truth = row["en_synonym"]

        start = time.time()

        try:
            candidates = expansion_retrieval_baseline(
                term, llm, embedder, index_terms, index_embeddings, top_k=5,
            )
        except Exception as e:
            print(f"Error with term '{term}': {e}")
            candidates = []

        elapsed = time.time() - start

        metrics = compute_metrics(ground_truth, candidates, embedder)

        rows.append({
            "topic": row["topic"],
            "term": term,
            "ground_truth": ground_truth,
            "method": run_id,
            "prediction": candidates[0] if candidates else "",
            "candidates": json.dumps(candidates, ensure_ascii=False),
            "num_final_candidates": len(candidates),
            "time_seconds": elapsed,
            **metrics,
        })

    results_df = pd.DataFrame(rows)

    output_csv = RESULTS_DIR / f"{run_id}_results.csv"
    output_json = RESULTS_DIR / f"{run_id}_summary.json"

    results_df.to_csv(output_csv, index=False)

    summary = summarize_results(results_df)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print(f"\nSaved detailed results to: {output_csv}")
    print(f"Saved summary to: {output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default="qwen3.5:latest")
    args = parser.parse_args()

    run(limit=args.limit, model_name=args.model)


if __name__ == "__main__":
    main()
