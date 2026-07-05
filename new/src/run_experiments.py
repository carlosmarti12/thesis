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


DATA_PATH = Path("data/topic_synonyms_clean.csv")
RESULTS_DIR = Path("results")


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


def compute_metrics(term, ground_truth, candidates, embedder):
    if not candidates:
        candidates = [""]

    norm_gt = normalize(ground_truth)
    norm_candidates = [normalize(c) for c in candidates]

    exact_top1 = norm_candidates[0] == norm_gt
    exact_any = norm_gt in norm_candidates

    fuzzy_scores = [
        fuzz.ratio(norm_gt, normalize(c)) / 100.0
        for c in candidates
    ]
    max_fuzzy = max(fuzzy_scores) if fuzzy_scores else 0.0

    # Semantic similarity between ground truth and each candidate
    texts = [ground_truth] + candidates
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    gt_emb = embeddings[0:1]
    cand_embs = embeddings[1:]

    semantic_scores = cosine_similarity(gt_emb, cand_embs)[0]
    max_semantic = float(np.max(semantic_scores)) if len(semantic_scores) else 0.0

    top_3_accuracy = norm_gt in norm_candidates[:3]
    top_5_accuracy = norm_gt in norm_candidates[:5]

    mrr = 0.0
    for idx, cand in enumerate(norm_candidates, start=1):
        if cand == norm_gt:
            mrr = 1.0 / idx
            break

    return {
        "exact_top1": exact_top1,
        "exact_any": exact_any,
        "fuzzy_similarity": max_fuzzy,
        "semantic_similarity": max_semantic,
        "top_3_accuracy": top_3_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "mrr": mrr,
    }


def same_term_baseline(term: str) -> list[str]:
    return [term]


def llm_single_prompt_baseline(term: str, llm) -> list[str]:
    messages = [
        SystemMessage(
            content=(
                "You are a domain terminology expert. "
                "Your task is to propose English synonyms for a given academic/domain term. "
                "Return only a JSON list of 5 short English candidate synonyms. "
                "Do not explain."
            )
        ),
        HumanMessage(
            content=f'Term: "{term}"'
        ),
    ]

    response = llm.invoke(messages)
    return parse_candidates(response.content)[:5]


def embedding_retrieval_baseline(term: str, candidate_pool: list[str], embedder, top_k: int = 5) -> list[str]:
    """
    Retrieval baseline:
    compares the input term against a vocabulary of known terms and returns nearest candidates.

    Warning:
    this uses the dataset vocabulary, so in the thesis you should describe it as a retrieval baseline,
    not as pure free generation.
    """
    all_texts = [term] + candidate_pool
    embeddings = embedder.encode(all_texts, normalize_embeddings=True)

    term_emb = embeddings[0:1]
    pool_embs = embeddings[1:]

    scores = cosine_similarity(term_emb, pool_embs)[0]
    ranked_indices = np.argsort(scores)[::-1]

    results = []
    seen = set()

    for idx in ranked_indices:
        candidate = candidate_pool[idx]
        norm_candidate = normalize(candidate)

        if norm_candidate == normalize(term):
            continue

        if norm_candidate not in seen:
            results.append(candidate)
            seen.add(norm_candidate)

        if len(results) >= top_k:
            break

    return results


def summarize_results(results_df: pd.DataFrame) -> dict:
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


def run_experiment(method: str, limit: int | None, model_name: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if limit is not None:
        df = df.head(limit)

    print(f"Running method: {method}")
    print(f"Rows: {len(df)}")

    print("Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    llm = None
    if method == "llm":
        print(f"Loading Ollama model: {model_name}")
        llm = ChatOllama(model=model_name, temperature=0)

    candidate_pool = sorted(
        set(
            df["en"].dropna().astype(str).tolist()
            + df["en_synonym"].dropna().astype(str).tolist()
        )
    )

    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    limit_label = f"limit_{limit}" if limit is not None else "full"

    if method == "llm":
        run_id = f"{method}_{safe_model_name}_{limit_label}"
    else:
        run_id = f"{method}_{limit_label}"

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        term = row["en"]
        ground_truth = row["en_synonym"]

        start = time.time()

        try:
            if method == "same_term":
                candidates = same_term_baseline(term)

            elif method == "llm":
                candidates = llm_single_prompt_baseline(term, llm)

            elif method == "embedding":
                candidates = embedding_retrieval_baseline(term, candidate_pool, embedder, top_k=5)

            else:
                raise ValueError(f"Unknown method: {method}")

        except Exception as e:
            print(f"Error with term '{term}': {e}")
            candidates = []

        elapsed = time.time() - start

        metrics = compute_metrics(term, ground_truth, candidates, embedder)

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

    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    limit_label = f"limit_{limit}" if limit is not None else "full"

    if method == "llm":
        run_id = f"{method}_{safe_model_name}_{limit_label}"
    else:
        run_id = f"{method}_{limit_label}"

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
    parser.add_argument(
        "--method",
        choices=["same_term", "llm", "embedding"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default="qwen3.5:latest")

    args = parser.parse_args()

    run_experiment(
        method=args.method,
        limit=args.limit,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()