import argparse
import csv
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


def embedding_retrieval_baseline(
    term: str,
    embedder,
    index_terms: list[str],
    index_embeddings: np.ndarray,
    top_k: int = 10,
) -> list[str]:
    """
    Closed-world output retrieval:
    encodes only the input term and compares it against the precomputed
    in-memory vector index of output candidates (df['en_synonym']).

    This is the exact retrieval step used by `run_experiments.py`'s
    `embedding` method, just with a wider top_k so the reranker below has
    a few extra candidates (beyond the final top 5) to promote from.
    """
    term_emb = embedder.encode([term], normalize_embeddings=True)[0]
    scores = np.dot(index_embeddings, term_emb)
    ranked_indices = np.argsort(scores)[::-1]

    results = []
    seen = set()
    norm_term = normalize(term)

    for idx in ranked_indices:
        candidate = index_terms[idx]
        norm_candidate = normalize(candidate)

        if norm_candidate == norm_term:
            continue

        if norm_candidate not in seen:
            results.append(candidate)
            seen.add(norm_candidate)

        if len(results) >= top_k:
            break

    return results


def llm_rerank(term: str, candidates: list[str], llm) -> list[str]:
    """
    Ask the LLM to reorder an already-retrieved candidate list from best to
    worst synonym for `term`. The LLM only sees the term and the candidates
    - never the ground truth - so this is a legitimate ranking step, not
    leakage.

    The LLM is only allowed to reorder: it cannot invent new candidates, and
    any candidate it omits from its answer is appended back at the end in
    its original (embedding-similarity) order. This isolates the effect of
    *ranking* from the effect of *recall* - the candidate set is identical
    to the embedding baseline's, only the order can change.
    """
    if len(candidates) <= 1:
        return candidates

    candidate_text = "\n".join(
        f"{i + 1}. {candidate}" for i, candidate in enumerate(candidates)
    )

    messages = [
        SystemMessage(
            content=(
                "You are a strict semantic ranking assistant for domain terminology. "
                "You will receive a term and a list of candidate synonyms for that term. "
                "Order the candidates from the BEST synonym to the WORST synonym. "
                "Do not invent new candidates. "
                "Do not drop any candidate. "
                "Use the exact candidate strings from the list. "
                "Return only a JSON list of the reordered candidate strings."
            )
        ),
        HumanMessage(
            content=f'Term: "{term}"\n\nCandidates:\n{candidate_text}'
        ),
    ]

    try:
        response = llm.invoke(messages)
        llm_ranked = parse_candidates(response.content)
    except Exception:
        llm_ranked = []

    original_by_norm = {
        normalize(candidate): candidate
        for candidate in candidates
        if normalize(candidate)
    }

    final_order = []
    seen = set()

    for item in llm_ranked:
        norm_item = normalize(item)

        if norm_item in original_by_norm and norm_item not in seen:
            final_order.append(original_by_norm[norm_item])
            seen.add(norm_item)

    # Anything the LLM dropped or garbled goes back in its original
    # retrieval order, so a bad LLM response never loses recall vs. the
    # plain embedding baseline.
    for candidate in candidates:
        norm_candidate = normalize(candidate)

        if norm_candidate and norm_candidate not in seen:
            final_order.append(candidate)
            seen.add(norm_candidate)

    return final_order


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


def run(limit, model_name, retrieval_k):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if limit is not None:
        df = df.head(limit)

    print(f"Rows: {len(df)}")
    print("Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"Loading Ollama model: {model_name}")
    # num_predict caps output length: the expected response is a short JSON
    # list of <= retrieval_k candidates (well under 300 tokens). Without
    # this cap, an occasional degenerate generation can loop for
    # thousands of tokens (observed: 39k+ tokens, ~9.5 minutes on a single
    # term) instead of stopping, stalling the whole run.
    llm = ChatOllama(model=model_name, temperature=0, num_predict=300, timeout=60)

    print("Building output vector index from en_synonym...")
    index_terms, index_embeddings = build_output_vector_index(df, embedder)

    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    limit_label = f"limit_{limit}" if limit is not None else "full"
    run_id = f"llm_rerank_{safe_model_name}_{limit_label}"

    output_csv = RESULTS_DIR / f"{run_id}_results.csv"
    output_json = RESULTS_DIR / f"{run_id}_summary.json"

    fieldnames = [
        "topic", "term", "ground_truth", "method", "prediction", "candidates",
        "num_final_candidates", "time_seconds", "exact_top1", "exact_any",
        "top_3_accuracy", "top_5_accuracy", "fuzzy_similarity",
        "semantic_similarity", "mrr",
    ]

    # Resume support: rows already written to output_csv (keyed by
    # topic+term) are skipped, so a crash/hang partway through a long run
    # doesn't throw away already-computed results - just rerun the same
    # command and it picks up where it left off.
    done_keys = set()
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        done_keys = set(zip(existing_df["topic"], existing_df["term"]))
        print(f"Resuming: {len(done_keys)} rows already done in {output_csv}, skipping them.")

    write_header = not output_csv.exists()
    csv_file = open(output_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        term = row["en"]
        ground_truth = row["en_synonym"]
        topic = row["topic"]

        if (topic, term) in done_keys:
            continue

        start = time.time()

        try:
            retrieved = embedding_retrieval_baseline(
                term, embedder, index_terms, index_embeddings, top_k=retrieval_k,
            )
            candidates = llm_rerank(term, retrieved, llm)[:5]
        except Exception as e:
            print(f"Error with term '{term}': {e}")
            candidates = []

        elapsed = time.time() - start

        metrics = compute_metrics(ground_truth, candidates, embedder)

        writer.writerow({
            "topic": topic,
            "term": term,
            "ground_truth": ground_truth,
            "method": run_id,
            "prediction": candidates[0] if candidates else "",
            "candidates": json.dumps(candidates, ensure_ascii=False),
            "num_final_candidates": len(candidates),
            "time_seconds": elapsed,
            **metrics,
        })
        csv_file.flush()

    csv_file.close()

    results_df = pd.read_csv(output_csv)

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
    parser.add_argument("--model", type=str, default="llama3.2:3b")
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="How many candidates the embedding retrieval step fetches before the LLM reranks/truncates to 5.",
    )
    args = parser.parse_args()

    run(limit=args.limit, model_name=args.model, retrieval_k=args.retrieval_k)


if __name__ == "__main__":
    main()
