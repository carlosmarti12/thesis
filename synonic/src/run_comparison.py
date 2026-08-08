"""
Run every method over the full dataset in a single pass (one embedder,
one WordNet index, one set of MAS graphs - built once, reused per term) and
write a single results/comparison_results.csv with one row per
(term, method) pair. Use --limit to cap how many rows are processed.

Resume: if the output CSV already exists, (term, method) pairs already
present are skipped, and the file is saved incrementally after every term -
so a long run covering all 8 methods can be interrupted and continued
(--no-resume to start fresh).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from . import baselines
from .data import load_full_dataset
from .evaluation import compute_metrics
from .graph import build_mas_graph, empty_state
from .open_vocab import get_wordnet_index
from .run_experiment import ALL_METHODS, MAS_VARIANT_BY_METHOD, get_candidates

RESULTS_DIR = Path(__file__).parent.parent / "results"
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_all_resources(methods: list[str], model_name: str) -> dict:
    resources = {"model": model_name}

    print("Loading embedding model...")
    resources["embedder"] = SentenceTransformer(EMBEDDER_NAME)

    if any(m in {"embedding_wordnet", "llm_expansion", "supervisor", *MAS_VARIANT_BY_METHOD} for m in methods):
        print("Loading/building WordNet vocabulary index (cached after first run)...")
        index_terms, index_embeddings = get_wordnet_index(resources["embedder"], EMBEDDER_NAME)
        resources["index_terms"] = index_terms
        resources["index_embeddings"] = index_embeddings

    resources["mas_apps"] = {}
    for method, variant in MAS_VARIANT_BY_METHOD.items():
        if method in methods:
            resources["mas_apps"][method] = build_mas_graph(
                embedder=resources["embedder"],
                candidate_pool_terms=resources["index_terms"],
                candidate_pool_embeddings=resources["index_embeddings"],
                model=model_name,
                variant=variant,
            )

    return resources


def run_comparison(
    limit: int = 0,
    model_name: str = "llama3.2:3b",
    methods: list[str] | None = None,
    output_path: str = "results/comparison_results.csv",
    resume: bool = True,
) -> pd.DataFrame:
    methods = methods or ALL_METHODS

    df = load_full_dataset()
    if limit > 0:
        df = df.head(limit)

    out_path = Path(output_path)
    rows: list = []
    done_pairs: set = set()

    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        for _, r in existing.iterrows():
            done_pairs.add((str(r["term"]), str(r["method"])))
        print(f"Resuming: {len(done_pairs)} result(s) already saved.")

    resources = build_all_resources(methods, model_name)
    embedder = resources["embedder"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        term = str(row["en"])
        # en_synonym is read here, per term, only for scoring after each
        # method's candidates are already produced below.
        ground_truth = str(row["en_synonym"])

        for method in methods:
            if (term, method) in done_pairs:
                continue

            start = time.time()
            try:
                candidates = get_candidates(method, term, resources)
            except Exception as e:
                tqdm.write(f"ERROR [{method}] '{term}': {e}")
                candidates = []
            elapsed = time.time() - start

            row_metrics = compute_metrics(ground_truth, candidates, embedder)

            rows.append({
                "term": term,
                "ground_truth": ground_truth,
                "method": method,
                "prediction": candidates[0] if candidates else "",
                "candidates": json.dumps(candidates, ensure_ascii=False),
                "num_final_candidates": len(candidates),
                "time_seconds": elapsed,
                **row_metrics,
            })
            done_pairs.add((term, method))

        # Incremental save after every term (all its methods) so progress
        # is never lost on a long multi-method run.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    summary = (
        results.groupby("method")
        .agg(
            n=("term", "count"),
            avg_num_candidates=("num_final_candidates", "mean"),
            hit_at_1=("hit@1", "mean"),
            exact_any=("exact_any", "mean"),
            hit_at_3=("hit@3", "mean"),
            hit_at_5=("hit@5", "mean"),
            ndcg_at_3=("ndcg@3", "mean"),
            ndcg_at_5=("ndcg@5", "mean"),
            mrr=("mrr", "mean"),
            fuzzy_similarity=("fuzzy_similarity", "mean"),
            semantic_similarity=("semantic_similarity", "mean"),
            avg_time_seconds=("time_seconds", "mean"),
        )
        .reset_index()
    )
    print("\n=== SUMMARY BY METHOD ===")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max terms (0 = full dataset, ~970 rows)")
    parser.add_argument("--model", type=str, default="llama3.2:3b")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--output", type=str, default="results/comparison_results.csv")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    results = run_comparison(
        limit=args.limit,
        model_name=args.model,
        methods=args.methods,
        output_path=args.output,
        resume=not args.no_resume,
    )

    print_summary(results)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
