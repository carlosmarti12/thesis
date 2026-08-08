"""
CLI harness: run one method over the full dataset (use --limit to cap how
many rows) and write results/{run_id}_results.csv + _summary.json.

Supports resume: if the output CSV already exists, terms already present are
skipped and the file is saved incrementally after every row, so a long MAS
run can be safely interrupted and continued later (--no-resume to start
fresh).

`en_synonym` is read exactly once per row, inside this file's main loop,
strictly AFTER `candidates = ...` has already been produced by the method -
grep this file for `en_synonym` to confirm it never appears before that line.
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
from .evaluation import compute_metrics, summarize_results
from .graph import build_mas_graph, empty_state
from .open_vocab import get_wordnet_index
from .supervisor import run_supervisor

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NEEDS_LLM = {
    "llm_zero_shot", "llm_expansion", "llm_rerank", "hybrid_fusion",
    "mas_base", "mas_llm_ranker", "mas_safe_hybrid", "supervisor",
}
NEEDS_WORDNET_INDEX = {
    "embedding_wordnet", "llm_expansion", "hybrid_fusion",
    "mas_base", "mas_llm_ranker", "mas_safe_hybrid", "supervisor",
}
MAS_VARIANT_BY_METHOD = {
    "mas_base": "base",
    "mas_llm_ranker": "llm_ranker",
    "mas_safe_hybrid": "safe_hybrid",
}
ALL_METHODS = [
    "wordnet_direct", "embedding_wordnet",
    "llm_zero_shot", "llm_expansion", "llm_rerank", "hybrid_fusion",
    "mas_base", "mas_llm_ranker", "mas_safe_hybrid",
    "supervisor",
]


def get_candidates(method: str, term: str, resources: dict) -> list[str]:
    if method == "wordnet_direct":
        return baselines.wordnet_direct_baseline(term)

    if method == "embedding_wordnet":
        return baselines.embedding_wordnet_baseline(
            term, resources["embedder"], resources["index_terms"], resources["index_embeddings"], top_k=5
        )

    if method == "llm_zero_shot":
        return baselines.llm_zero_shot_baseline(term, model=resources["model"], top_k=5)

    if method == "llm_expansion":
        return baselines.llm_expansion_baseline(
            term, resources["embedder"], resources["index_terms"], resources["index_embeddings"],
            model=resources["model"], top_k=5,
        )

    if method == "llm_rerank":
        return baselines.llm_generate_rerank_baseline(
            term, resources["embedder"], model=resources["model"], n_generate=10, top_k=5,
        )

    if method == "hybrid_fusion":
        return baselines.hybrid_fusion_baseline(
            term, resources["embedder"], resources["index_terms"], resources["index_embeddings"],
            model=resources["model"], top_k=5,
        )

    if method in MAS_VARIANT_BY_METHOD: # comprueba si el metodo es alguno de las MAS
        app = resources["mas_apps"][method]
        result = app.invoke(empty_state(term))
        return result.get("final_candidates", [])

    if method == "supervisor":
        final_state = run_supervisor(
            term,
            embedder=resources["embedder"],
            candidate_pool_terms=resources["index_terms"],
            candidate_pool_embeddings=resources["index_embeddings"],
            model=resources["model"],
        )
        return final_state.get("final_candidates", [])

    raise ValueError(f"Unknown method: {method}")


def build_resources(method: str, model_name: str) -> dict:
    resources = {"model": model_name}

    print("Loading embedding model...")
    resources["embedder"] = SentenceTransformer(EMBEDDER_NAME)

    if method in NEEDS_WORDNET_INDEX:
        print("Loading/building WordNet vocabulary index (cached after first run)...")
        index_terms, index_embeddings = get_wordnet_index(resources["embedder"], EMBEDDER_NAME)
        resources["index_terms"] = index_terms
        resources["index_embeddings"] = index_embeddings

    if method in MAS_VARIANT_BY_METHOD:
        variant = MAS_VARIANT_BY_METHOD[method]
        app = build_mas_graph(
            embedder=resources["embedder"],
            candidate_pool_terms=resources["index_terms"],
            candidate_pool_embeddings=resources["index_embeddings"],
            model=model_name,
            variant=variant,
        )
        resources["mas_apps"] = {method: app}

    return resources


def run_experiment(
    method: str,
    limit: int = 0,
    model_name: str = "llama3.2:3b",
    output_path: str | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_full_dataset()
    if limit > 0:
        df = df.head(limit)

    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    limit_label = f"limit_{limit}" if limit > 0 else "full"
    run_id = f"{method}_{safe_model_name}_{limit_label}" if method in NEEDS_LLM else f"{method}_{limit_label}"

    out_path = Path(output_path) if output_path else RESULTS_DIR / f"{run_id}_results.csv"

    rows: list = []
    done_terms: set = set()
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        done_terms = set(existing["term"].astype(str))
        print(f"Resuming: {len(done_terms)} term(s) already saved.")

    print(f"Running method: {method}")
    print(f"Rows: {len(df)}")

    resources = build_resources(method, model_name)
    embedder = resources["embedder"]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        term = str(row["en"])

        if term in done_terms:
            continue

        start = time.time()
        try:
            candidates = get_candidates(method, term, resources)
        except Exception as e:
            tqdm.write(f"ERROR with term '{term}': {e}")
            candidates = []
        elapsed = time.time() - start

        # en_synonym is read here, AFTER candidates were produced above, and
        # ONLY for scoring - never passed into get_candidates()/baselines.py/agents.py.
        ground_truth = str(row["en_synonym"])
        row_metrics = compute_metrics(ground_truth, candidates, embedder)

        rows.append({
            "term": term,
            "ground_truth": ground_truth,
            "method": run_id,
            "prediction": candidates[0] if candidates else "",
            "candidates": json.dumps(candidates, ensure_ascii=False),
            "num_final_candidates": len(candidates),
            "time_seconds": elapsed,
            **row_metrics,
        })
        done_terms.add(term)

        # Incremental save after every term so a long MAS run is never lost.
        pd.DataFrame(rows).to_csv(out_path, index=False)

    results_df = pd.DataFrame(rows)

    summary = summarize_results(results_df)
    summary["method"] = run_id
    output_json = out_path.with_name(out_path.stem.replace("_results", "") + "_summary.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved detailed results to: {out_path}")
    print(f"Saved summary to: {output_json}")

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=ALL_METHODS, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Max terms (0 = full dataset, ~970 rows)")
    parser.add_argument("--model", type=str, default="llama3.2:3b")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore existing CSV")
    args = parser.parse_args()

    run_experiment(
        method=args.method,
        limit=args.limit,
        model_name=args.model,
        output_path=args.output,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
