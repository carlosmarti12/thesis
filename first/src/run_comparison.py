import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm

from src.data import load_synonym_dataset
from src.baselines import same_term_baseline, llm_single_prompt_baseline
from src.graph import build_synonym_graph
from src.evaluation import evaluate_prediction


def run_same_term_method(term: str, topic: str) -> Dict[str, Any]:
    prediction = same_term_baseline(term, topic)

    return {
        "method": "same_term_baseline",
        "prediction": prediction,
        "candidates": [prediction],
        "raw_result": {},
    }


def run_llm_single_prompt_method(term: str, topic: str) -> Dict[str, Any]:
    prediction = llm_single_prompt_baseline(term, topic)

    return {
        "method": "llm_single_prompt_baseline",
        "prediction": prediction,
        "candidates": [prediction],
        "raw_result": {},
    }


def run_mas_langgraph_method(app, term: str, topic: str) -> Dict[str, Any]:
    result = app.invoke(
        {
            "term": term,
            "topic": topic,
            "log": [],
        }
    )

    prediction = result.get("final_answer", "")

    ranked_candidates = result.get("ranked_candidates", [])
    candidates = [
        item["candidate"]
        for item in ranked_candidates
        if isinstance(item, dict) and "candidate" in item
    ]

    if not candidates:
        candidates = result.get("filtered_candidates", [])

    if not candidates:
        candidates = result.get("candidates", [])

    return {
        "method": "mas_langgraph",
        "prediction": prediction,
        "candidates": candidates,
        "raw_result": result,
    }


def run_comparison(
    limit: int = 10,
    offset: int = 0,
    methods: List[str] | None = None,
    output_path: str = "results/comparison_results.csv",
) -> pd.DataFrame:
    if methods is None:
        methods = ["same", "llm", "mas"]

    df = load_synonym_dataset().copy()

    if offset:
        df = df.iloc[offset:]

    df = df.head(limit)

    app = None
    if "mas" in methods:
        app = build_synonym_graph()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        term = row["en"]
        topic = row["topic"]
        ground_truth = row["en_synonym"]

        method_outputs = []

        if "same" in methods:
            method_outputs.append(run_same_term_method(term, topic))

        if "llm" in methods:
            method_outputs.append(run_llm_single_prompt_method(term, topic))

        if "mas" in methods:
            method_outputs.append(run_mas_langgraph_method(app, term, topic))

        for output in method_outputs:
            prediction = output["prediction"]
            candidates = output["candidates"]

            metrics = evaluate_prediction(
                prediction=prediction,
                ground_truth=ground_truth,
                candidates=candidates,
            )

            rows.append(
                {
                    "topic": topic,
                    "term": term,
                    "ground_truth": ground_truth,
                    "method": output["method"],
                    "prediction": prediction,
                    "candidates": json.dumps(candidates, ensure_ascii=False),
                    "exact_match": metrics["exact_match"],
                    "fuzzy_similarity": metrics["fuzzy_similarity"],
                    "semantic_similarity": metrics["semantic_similarity"],
                    "top_3_accuracy": metrics["top_3_accuracy"],
                    "top_5_accuracy": metrics["top_5_accuracy"],
                }
            )

    results_df = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)

    return results_df


def print_summary(results: pd.DataFrame) -> None:
    summary = (
        results.groupby("method")
        .agg(
            rows=("term", "count"),
            exact_match=("exact_match", "mean"),
            fuzzy_similarity=("fuzzy_similarity", "mean"),
            semantic_similarity=("semantic_similarity", "mean"),
            top_3_accuracy=("top_3_accuracy", "mean"),
            top_5_accuracy=("top_5_accuracy", "mean"),
        )
        .reset_index()
    )

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== SAMPLE RESULTS ===")
    sample_columns = [
        "term",
        "ground_truth",
        "method",
        "prediction",
        "exact_match",
        "semantic_similarity",
    ]
    print(results[sample_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["same", "llm", "mas"],
        choices=["same", "llm", "mas"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison_results.csv",
    )

    args = parser.parse_args()

    results = run_comparison(
        limit=args.limit,
        offset=args.offset,
        methods=args.methods,
        output_path=args.output,
    )

    print_summary(results)

    print(f"\nSaved results to: {args.output}")