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


# Internal method name constants
_METHOD_NAMES = {
    "same": "same_term_baseline",
    "llm": "llm_single_prompt_baseline",
    "mas": "mas_langgraph",
}


def run_same_term_method(term: str, topic: str) -> Dict[str, Any]:
    prediction = same_term_baseline(term, topic)
    return {
        "method": _METHOD_NAMES["same"],
        "prediction": prediction,
        "candidates": [prediction],
    }


def run_llm_single_prompt_method(term: str, topic: str) -> Dict[str, Any]:
    prediction = llm_single_prompt_baseline(term, topic)
    return {
        "method": _METHOD_NAMES["llm"],
        "prediction": prediction,
        "candidates": [prediction],
    }


def run_mas_langgraph_method(app, term: str, topic: str) -> Dict[str, Any]:
    result = app.invoke({"term": term, "topic": topic, "log": []})

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

    # top-1 ranked candidate used as point prediction for single-value metrics
    prediction = candidates[0] if candidates else ""

    return {
        "method": _METHOD_NAMES["mas"],
        "prediction": prediction,
        "candidates": candidates,
        "domain": result.get("domain", ""),
    }


def _build_row(topic, term, ground_truth, output: Dict[str, Any]) -> Dict[str, Any]:
    metrics = evaluate_prediction(
        prediction=output["prediction"],
        ground_truth=ground_truth,
        candidates=output["candidates"],
    )
    return {
        "topic": topic,
        "term": term,
        "ground_truth": ground_truth,
        "method": output["method"],
        "domain": output.get("domain", ""),
        "prediction": output["prediction"],
        "candidates": json.dumps(output["candidates"], ensure_ascii=False),
        "exact_match": metrics["exact_match"],
        "fuzzy_similarity": metrics["fuzzy_similarity"],
        "semantic_similarity": metrics["semantic_similarity"],
        "top_3_accuracy": metrics["top_3_accuracy"],
        "top_5_accuracy": metrics["top_5_accuracy"],
        "top_8_accuracy": metrics["top_8_accuracy"],
        "top_8_fuzzy_accuracy": metrics["top_8_fuzzy_accuracy"],
    }


def run_comparison(
    limit: int = 10,
    offset: int = 0,
    methods: List[str] | None = None,
    output_path: str = "results/comparison_results.csv",
    resume: bool = True,
) -> pd.DataFrame:
    if methods is None:
        methods = ["same", "llm", "mas"]

    df = load_synonym_dataset().copy()
    if offset:
        df = df.iloc[offset:]
    if limit > 0:
        df = df.head(limit)

    # Resume: load existing results and skip already-done (term, method) pairs
    rows: List[Dict] = []
    done_pairs: set = set()
    out_path = Path(output_path)

    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        for _, r in existing.iterrows():
            done_pairs.add((str(r["term"]), str(r["method"])))
        print(f"Resuming: {len(done_pairs)} result(s) already saved.")

    app = None
    if "mas" in methods:
        app = build_synonym_graph()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        term = str(row["en"])
        topic = str(row["topic"])
        ground_truth = str(row["en_synonym"])

        new_outputs: List[Dict[str, Any]] = []

        if "same" in methods and (term, _METHOD_NAMES["same"]) not in done_pairs:
            new_outputs.append(run_same_term_method(term, topic))

        if "llm" in methods and (term, _METHOD_NAMES["llm"]) not in done_pairs:
            try:
                new_outputs.append(run_llm_single_prompt_method(term, topic))
            except Exception as e:
                tqdm.write(f"ERROR [llm] '{term}': {e}")

        if "mas" in methods and (term, _METHOD_NAMES["mas"]) not in done_pairs:
            try:
                new_outputs.append(run_mas_langgraph_method(app, term, topic))
            except Exception as e:
                tqdm.write(f"ERROR [mas] '{term}': {e}")

        if not new_outputs:
            continue

        for output in new_outputs:
            rows.append(_build_row(topic, term, ground_truth, output))
            done_pairs.add((term, output["method"]))

        # Incremental save after every term so progress is never lost
        out_path.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    summary = (
        results.groupby("method")
        .agg(
            n=("term", "count"),
            exact_match=("exact_match", "mean"),
            semantic_similarity=("semantic_similarity", "mean"),
            top_5_accuracy=("top_5_accuracy", "mean"),
            top_8_accuracy=("top_8_accuracy", "mean"),
            top_8_fuzzy_accuracy=("top_8_fuzzy_accuracy", "mean"),
        )
        .reset_index()
    )

    print("\n=== SUMMARY BY METHOD ===")
    print(summary.to_string(index=False))
    print("\nKey metrics:")
    print("  top_8_accuracy       — ground truth appears exactly in the 8 candidates (MAS primary metric)")
    print("  top_8_fuzzy_accuracy — same but allowing fuzzy partial matches (token_set_ratio >= 80)")
    print("  For baselines: top_8 == exact_match (single candidate list).")

    empty_mask = (results["prediction"] == "") | results["prediction"].isna()
    if empty_mask.any():
        print(f"\n⚠  Empty predictions: {empty_mask.sum()} row(s)")
        print(results.loc[empty_mask, ["term", "method"]].to_string(index=False))

    print("\n=== SAMPLE RESULTS ===")
    cols = ["term", "ground_truth", "method", "prediction", "top_8_accuracy", "top_8_fuzzy_accuracy", "semantic_similarity"]
    print(results[cols].head(24).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare synonym methods.")
    parser.add_argument("--limit", type=int, default=10, help="Max terms to evaluate (0 = full dataset)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["same", "llm", "mas"],
        choices=["same", "llm", "mas"],
    )
    parser.add_argument("--output", type=str, default="results/comparison_results.csv")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore existing CSV")

    args = parser.parse_args()

    results = run_comparison(
        limit=args.limit,
        offset=args.offset,
        methods=args.methods,
        output_path=args.output,
        resume=not args.no_resume,
    )

    print_summary(results)
    print(f"\nSaved to: {args.output}")
