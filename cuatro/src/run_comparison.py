import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm

from src.data import load_synonym_dataset_grouped
from src.baselines import same_term_baseline, llm_single_prompt_baseline, llm_single_prompt_large_baseline
from src.graph import build_synonym_graph
from src.evaluation import evaluate_prediction

_METHOD_NAMES = {
    "same": "same_term_baseline",
    "llm_small": "llm_single_prompt_small",
    "llm_large": "llm_single_prompt_large",
    "mas": "mas_parallel",
}


def _run_same(term: str, topic: str) -> Dict[str, Any]:
    pred = same_term_baseline(term, topic)
    return {"method": _METHOD_NAMES["same"], "prediction": pred, "candidates": [pred], "domain": "", "term_type": ""}


def _run_llm_small(term: str, topic: str) -> Dict[str, Any]:
    pred = llm_single_prompt_baseline(term, topic)
    return {"method": _METHOD_NAMES["llm_small"], "prediction": pred, "candidates": [pred], "domain": "", "term_type": ""}


def _run_llm_large(term: str, topic: str) -> Dict[str, Any]:
    pred = llm_single_prompt_large_baseline(term, topic)
    return {"method": _METHOD_NAMES["llm_large"], "prediction": pred, "candidates": [pred], "domain": "", "term_type": ""}


def _run_mas(app, term: str, topic: str) -> Dict[str, Any]:
    result = app.invoke({"term": term, "topic": topic, "log": [], "has_reflected": False})
    ranked = result.get("ranked_candidates", [])
    candidates = [
        item["candidate"] for item in ranked if isinstance(item, dict) and "candidate" in item
    ]
    if not candidates:
        candidates = result.get("filtered_candidates", result.get("candidate_pool", []))
    prediction = candidates[0] if candidates else ""
    return {
        "method": _METHOD_NAMES["mas"],
        "prediction": prediction,
        "candidates": candidates,
        "domain": result.get("domain", ""),
        "term_type": result.get("term_type", ""),
    }


def _build_row(
    topic: str,
    term: str,
    ground_truths: List[str],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = evaluate_prediction(
        prediction=output["prediction"],
        ground_truths=ground_truths,
        candidates=output["candidates"],
    )
    return {
        "topic": topic,
        "term": term,
        "ground_truths": json.dumps(ground_truths, ensure_ascii=False),
        "method": output["method"],
        "domain": output.get("domain", ""),
        "term_type": output.get("term_type", ""),
        "prediction": output["prediction"],
        "candidates": json.dumps(output["candidates"], ensure_ascii=False),
        **metrics,
    }


def run_comparison(
    limit: int = 10,
    offset: int = 0,
    methods: List[str] | None = None,
    output_path: str = "results/comparison_results.csv",
    resume: bool = True,
    use_strong_model: bool = False,
    use_reflection: bool = False,
    reflection_threshold: float = 0.45,
) -> pd.DataFrame:
    if methods is None:
        methods = ["same", "llm_small", "llm_large", "mas"]

    df = load_synonym_dataset_grouped().copy()
    if offset:
        df = df.iloc[offset:]
    if limit > 0:
        df = df.head(limit)

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
        app = build_synonym_graph(
            use_strong_model=use_strong_model,
            use_reflection=use_reflection,
            reflection_threshold=reflection_threshold,
        )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Comparison"):
        term = str(row["en"])
        topic = str(row["topic"])
        ground_truths: List[str] = row["en_synonyms"]

        new_outputs: List[Dict[str, Any]] = []

        if "same" in methods and (term, _METHOD_NAMES["same"]) not in done_pairs:
            new_outputs.append(_run_same(term, topic))

        if "llm_small" in methods and (term, _METHOD_NAMES["llm_small"]) not in done_pairs:
            try:
                new_outputs.append(_run_llm_small(term, topic))
            except Exception as e:
                tqdm.write(f"ERROR [llm_small] '{term}': {e}")

        if "llm_large" in methods and (term, _METHOD_NAMES["llm_large"]) not in done_pairs:
            try:
                new_outputs.append(_run_llm_large(term, topic))
            except Exception as e:
                tqdm.write(f"ERROR [llm_large] '{term}': {e}")

        if "mas" in methods and (term, _METHOD_NAMES["mas"]) not in done_pairs:
            try:
                new_outputs.append(_run_mas(app, term, topic))
            except Exception as e:
                tqdm.write(f"ERROR [mas] '{term}': {e}")

        if not new_outputs:
            continue

        for output in new_outputs:
            rows.append(_build_row(topic, term, ground_truths, output))
            done_pairs.add((term, output["method"]))

        out_path.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    agg_cols = {
        "n": ("term", "count"),
        "mrr": ("mrr", "mean"),
        "mrr_fuzzy": ("mrr_fuzzy", "mean"),
        "exact_match": ("exact_match", "mean"),
        "semantic_similarity": ("semantic_similarity", "mean"),
        "top_5_accuracy": ("top_5_accuracy", "mean"),
        "top_10_accuracy": ("top_10_accuracy", "mean"),
        "top_10_fuzzy_accuracy": ("top_10_fuzzy_accuracy", "mean"),
    }
    # Only aggregate columns that exist in the dataframe
    valid_agg = {k: v for k, v in agg_cols.items() if v[0] in results.columns}

    summary = (
        results.groupby("method")
        .agg(**valid_agg)
        .reset_index()
    )

    print("\n=== SUMMARY BY METHOD ===")
    print(summary.to_string(index=False))
    print("\nPrimary metric: mrr (Mean Reciprocal Rank — rewards better ranking)")
    print("Secondary:      top_10_accuracy, top_5_accuracy")
    print("Tertiary:       exact_match, semantic_similarity (comparable with first/ and second/)")

    empty_mask = (results["prediction"] == "") | results["prediction"].isna()
    if empty_mask.any():
        print(f"\nWARNING: Empty predictions: {empty_mask.sum()} row(s)")
        print(results.loc[empty_mask, ["term", "method"]].to_string(index=False))

    print("\n=== SAMPLE RESULTS (first 20 rows) ===")
    cols = ["term", "ground_truths", "method", "prediction", "mrr", "top_10_accuracy", "semantic_similarity"]
    available = [c for c in cols if c in results.columns]
    print(results[available].head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare synonym methods in cuatro/.")
    parser.add_argument("--limit", type=int, default=10, help="Max unique terms (0 = full dataset)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["same", "llm_small", "llm_large", "mas"],
        choices=list(_METHOD_NAMES.keys()),
    )
    parser.add_argument("--output", type=str, default="results/comparison_results.csv")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--model-strong", action="store_true", help="Use deepseek-r1:8b for MAS critic")
    parser.add_argument("--reflect", action="store_true", help="Enable reflection loop in MAS")
    parser.add_argument("--reflect-threshold", type=float, default=0.45)

    args = parser.parse_args()

    results = run_comparison(
        limit=args.limit,
        offset=args.offset,
        methods=args.methods,
        output_path=args.output,
        resume=not args.no_resume,
        use_strong_model=args.model_strong,
        use_reflection=args.reflect,
        reflection_threshold=args.reflect_threshold,
    )

    print_summary(results)
    print(f"\nSaved to: {args.output}")
