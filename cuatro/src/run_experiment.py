import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import load_synonym_dataset_grouped
from src.graph import build_synonym_graph
from src.evaluation import evaluate_prediction, FUZZY_THRESHOLD


def run_experiment(
    limit: int = 10,
    offset: int = 0,
    output_path: str = "results/mas_results.csv",
    resume: bool = True,
    use_strong_model: bool = False,
    use_reflection: bool = False,
    reflection_threshold: float = 0.45,
) -> pd.DataFrame:
    df = load_synonym_dataset_grouped().copy()
    if offset:
        df = df.iloc[offset:]
    if limit > 0:
        df = df.head(limit)

    rows: list = []
    done_terms: set = set()
    out_path = Path(output_path)

    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        done_terms = set(existing["term"].astype(str))
        print(f"Resuming: {len(done_terms)} term(s) already saved.")

    app = build_synonym_graph(
        use_strong_model=use_strong_model,
        use_reflection=use_reflection,
        reflection_threshold=reflection_threshold,
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MAS experiment"):
        term = str(row["en"])
        topic = str(row["topic"])
        ground_truths = row["en_synonyms"]

        if term in done_terms:
            continue

        try:
            result = app.invoke({
                "term": term,
                "topic": topic,
                "log": [],
                "has_reflected": False,
            })
        except Exception as e:
            tqdm.write(f"ERROR '{term}': {e}")
            result = {}

        ranked = result.get("ranked_candidates", [])
        candidates = [
            item["candidate"]
            for item in ranked
            if isinstance(item, dict) and "candidate" in item
        ]
        if not candidates:
            candidates = result.get("filtered_candidates", result.get("candidate_pool", []))

        prediction = candidates[0] if candidates else ""
        metrics = evaluate_prediction(prediction, ground_truths, candidates)

        rows.append({
            "topic": topic,
            "term": term,
            "ground_truths": str(ground_truths),
            "domain": result.get("domain", ""),
            "term_type": result.get("term_type", ""),
            "prediction": prediction,
            "candidates": str(candidates),
            **metrics,
        })
        done_terms.add(term)

        out_path.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cuatro MAS experiment.")
    parser.add_argument("--limit", type=int, default=10, help="Max unique terms (0 = full dataset)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/mas_results.csv")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--model-strong", action="store_true", help="Use deepseek-r1:8b for critic")
    parser.add_argument("--reflect", action="store_true", help="Enable reflection loop")
    parser.add_argument("--reflect-threshold", type=float, default=0.45)

    args = parser.parse_args()

    results = run_experiment(
        limit=args.limit,
        offset=args.offset,
        output_path=args.output,
        resume=not args.no_resume,
        use_strong_model=args.model_strong,
        use_reflection=args.reflect,
        reflection_threshold=args.reflect_threshold,
    )

    print("\n=== MAS METRICS ===")
    print(f"Terms evaluated: {len(results)}")
    for col in ["exact_match", "fuzzy_similarity", "semantic_similarity",
                "top_3_accuracy", "top_5_accuracy", "top_10_accuracy",
                "top_10_fuzzy_accuracy", "mrr", "mrr_fuzzy"]:
        if col in results.columns:
            print(f"  {col:<25} {results[col].mean():.4f}")

    print(f"\nSaved to {args.output}")
