import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import load_synonym_dataset
from src.graph import build_synonym_graph
from src.evaluation import top_k_contains_ground_truth, semantic_similarity, FUZZY_THRESHOLD


def run_experiment(
    limit: int = 10,
    offset: int = 0,
    output_path: str = "results/mas_langgraph_results.csv",
    resume: bool = True,
) -> pd.DataFrame:
    df = load_synonym_dataset().copy()
    if offset:
        df = df.iloc[offset:]
    if limit > 0:
        df = df.head(limit)

    # Resume: skip terms already in the output file
    rows: list = []
    done_terms: set = set()
    out_path = Path(output_path)

    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        done_terms = set(existing["term"].astype(str))
        print(f"Resuming: {len(done_terms)} term(s) already saved.")

    app = build_synonym_graph()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        term = str(row["en"])
        topic = str(row["topic"])
        ground_truth = str(row["en_synonym"])

        if term in done_terms:
            continue

        try:
            result = app.invoke({"term": term, "topic": topic, "log": []})
        except Exception as e:
            tqdm.write(f"ERROR '{term}': {e}")
            result = {}

        ranked = result.get("ranked_candidates", [])
        final_candidates = [
            item["candidate"]
            for item in ranked
            if isinstance(item, dict) and "candidate" in item
        ]
        if not final_candidates:
            final_candidates = result.get("filtered_candidates", result.get("candidates", []))

        top8_match = top_k_contains_ground_truth(final_candidates, ground_truth, k=8, fuzzy=False)
        top8_fuzzy_match = top_k_contains_ground_truth(final_candidates, ground_truth, k=8, fuzzy=True)
        top1_candidate = final_candidates[0] if final_candidates else ""

        rows.append(
            {
                "topic": topic,
                "term": term,
                "ground_truth": ground_truth,
                "top8_match": top8_match,
                "top8_fuzzy_match": top8_fuzzy_match,
                "top1_candidate": top1_candidate,
                "semantic_similarity": semantic_similarity(top1_candidate, ground_truth),
                "final_candidates": final_candidates,
                "candidates": result.get("candidates", []),
                "filtered_candidates": result.get("filtered_candidates", []),
            }
        )
        done_terms.add(term)

        out_path.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAS LangGraph experiment.")
    parser.add_argument("--limit", type=int, default=10, help="Max terms (0 = full dataset)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/mas_langgraph_results.csv")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")

    args = parser.parse_args()

    results = run_experiment(
        limit=args.limit,
        offset=args.offset,
        output_path=args.output,
        resume=not args.no_resume,
    )

    print("\n=== METRICS ===")
    print(f"Rows:                      {len(results)}")
    print(f"Top-8 Exact Accuracy:      {results['top8_match'].mean():.3f}")
    print(f"Top-8 Fuzzy Accuracy:      {results['top8_fuzzy_match'].mean():.3f}  (token_set_ratio >= {FUZZY_THRESHOLD})")
    print(f"Avg Semantic Similarity:   {results['semantic_similarity'].mean():.3f}")

    empty = results["final_candidates"].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)
    if empty.any():
        print(f"\n⚠  Empty candidate lists: {empty.sum()} row(s)")
        print(results.loc[empty, "term"].tolist())

    print(f"\nSaved to {args.output}")
