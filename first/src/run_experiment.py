import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import load_synonym_dataset
from src.graph import build_synonym_graph
from src.evaluation import exact_match, semantic_similarity


def run_experiment(limit: int = 10) -> pd.DataFrame:
    df = load_synonym_dataset().head(limit).copy()

    app = build_synonym_graph()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        term = row["en"]
        topic = row["topic"]
        ground_truth = row["en_synonym"]

        result = app.invoke(
            {
                "term": term,
                "topic": topic,
                "log": [],
            }
        )

        prediction = result.get("final_answer", "")

        rows.append(
            {
                "topic": topic,
                "term": term,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "exact_match": exact_match(prediction, ground_truth),
                "semantic_similarity": semantic_similarity(prediction, ground_truth),
                "candidates": result.get("candidates", []),
                "filtered_candidates": result.get("filtered_candidates", []),
                "ranked_candidates": result.get("ranked_candidates", []),
            }
        )

    results_df = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)

    output_path = Path("results/mas_langgraph_results.csv")
    results_df.to_csv(output_path, index=False)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    results = run_experiment(limit=args.limit)

    print("\n=== METRICS ===")
    print("Rows:", len(results))
    print("Exact Match:", results["exact_match"].mean())
    print("Average Semantic Similarity:", results["semantic_similarity"].mean())

    print("\nSaved to results/mas_langgraph_results.csv")