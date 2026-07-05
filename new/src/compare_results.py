import json
from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")


def main():
    summaries = []

    for path in RESULTS_DIR.glob("*_summary.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        method = path.name.replace("_summary.json", "")
        data["method"] = method
        summaries.append(data)

    df = pd.DataFrame(summaries)

    preferred_cols = [
        "method",
        "rows",
        "exact_top1",
        "exact_any",
        "top_3_accuracy",
        "top_5_accuracy",
        "mean_fuzzy_similarity",
        "mean_semantic_similarity",
        "mean_mrr",
        "avg_time_seconds",
    ]

    df = df[preferred_cols]
    df = df.sort_values(by="mean_semantic_similarity", ascending=False)

    print(df.to_string(index=False))

    output_path = RESULTS_DIR / "comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()