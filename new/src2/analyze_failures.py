import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    path = Path(args.file)
    df = pd.read_csv(path)

    failures = df[df["exact_any"] == False].copy()

    print(f"Total rows: {len(df)}")
    print(f"Failures exact_any=False: {len(failures)}")
    print()

    cols = ["term", "ground_truth", "prediction", "candidates", "semantic_similarity", "fuzzy_similarity"]

    for _, row in failures.head(args.top).iterrows():
        print("=" * 80)
        print(f"Term: {row['term']}")
        print(f"Ground truth: {row['ground_truth']}")
        print(f"Prediction: {row['prediction']}")
        print(f"Candidates: {row['candidates']}")
        print(f"Semantic similarity: {row['semantic_similarity']}")
        print(f"Fuzzy similarity: {row['fuzzy_similarity']}")


if __name__ == "__main__":
    main()