"""
Draw a fixed, seeded sample from topic_synonyms_clean.csv so every method is
evaluated on exactly the same rows. Re-running this script always produces
the same sample (seed=42) - that determinism is the point: results across
methods/sessions stay comparable.
"""

import argparse
from pathlib import Path

import pandas as pd

SEED = 42
SOURCE_PATH = Path(__file__).parent / "topic_synonyms_clean.csv"
OUTPUT_PATH = Path(__file__).parent / "eval_sample.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="sample size")
    args = parser.parse_args()

    df = pd.read_csv(SOURCE_PATH)
    sample = df.sample(n=args.n, random_state=SEED).sort_index().reset_index(drop=True)
    sample.to_csv(OUTPUT_PATH, index=False)

    print(f"Source rows: {len(df)}")
    print(f"Sampled rows: {len(sample)} (seed={SEED})")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
