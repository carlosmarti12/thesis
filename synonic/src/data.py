"""
Dataset loading. Mirrors second/src/data.py's role (a single function that
returns the clean dataframe). run_experiment.py/run_comparison.py load the
full dataset and cap it with --limit; load_eval_sample() below is a separate,
fixed 100-row sample (data/eval_sample.csv, see data/make_eval_sample.py) kept
available for ad-hoc use but not called by either run script by default.

`topic` is read from the cached CSV (kept on disk for provenance) but dropped
before the dataframe is returned: no method here ever received `topic` as
input, and no summary broke it down by topic either, so it was pure passenger
metadata - removed per advisor feedback after confirming it made no
difference (there was nothing using it to make a difference in).
"""

from pathlib import Path

import pandas as pd

DATASET_NAME = "jensjorisdecorte/TU-Expert-Collection-Topic-Synonyms"
DATA_DIR = Path(__file__).parent.parent / "data"
CLEAN_CSV_PATH = DATA_DIR / "topic_synonyms_clean.csv"
EVAL_SAMPLE_PATH = DATA_DIR / "eval_sample.csv"


def load_full_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Return the full cleaned dataset (en, en_synonym only - nl/nl_synonym and
    topic dropped). Reads the cached CSV unless it's missing or
    force_download=True, in which case it re-downloads from HuggingFace (see
    data/prepare_dataset.py for the same logic as a script).
    """
    if not force_download and CLEAN_CSV_PATH.exists():
        df = pd.read_csv(CLEAN_CSV_PATH)
        return df[["en", "en_synonym"]]

    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()
    df = df[["topic", "en", "en_synonym"]].copy()

    df["en"] = df["en"].astype(str).str.strip()
    df["en_synonym"] = df["en_synonym"].astype(str).str.strip()
    df = df[
        (df["en"] != "-") & (df["en_synonym"] != "-") & (df["en"] != "") & (df["en_synonym"] != "")
    ]
    df = df.drop_duplicates(subset=["en", "en_synonym"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV_PATH, index=False)

    return df[["en", "en_synonym"]]


def load_eval_sample() -> pd.DataFrame:
    """Return the fixed, seeded 100-row evaluation sample (data/eval_sample.csv).
    Regenerate it with `python -m data.make_eval_sample` if missing."""
    if not EVAL_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"{EVAL_SAMPLE_PATH} not found - run `python data/make_eval_sample.py` first."
        )
    return pd.read_csv(EVAL_SAMPLE_PATH)


if __name__ == "__main__":
    df = load_full_dataset()
    print(df.head())
    print("Rows:", len(df))
