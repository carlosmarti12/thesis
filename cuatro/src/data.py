from datasets import load_dataset
import pandas as pd
from typing import List


DATASET_NAME = "jensjorisdecorte/TU-Expert-Collection-Topic-Synonyms"


def load_synonym_dataset() -> pd.DataFrame:
    """Load the raw dataset, one row per (term, synonym) pair."""
    ds = load_dataset(DATASET_NAME)
    df = ds["train"].to_pandas()
    df = df[["topic", "en", "en_synonym"]].copy()
    df["en"] = df["en"].astype(str).str.strip()
    df["en_synonym"] = df["en_synonym"].astype(str).str.strip()
    df = df[
        (df["en"] != "-")
        & (df["en_synonym"] != "-")
        & (df["en"] != "")
        & (df["en_synonym"] != "")
    ]
    df = df.drop_duplicates()
    return df


def load_synonym_dataset_grouped() -> pd.DataFrame:
    """
    Load the dataset grouped by (topic, en) so that each row has a list of
    all known ground-truth synonyms for that term.

    Returns a DataFrame with columns: topic, en, en_synonyms (List[str]).
    Running the pipeline once per unique term is more efficient and makes
    evaluation fair — a prediction that matches any known synonym counts as correct.
    """
    df = load_synonym_dataset()
    grouped = (
        df.groupby(["topic", "en"], sort=False)["en_synonym"]
        .apply(list)
        .reset_index()
        .rename(columns={"en_synonym": "en_synonyms"})
    )
    return grouped


if __name__ == "__main__":
    raw = load_synonym_dataset()
    print("Raw rows:", len(raw))
    print(raw.head())

    grouped = load_synonym_dataset_grouped()
    print("\nGrouped rows:", len(grouped))
    print(grouped.head())
    multi = grouped[grouped["en_synonyms"].apply(len) > 1]
    print(f"\nTerms with >1 synonym: {len(multi)}")
    print(multi.head())
