from datasets import load_dataset
import pandas as pd


DATASET_NAME = "jensjorisdecorte/TU-Expert-Collection-Topic-Synonyms"


def load_synonym_dataset() -> pd.DataFrame:
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


if __name__ == "__main__":
    df = load_synonym_dataset()
    print(df.head())
    print("Rows:", len(df))