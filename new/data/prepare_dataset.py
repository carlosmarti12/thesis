from datasets import load_dataset
import pandas as pd
from pathlib import Path


DATASET_NAME = "jensjorisdecorte/TU-Expert-Collection-Topic-Synonyms"
OUTPUT_PATH = Path("./topic_synonyms_clean.csv")


def clean_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value == "" or value == "-":
        return None
    return value


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()

    print("Original columns:", df.columns.tolist())
    print("Original rows:", len(df))

    df = df[["topic", "en", "en_synonym"]].copy()

    df["en"] = df["en"].apply(clean_text)
    df["en_synonym"] = df["en_synonym"].apply(clean_text)

    df = df.dropna(subset=["en", "en_synonym"])
    df = df.drop_duplicates(subset=["en", "en_synonym"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Clean rows:", len(df))
    print(f"Saved to {OUTPUT_PATH}")
    print(df.head(10))


if __name__ == "__main__":
    main()