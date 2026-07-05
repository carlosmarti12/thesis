from datasets import load_dataset
import pandas as pd

DATASET_NAME = "jensjorisdecorte/TU-Expert-Collection-Topic-Synonyms"


def load_grouped() -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME)
    df = ds["train"].to_pandas()[["topic", "en", "en_synonym"]].copy()
    df["en"] = df["en"].astype(str).str.strip()
    df["en_synonym"] = df["en_synonym"].astype(str).str.strip()
    df = df[(df["en"] != "-") & (df["en_synonym"] != "-") & (df["en"] != "") & (df["en_synonym"] != "")]
    df = df.drop_duplicates()
    grouped = (
        df.groupby(["topic", "en"], sort=False)["en_synonym"]
        .apply(list).reset_index()
        .rename(columns={"en_synonym": "en_synonyms"})
    )
    return grouped
