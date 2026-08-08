"""
Crea splits reproducibles train/val/test a partir de data/synonyms_clean.csv.

No existian splits en esta carpeta hasta ahora (todos los experimentos
previos evaluaban sobre el dataset completo o sobre un head(n) arbitrario).
Esto hace imposible tunear nada (modelo de embeddings, umbrales, prompts)
sin arriesgar overfitting al propio conjunto de evaluacion.

Particion por termino `en` UNICO (no por fila) con seed fijo, para que un
mismo termino de entrada no caiga partido entre dos splits distintos (hay 10
terminos `en` duplicados con dos `en_synonym` distintos - ver docstring de
detect_degenerate_rows). El vocabulario candidato (pool de en_synonym) sigue
construyendose del dataset COMPLETO en cada script, independientemente de
estos splits - los splits solo determinan que pares (term, gold) se usan
para tunear (val) o para la confirmacion final (test), nunca que candidatos
existen (eso no es leakage, ver memoria del proyecto).

Tambien excluye del train/val/test la unica fila degenerada donde
en == en_synonym ("modernisation" / "modernisation"): con exclude_terms=[term]
en embedding_retrieval.retrieve_top_k, esa fila es estructuralmente
imposible de acertar (el propio termino, que seria la respuesta correcta,
esta excluido a proposito del pool de busqueda). No es un bug del pipeline,
es una fila del dataset donde la "sinonimia" es identidad - se documenta y
se deja fuera de la evaluacion en vez de contaminar las metricas con un 0
garantizado que no dice nada del metodo.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "synonyms_clean.csv"
SPLITS_DIR = ROOT / "splits"

SEED = 42
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
# el resto (0.2) va a test


def normalize_simple(text: str) -> str:
    return str(text).strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)

    degenerate = df["en"].apply(normalize_simple) == df["en_synonym"].apply(normalize_simple)
    n_degenerate = int(degenerate.sum())
    if n_degenerate:
        print(f"Excluding {n_degenerate} degenerate row(s) where en == en_synonym: "
              f"{df.loc[degenerate, 'en'].tolist()}")
    df = df.loc[~degenerate].reset_index(drop=True)

    terms = sorted(df["en"].unique().tolist())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(terms)

    n_terms = len(terms)
    n_train = int(round(n_terms * TRAIN_FRAC))
    n_val = int(round(n_terms * VAL_FRAC))

    train_terms = set(terms[:n_train])
    val_terms = set(terms[n_train:n_train + n_val])
    test_terms = set(terms[n_train + n_val:])

    def assign(term):
        if term in train_terms:
            return "train"
        if term in val_terms:
            return "val"
        return "test"

    df["split"] = df["en"].apply(assign)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        out_path = SPLITS_DIR / f"{split}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"{split}: {len(split_df)} rows ({len(set(split_df['en']) )} unique terms) -> {out_path}")

    # "full" = train+val+test concatenated (todo el dataset limpio, sin la
    # fila degenerada) - SOLO para reportar el rendimiento final del método
    # ya elegido sobre todos los datos disponibles, nunca para tunear nada
    # (eso sigue siendo train/val, con test confirmado una única vez).
    full_df = df.drop(columns=["split"]).reset_index(drop=True)
    full_path = SPLITS_DIR / "full.csv"
    full_df.to_csv(full_path, index=False)
    print(f"full (train+val+test, reporting only): {len(full_df)} rows -> {full_path}")

    print(f"\nTotal terms: {n_terms} | seed: {args.seed}")
    print("Candidate pool (en_synonym vocabulary) is still built from the FULL "
          "dataset by every run_*.py script - these splits only govern which "
          "(term, gold) pairs are used for tuning (val) vs. final confirmation (test).")


if __name__ == "__main__":
    main()
