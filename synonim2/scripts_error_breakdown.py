"""
Desglose de errores de llm_expansion_weighted_t0 en val por categoría de
término, para entender dónde se concentra el gap entre hit@1 y hit@3/5 (el
objetivo principal del ciclo 2026-07-20) y para justificar el umbral de
Fase C.5 (rerank condicionado por confianza).

Categorías (heurísticas simples, no requieren NLP adicional):
- "abbreviation_like": un solo token, todo mayúsculas o <=5 caracteres.
- "compound": contiene un guion, o exactamente dos palabras de contenido.
- "multi_word_phrase": 3 o más palabras.
- "single_word": el resto (una palabra normal, no abreviatura).

Requiere que results/eval/llm_expansion_weighted_t0__val.csv ya exista (ver
run_eval.py --method llm_expansion_weighted_t0 --split val).
"""

import json

import pandas as pd


def categorize(term: str) -> str:
    words = term.split()
    if len(words) == 1:
        token = words[0]
        if token.isupper() or len(token) <= 5:
            return "abbreviation_like"
        return "single_word"
    if "-" in term or len(words) == 2:
        return "compound"
    return "multi_word_phrase"


df = pd.read_csv("results/eval/llm_expansion_weighted_t0__val.csv")
df["category"] = df["term"].astype(str).apply(categorize)

# gold en top3-5 pero no en rank1: exactamente el gap que se quiere cerrar.
df["gap_case"] = df["hit@3"] & ~df["hit@1"]


def rank_of_gold(row) -> int | None:
    try:
        candidates = json.loads(row["candidates"])
    except Exception:
        return None
    gt = row["ground_truth"]
    for i, c in enumerate(candidates, start=1):
        if c == gt:
            return i
    return None


df["gold_rank"] = df.apply(rank_of_gold, axis=1)

summary = df.groupby("category").agg(
    n=("term", "count"),
    hit_at_1=("hit@1", "mean"),
    hit_at_3=("hit@3", "mean"),
    hit_at_5=("hit@5", "mean"),
    n_gap_cases=("gap_case", "sum"),
).reset_index()
summary["gap_rate"] = summary["n_gap_cases"] / summary["n"]

print(summary.to_string(index=False))
print(f"\nTotal gap cases (gold in top3-5, not rank1): {df['gap_case'].sum()} / {len(df)}")
print("\nGold rank distribution among gap cases:")
print(df[df["gap_case"]]["gold_rank"].value_counts().sort_index().to_string())

summary.to_csv("results/eval/error_breakdown_val.csv", index=False)
df[["term", "ground_truth", "prediction", "category", "gold_rank", "hit@1"]].to_csv(
    "results/eval/error_breakdown_val_detail.csv", index=False
)
