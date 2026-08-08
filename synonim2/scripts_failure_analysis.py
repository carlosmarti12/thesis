"""
Desglose de los 172 casos que no aciertan rank1 en el dataset completo
(969 filas, `llm_expansion_weighted_t0_rerank_qwen`), pedido por el asesor
2026-07-22: separar "retrieval miss" (gold fuera del top-5 - el reranker no
puede arreglarlo, ver `in_top5=False`) de "ranking error" (gold en el top-5
pero no en rank1 - el reranker sí lo vio pero lo ordenó mal), y desglosar
ambos por tipo de término (jerga de dominio / palabra corta vs frase larga -
mismas categorías heurísticas de scripts_error_breakdown.py, reutilizadas
para no duplicar criterios entre análisis).

No se intenta una heurística automática de polisemia (necesitaría
WordNet/nltk, no instalado en este entorno - ver memoria
`env-pip-install-reverts`) - en su lugar, los ejemplos impresos de cada
categoría se usan para inspección manual/cualitativa de polisemia y jerga,
que es lo que de verdad puede identificar un lector humano.

Requiere results/eval/error_analysis_rerank_qwen_full.csv (generado por el
run `llm_expansion_weighted_t0_rerank_qwen --split full`, ver README.md).
"""

import json

import pandas as pd

df = pd.read_csv("results/eval/error_analysis_rerank_qwen_full.csv")


def categorize(term: str) -> str:
    words = str(term).split()
    if len(words) == 1:
        token = words[0]
        if token.isupper() or len(token) <= 5:
            return "abbreviation_like"
        return "single_word"
    if "-" in str(term) or len(words) == 2:
        return "compound"
    return "multi_word_phrase"


df["category"] = df["term"].astype(str).apply(categorize)
df["failure_type"] = "correct"
df.loc[(df["in_top5"]) & (~df["hit@1"]), "failure_type"] = "ranking_error"
df.loc[~df["in_top5"], "failure_type"] = "retrieval_miss"

print("=== Failure type x term category (full dataset, n=969) ===\n")
pivot = pd.crosstab(df["category"], df["failure_type"])
pivot["total"] = pivot.sum(axis=1)
for col in ("correct", "ranking_error", "retrieval_miss"):
    if col not in pivot.columns:
        pivot[col] = 0
pivot["ranking_error_rate"] = (pivot["ranking_error"] / pivot["total"]).round(3)
pivot["retrieval_miss_rate"] = (pivot["retrieval_miss"] / pivot["total"]).round(3)
pivot = pivot[["total", "correct", "ranking_error", "retrieval_miss",
               "ranking_error_rate", "retrieval_miss_rate"]]
print(pivot.to_string())

print("\n=== Gold-rank distribution among ranking errors (n=119) ===")
print(df.loc[df["failure_type"] == "ranking_error", "gold_rank"].value_counts().sort_index().to_string())

print("\n=== Sample retrieval-miss cases (gold never in top-5, n=53) - for manual jargon/polysemy inspection ===")
miss = df[df["failure_type"] == "retrieval_miss"][["term", "ground_truth", "candidates", "category"]]
for _, row in miss.head(20).iterrows():
    try:
        cands = json.loads(row["candidates"])
    except Exception:
        cands = row["candidates"]
    print(f"  [{row['category']}] {row['term']!r} -> gold {row['ground_truth']!r} | retrieved: {cands}")

print("\n=== Sample ranking-error cases at gold_rank=2 (closest misses, n={}) ===".format(
    int((df["failure_type"] == "ranking_error").sum() and (df.loc[df['failure_type']=='ranking_error','gold_rank']==2).sum())
))
rank2 = df[(df["failure_type"] == "ranking_error") & (df["gold_rank"] == 2)][
    ["term", "ground_truth", "prediction", "category"]
]
for _, row in rank2.head(15).iterrows():
    print(f"  [{row['category']}] {row['term']!r} -> gold {row['ground_truth']!r}, predicted {row['prediction']!r}")

out_cols = ["term", "ground_truth", "prediction", "category", "failure_type", "gold_rank", "rank"]
df[out_cols].to_csv("results/eval/failure_analysis_full_by_category.csv", index=False)
pivot.to_csv("results/eval/failure_analysis_full_summary.csv")
print("\nSaved: results/eval/failure_analysis_full_by_category.csv, results/eval/failure_analysis_full_summary.csv")
